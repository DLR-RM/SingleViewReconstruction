import tensorflow as tf
import numpy as np
import math


class PadLayer(tf.keras.layers.Layer):

    def __init__(self, padding, **kwargs):
        self.padding = padding
        super(PadLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PadLayer, self).build(input_shape)

    def call(self, input, **kwargs):
        w_pad, h_pad, d_pad = self.padding
        return tf.pad(input, [[0, 0], [0, 0], [w_pad, w_pad], [h_pad, h_pad], [d_pad, d_pad]], "REFLECT")

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2] + 2 * self.padding[0], input_shape[3] + 2 * \
               self.padding[1], input_shape[4] + 2 * self.padding[2]


class TreeModel(object):

    def __init__(self, input, settings):
        self._total_height = settings.height
        self._used_heights_new = []
        self._global_map_collector = {}
        self._result_size = settings.result_size
        self._activation_type = tf.nn.leaky_relu
        self._d_type = settings.d_type
        self._filters = settings.filters_for_level
        self._residual_levels = settings.residual_levels
        self._pool_levels = settings.pool_levels
        self._settings = settings
        input_structure = settings.input_structure
        self.checkpoints = []
        self._layer_elements = []
        for height in range(self._total_height + 1):
            self._layer_elements.append([None] * int(2 ** height))

        def float32_variable_storage_getter(getter, name, shape=None, dtype=None,
                                            initializer=None, regularizer=None,
                                            trainable=True,
                                            *args, **kwargs):
            """Custom variable getter that forces trainable variables to be stored in float32 precision and
            then casts them to the training precision.
			"""
            storage_dtype = tf.float32 if trainable else dtype
            variable = getter(name, shape, dtype=storage_dtype,
                              initializer=initializer, regularizer=regularizer,
                              trainable=trainable,
                              *args, **kwargs)
            if trainable and dtype != tf.float32:
                variable = tf.cast(variable, dtype)
            return variable

        # forces all variables to be stored as tf.float32
        with tf.variable_scope('fp32_storage', custom_getter=float32_variable_storage_getter):
            if len(settings.pool_levels) == 0:
                with tf.name_scope('size_reduction'):
                    current_layer = input
                    for counter, value in enumerate(input_structure):
                        if value != -1:  # not pooling
                            current_layer = self.normal_conv2d_act_batch(current_layer, value, 'layer_' + str(counter)
                                                                         + "_with_" + str(value) + '_intern')
                        else:
                            current_layer = self.max_pooling_2d(current_layer)
            else:
                current_layer = input
            with tf.name_scope('3D_graph'):
                result = self.recursive_create(input=current_layer, height=self._total_height, id=1)
            self._intermediate_tree_results = []
            for height, layer in enumerate(self._layer_elements):
                print("Height: " + str(height) + ", filter of the nodes: " + str([ele.shape[1].value for ele in layer]))
                self._intermediate_tree_results.append([])
                amount_of_filters = self._settings.amount_of_output_channels
                for node in layer:
                    intermediate_result = self.normal_conv2d_act_batch(node, amount_of_filters, 'reduce_' + str(height),
                                                                       filter_size=1)
                    self._intermediate_tree_results[-1].append(intermediate_result)
            self.last_layer = self.added_3D_convolution(input=result)

    def normal_conv2d_act_batch(self, input, amount_of_filters, name, filter_size=3):
        out = self.conv2d_layer(input, amount_of_filters, name, filter_size)
        out = self._activation_type(out)
        return out

    def add_conv2d_act_batch(self, input, add_element, amount_of_filters, name, filter_size=3):
        out = self.conv2d_layer(input, amount_of_filters, name, filter_size)
        out = self._activation_type(out + add_element)
        return out

    def recursive_create(self, input, height, id):
        current_height = self._total_height - height
        current_node_nr_in_layer = int(id - 2 ** current_height)
        print("\t" * current_height + "Id: {}, node: {}".format(id, current_node_nr_in_layer))
        self._layer_elements[current_height][current_node_nr_in_layer] = input
        if height > 0:
            amount_of_filters = self._filters[height - 1]
            res_level = self._residual_levels[height - 1]
            if height == 1:
                amount_of_nodes_in_this_layer = int(2 ** self._settings.height)
                individual_size = self._settings.result_size // amount_of_nodes_in_this_layer
                amount_of_filters = self._settings.amount_of_filters_in_first_3D * individual_size
            if height == 1 and amount_of_filters == 0:
                print("The amount of filters is zero, this can not be right!")
            if height not in self._used_heights_new:
                # print("Height: " + str(height) + ", filters: " + str(amount_of_filters) + ", res: " + str(res_level))
                self._used_heights_new.append(height)
            front_tensor, back_tensor = self.split_input(input, amount_of_filters, height, res_level)
            if height >= self._total_height - 1:  # 4 and 5
                label_name = 'height_' + str(height)
                self._global_map_collector[label_name + '_front'] = front_tensor
                self._global_map_collector[label_name + '_back'] = back_tensor
            if height in self._pool_levels:
                front_tensor = self.max_pooling_2d(front_tensor)
                back_tensor = self.max_pooling_2d(back_tensor)
            out = []
            out += self.recursive_create(front_tensor, height - 1, id * 2)
            out += self.recursive_create(back_tensor, height - 1, id * 2 + 1)
            return out
        else:
            return [input]

    def max_pooling_2d(self, input):
        return tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), data_format='channels_first')(input)

    def conv2d_layer(self, input, amount_of_filters, name, filter_size):
        if self._settings.regularizer_scale > 1e-13:
            kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=self._settings.regularizer_scale)
        else:
            kernel_regularizer = None
        with tf.name_scope(name + "_conv_layer"):
            if filter_size == 1:
                ret = tf.keras.layers.Conv2D(amount_of_filters, (filter_size, filter_size), padding="same",
                                             data_format="channels_first", kernel_regularizer=kernel_regularizer,
                                             bias_regularizer=kernel_regularizer)(input)
            else:
                inception_layers = []
                used_dil_values = [1, 2, 4][:amount_of_filters]  # to make sure only a certain amount of filters is used
                for dil_value in used_dil_values:
                    filters = int(np.max([amount_of_filters * 0.5 if dil_value == 1 else amount_of_filters * 0.25, 1.]))
                    current_input = input
                    with tf.name_scope(name + "_inception_" + str(dil_value)):
                        inception_layer = tf.keras.layers.Conv2D(filters, (filter_size, filter_size),
                                                                 padding="same",
                                                                 data_format="channels_first",
                                                                 dilation_rate=[dil_value, dil_value],
                                                                 kernel_regularizer=kernel_regularizer,
                                                                 bias_regularizer=kernel_regularizer)(current_input)
                    inception_layers.append(inception_layer)
                ret = tf.concat(inception_layers, axis=1)
            return ret

    def split_input(self, input, amount_of_filters, height, amount_of_residual_blocks):
        layer_name = 'height_' + str(height) + '_filters_' + str(amount_of_filters)
        with tf.name_scope('height_' + str(height)):
            if amount_of_residual_blocks > 0:
                first_out, second_out = self.create_resnet_block(input, input, amount_of_filters, height,
                                                                 layer_name, amount_of_residual_blocks)
            else:
                with tf.name_scope('height_' + str(height) + '_split'):
                    first_out = self.normal_conv2d_act_batch(input, amount_of_filters, layer_name + '_front')
                    second_out = self.normal_conv2d_act_batch(input, amount_of_filters,
                                                              layer_name + '_back')
        return first_out, second_out

    def create_resnet_block(self, next_input_first, next_input_second, amount_of_filters, height, layer_name,
                            amount_of_residual_blocks):
        if height == 1:
            used_amount_of_filters = self._settings.filters_in_deepest_node
        else:
            used_amount_of_filters = amount_of_filters
        if next_input_first.shape[1] != used_amount_of_filters:
            with tf.name_scope('height_' + str(height) + '_rescale_filter_size'):
                next_input_first = self.normal_conv2d_act_batch(next_input_first, used_amount_of_filters,
                                                                layer_name + '_front_resize', 1)
                next_input_second = self.normal_conv2d_act_batch(next_input_second, used_amount_of_filters,
                                                                 layer_name + '_back_resize', 1)
        front_name = layer_name + '_front_nr_'
        back_name = layer_name + '_back_nr_'
        for i in range(amount_of_residual_blocks):
            with tf.name_scope('height_' + str(height) + '_level_' + str(i)):
                first_out = self.normal_conv2d_act_batch(next_input_first, used_amount_of_filters,
                                                         front_name + str(i * 2))
                first_out = self.add_conv2d_act_batch(first_out, next_input_first, used_amount_of_filters,
                                                      front_name + str(i * 2 + 1))
                second_out = self.normal_conv2d_act_batch(next_input_second, used_amount_of_filters,
                                                          back_name + str(i * 2))
                second_out = self.add_conv2d_act_batch(second_out, next_input_second,
                                                       used_amount_of_filters, back_name + str(i * 2 + 1))
                next_input_first, next_input_second = first_out, second_out
        if height == 1:
            next_input_first = self.normal_conv2d_act_batch(next_input_first, amount_of_filters,
                                                            layer_name + '_front_resize_last', 1)
            next_input_second = self.normal_conv2d_act_batch(next_input_second, amount_of_filters,
                                                             layer_name + '_back_resize_last', 1)
        return next_input_first, next_input_second

    def generate_inception_3d_layer(self, input, kernel_size, dil_values, amount_of_filters, prefix):
        collection_dilated_layers = []
        if len(dil_values) > 1:
            for index, dil_value in enumerate(dil_values):
                name = prefix + "_" + str(dil_value)
                next_layer = self.generate_3d_conv(input=input, kernel_size=kernel_size,
                                                   output_filters=amount_of_filters[index], dil_value=dil_value,
                                                   act_type=self._activation_type, prefix=name)
                collection_dilated_layers.append(next_layer)
            next_layer = tf.concat(collection_dilated_layers, 1)
            return next_layer
        else:
            return self.generate_3d_conv(input, kernel_size, amount_of_filters[0], dil_values[0], self._activation_type,
                                         prefix)

    def generate_3d_conv(self, input, kernel_size, output_filters, dil_value, act_type, prefix):
        if self._settings.regularizer_scale > 1e-13:
            kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=self._settings.regularizer_scale)
        else:
            kernel_regularizer = None
        current_input = input
        padding_type = "same"
        if self._settings.use_reflective_padding_3D:
            padding_type = "valid"
            paddings = [int(math.floor(val * 0.5) * dil_value) for val in kernel_size]
            current_input = PadLayer(paddings)(input)
        dil_values = tuple([dil_value if kernel_val > 1 else 1 for kernel_val in kernel_size])
        return tf.keras.layers.Conv3D(filters=output_filters, kernel_size=kernel_size, padding=padding_type,
                                      data_format='channels_first', dilation_rate=dil_values, activation=act_type,
                                      use_bias=True, kernel_initializer=tf.glorot_normal_initializer(),
                                      kernel_regularizer=kernel_regularizer,
                                      bias_regularizer=kernel_regularizer, name=prefix)(current_input)

    def single_concat_2d_to_3d(self, input):
        res = tf.concat(input, 1)  # it is always one, 0 is batch size, 1 is the channel
        if len(res.shape) == 4:  # first concat -> batch size, feature maps, x, y
            return tf.transpose(res, (0, 2, 3, 1))  # new order: batch size, x, y, feature maps
        elif len(res.shape) == 5:  # second concat -> batch size, feature maps, x, y, z
            return res  # leave as is

    def concat_2d_layers_to_3d(self, input):
        if len(input) == 0:
            raise Exception("Something went wrong, no tree elements")
        with tf.name_scope('to3D'):
            # split them accordingly
            multi_dim_collection = []
            individual_size = int(self._settings.result_size // len(input))
            amount_of_new_channels = self._settings.amount_of_filters_in_first_3D
            for i in range(amount_of_new_channels):
                current_collection = []
                current_start = i * individual_size
                current_end = (i + 1) * individual_size
                for element in input:
                    current_collection.append(element[:, current_start:current_end, :, :])
                new_3d_channel = self.single_concat_2d_to_3d(current_collection)
                new_3d_channel = tf.cast(tf.expand_dims(new_3d_channel, 1), tf.float32)
                multi_dim_collection.append(new_3d_channel)
            self.layer_before_3D = self.single_concat_2d_to_3d(multi_dim_collection)
            self.checkpoints.append(self.layer_before_3D)
            return self.layer_before_3D

    def added_3D_convolution(self, input):
        set = self._settings
        output_channels = set.amount_of_output_channels
        with tf.name_scope('3D_convolution'):
            next_layer = self.concat_2d_layers_to_3d(input)
            self.rescaled_layer_before_3D = self.generate_3d_conv(self.layer_before_3D, kernel_size=(1, 1, 1),
                                                                  output_filters=output_channels, dil_value=1,
                                                                  act_type=None, prefix='rescaled_layer_before_3D')
            layer_nr = 0

            def convert_to_string(filter, dil):
                if len(filter) == 1:
                    return str(filter) + " filters"
                else:
                    return "(" + ", ".join([str(f) + " with dil " + str(d) for f, d in zip(filter, dil)]) + ")"

            for type in set.layer_structure_for_3d:
                text = "Add 3d-layer with "
                if type == 0:  # means reduced layer
                    for i in range(3):
                        if set.use_plane_for_separable_3d:
                            kernel_size = [3, 3, 3]
                            kernel_size[i] = 1
                        else:
                            kernel_size = [1, 1, 1]
                            kernel_size[i] = 3
                        kernel_size = tuple(kernel_size)
                        prefix_name = "filter_3D_" + str(layer_nr) + "_" + '_'.join([str(e) for e in kernel_size])
                        next_layer = self.generate_inception_3d_layer(next_layer, kernel_size,
                                                                      set.dil_values_for_3d_separable,
                                                                      set.filter_amounts_for_3d_separable,
                                                                      prefix_name)
                        layer_nr += 1
                    text += "3 layers with kernel "
                    text += convert_to_string(set.filter_amounts_for_3d_separable, set.dil_values_for_3d_separable)
                elif type == 1:  # mean normal layer
                    text += convert_to_string(set.filter_amounts_for_3d, set.dil_values_for_3d)
                    prefix_name = "filter_3D_" + str(layer_nr)
                    next_layer = self.generate_inception_3d_layer(next_layer, (3, 3, 3), set.dil_values_for_3d,
                                                                  set.filter_amounts_for_3d, prefix_name)
                    layer_nr += 1
                else:
                    print("Type is not known: " + str(type))
                print(text)
            print("Add last 3d-layer with " + str(output_channels) + " filters")
            return self.generate_3d_conv(next_layer, kernel_size=(3, 3, 3), output_filters=output_channels, dil_value=1,
                                         act_type=None, prefix='filter_back_to_one')

    def get_intermediate_inner_tree_results(self):
        return self._intermediate_tree_results
