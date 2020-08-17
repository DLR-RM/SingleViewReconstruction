import os

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from math import ceil

class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1, 1), **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)
        self.padding = tuple(padding)

    def call(self, x, mask=None):
        w_pad, h_pad, d_pad = self.padding
        y = tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [d_pad, d_pad], [0, 0]], "SYMMETRIC")
        return y

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3] + 2 * self.padding[2], input_shape[4]


class Autoencoder:
    def __init__(self, config, dataset):
        """
        Args:
            config: The configuration object.
            dataset: The dataset object which is used to get different information about the input.
        """
        self.dataset = dataset
        self.input_shape = [None] * 3 + [1]
        self.truncation_threshold = dataset.truncation_threshold
        self.model_config = config.model
        self.optimizer_config = config.optimizer

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=tf_config)
        tf.keras.backend.set_session(self.sess)
        self.encoder = tf.keras.models.Sequential()
        self.decoder = tf.keras.models.Sequential()

        self._build_model()

        self.train_iterator = None
        self.validation_iterator = None
        self.eval_iterator = None

        self.train_loss = None
        self.train_step = None
        self.validation_loss = None
        self.eval_preds = None
        self.eval_latent = None
        self.eval_latent_placeholder = None
        self.eval_preds_from_latent = None
        self.eval_label = None
        self.eval_block_index = None

        self.eval_input = None
        self.eval_placeholder = None
        self.eval_latent_from_placeholder = None
        self.eval_latent_mask_placeholder = None
        self.eval_preds_from_placeholder = None

    def set_iterators(self, train_iterator=None, validation_iterator=None, eval_from_input_iterator=None, eval_from_latent_iterator=None, eval_from_placeholder=False, eval_uses_fast_inference=False):
        """ Builds the computational graph to every corresponding given iterator.

        Args:
            train_iterator: Dataset Iterator
            validation_iterator: Dataset Iterator
            eval_iterator: Dataset Iterator
        """
        if train_iterator is not None:
            self._set_train_iterator(train_iterator)

        if validation_iterator is not None:
            self._set_validation_iterator(validation_iterator)

        if eval_from_input_iterator is not None:
            self._set_eval_from_input_iterator(eval_from_input_iterator, eval_uses_fast_inference)

        if eval_from_latent_iterator is not None:
            self._set_eval_from_latent_iterator(eval_from_latent_iterator)
            
        if eval_from_placeholder:
            self._set_eval_from_placeholder()            

        self.sess.run(tf.global_variables_initializer())

    def _set_train_iterator(self, train_iterator):
        """ Builds the training model which uses the given iterator as input and produces predictions.
        Those predictions are then used to build the loss function and the optimizer.

        Args:
            train_iterator: Dataset Iterator
        """
        self.train_iterator = train_iterator
        self.train_input, self.train_label, self.train_latent_mask = self.train_iterator.get_next()
        self.train_preds = self._model_graph(self.train_input, self.train_latent_mask)

        if self.optimizer_config.get_string("loss") == "L1":
            self.train_loss = tf.reduce_mean(tf.keras.losses.mean_absolute_error(self.train_label, self.train_preds))
        elif self.optimizer_config.get_string("loss") == "weighted_L1":
            dist = tf.distributions.Normal(loc=0.0, scale=self.truncation_threshold / 4)

            train_label = tf.reshape(self.train_label, [-1, self.dataset.output_size() ** 3])
            train_preds = tf.reshape(self.train_preds, [-1, self.dataset.output_size() ** 3])

            densities = dist.prob(train_label)
            diff = tf.abs(train_preds - train_label)

            batch_topk_factor = self.optimizer_config.get_float("batch_topk_loss")
            if batch_topk_factor > 0:
                samplewise_diff = tf.reduce_mean(diff, [1])
                _, indices = tf.nn.top_k(samplewise_diff, int(batch_topk_factor * self.dataset.batch_size))

                diff = tf.gather(diff, indices)
                densities = tf.gather(densities, indices)

            topk_factor = self.optimizer_config.get_float("topk_loss")
            if topk_factor > 0:
                diff, indices = tf.nn.top_k(diff, int(topk_factor * train_preds.shape[-1].value))
                densities = tf.batch_gather(densities, indices)

            #diff = tf.Print(diff, [tf.shape(diff), tf.shape(densities)], summarize=5)

            self.train_loss = tf.reduce_mean(diff + diff * densities * 4 * self.truncation_threshold)
        else:
            raise LookupError("No loss with name " + self.optimizer_config.get_float("loss"))

        self.train_step = tf.train.AdamOptimizer(self.optimizer_config.get_float("lr")).minimize(self.train_loss)

    def _set_validation_iterator(self, validation_iterator):
        """ Builds the validation model which uses the given iterator as input and produces predictions.
        Those predictions are then just used to calculate the validation loss.

        Args:
            validation_iterator: Dataset Iterator
        """
        self.validation_iterator = validation_iterator
        validation_input, validation_label, validation_latent_mask = self.validation_iterator.get_next()
        validation_preds = self._model_graph(validation_input, validation_latent_mask)

        self.validation_loss = tf.reduce_mean(tf.keras.losses.mean_absolute_error(validation_label, validation_preds))

    def _set_eval_from_input_iterator(self, eval_iterator, eval_uses_fast_inference):
        """ Builds the evaluation models, which allows various ways to decode/encode data.

       Args:
           eval_iterator: Dataset Iterator
       """
        self.eval_iterator = eval_iterator
        if eval_uses_fast_inference:
            self.eval_input, self.eval_latent_mask, self.eval_path = self.eval_iterator.get_next()
        else:
            self.eval_input, self.eval_label, self.eval_latent_mask = self.eval_iterator.get_next()

        self.eval_latent = self.encoder(self.eval_input)

        # Decode the latent representation again by using the user given latent mask
        self.eval_preds = self.decoder(self.eval_latent * self.eval_latent_mask)

    def _set_eval_from_latent_iterator(self, eval_iterator):
        """ Builds the evaluation models, which allows various ways to decode/encode data.

       Args:
           eval_iterator: Dataset Iterator
       """
        self.eval_iterator = eval_iterator
        self.eval_latent_input, self.eval_latent_mask, self.eval_path = self.eval_iterator.get_next()
        self.eval_preds_from_latent = self.decoder(self.eval_latent_input)

    def _set_eval_from_placeholder(self):
        """ Builds the evaluation models, which allows various ways to decode/encode data.

       Args:
           eval_iterator: Dataset Iterator
       """

        self.eval_placeholder = tf.placeholder(tf.float32, shape=[None] + self.input_shape)
        self.eval_latent_mask_placeholder = tf.placeholder(tf.float32, shape=[None] * 4 + [1])

        self.eval_latent_from_placeholder = self.encoder(self.eval_placeholder)

        # Decode the latent representation again by using the user given latent mask
        self.eval_preds_from_placeholder = self.decoder(self.eval_latent_from_placeholder * self.eval_latent_mask_placeholder)
        
    def _model_graph(self, input, latent_mask):
        """ Builds the autoencoder graph. The only difference to
        the usual architecture that the latent tensor is multiplied by a latent mask.
        This mask makes sure to set encoded blocks to zero which just act as a padding at the
        border of the voxelmap.

        Args:
            input: Tensor
            latent_mask: Tensor

        Returns:
            The output tensor of the autoencoder.
        """
        latent = self.encoder(input)
        latent *= latent_mask
        return self.decoder(latent)

    def _build_model(self):
        """ Builds the keras model and creates parameters.

        """
        self._build_encoder()
        self._build_decoder()

    def _build_encoder(self):
        """ Builds the encoder which just consists of consecutive conv/pool layers.
        The output shape of the last layer will determine the latent shape of the autoencoder.
        """
        default_conf_prefix = "encoder/conv/default"
        for i in range(self.model_config.get_int("encoder/conv/total")):
            conf_prefix = "encoder/conv/" + str(i)
            self._build_conv_layer(self.encoder, conf_prefix, default_conf_prefix, self.input_shape)
            self._build_pool_layer(self.encoder, conf_prefix, default_conf_prefix)

        self.latent_shape = list(self.encoder.layers[-1].output_shape[1:])

    def _build_decoder(self):
        """  Builds the decoder which consists of consecutive conv/upsampling layers.
        """
        default_conf_prefix = "decoder/conv/default"
        #self.decoder.add(ReflectionPadding2D(input_shape=self.latent_shape))
        #self.decoder.add(ReflectionPadding2D(input_shape=self.latent_shape))
        for i in range(self.model_config.get_int("decoder/conv/total")):
            conf_prefix = "decoder/conv/" + str(i)
            self._build_conv_layer(self.decoder, conf_prefix, default_conf_prefix, self.latent_shape)
            self._build_upsampling_layer(self.decoder, conf_prefix, default_conf_prefix)
            self._build_pool_layer(self.decoder, conf_prefix, default_conf_prefix)

        kernel_size = self.model_config.get_int("decoder/output/kernel_size")
        if self.model_config.get_string("decoder/output/padding") == "same":
            self.decoder.add(ReflectionPadding2D())

        self.decoder.add(tf.keras.layers.Conv3D(1, (kernel_size, kernel_size, kernel_size), activation='linear', padding='valid'))
        self.decoder.add(tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, -self.truncation_threshold, self.truncation_threshold)))

    def _build_conv_layer(self, model, conf_prefix, default_conf_prefix, input_shape):
        """ Builds a conv layer.

        Args:
            model: The model where to add the new layer.
            conf_prefix: The config path where this conv layer is configured.
            default_conf_prefix: The config path where the default configuration for conv layers can be found. Will be used as fallback.
            input_shape: The input shape, if this is the first layer.
        """
        kernel_size = self.model_config.get_int(conf_prefix + "/kernel_size", default_conf_prefix + "/kernel_size")
        if self.model_config.get_string(conf_prefix + "/padding", default_conf_prefix + "/padding") == "same":
            model.add(ReflectionPadding2D(input_shape=input_shape))

        model.add(tf.keras.layers.Conv3D(self.model_config.get_int(conf_prefix + "/channels"), (kernel_size, kernel_size, kernel_size), input_shape=input_shape, activation=self.model_config.get_string(conf_prefix + "/activation", default_conf_prefix + "/activation"), padding="valid"))

        if self.model_config.get_bool(conf_prefix + "/batch_normalization", default_conf_prefix + "/batch_normalization"):
            model.add(tf.keras.layers.BatchNormalization())

    def _build_pool_layer(self, model, conf_prefix, default_conf_prefix):
        """ Builds a pool layer.

        Args:
            model: The model where to add the new layer.
            conf_prefix: The config path where this pool layer is configured.
            default_conf_prefix: The config path where the default configuration for pool layers can be found. Will be used as fallback.
        """
        if self.model_config.get_bool(conf_prefix + "/pooling_enabled", default_conf_prefix + "/pooling_enabled"):
            if self.model_config.get_bool(conf_prefix + "/learned_pool_upsampling", default_conf_prefix + "/learned_pool_upsampling"):
                model.add(tf.keras.layers.Conv3D(self.model_config.get_int(conf_prefix + "/channels"), (2, 2, 2), strides=2, padding='valid'))
            else:
                model.add(tf.keras.layers.MaxPooling3D((2, 2, 2), padding='same'))


    def _build_upsampling_layer(self, model, conf_prefix, default_conf_prefix):
        """ Builds an upsampling layer.

       Args:
           model: The model where to add the new layer.
           conf_prefix: The config path where this upsampling layer is configured.
           default_conf_prefix: The config path where the default configuration for upsampling layers can be found. Will be used as fallback.
       """
        if self.model_config.get_bool(conf_prefix + "/upsampling_enabled", default_conf_prefix + "/upsampling_enabled"):
            if self.model_config.get_bool(conf_prefix + "/learned_pool_upsampling", default_conf_prefix + "/learned_pool_upsampling"):
                model.add(tf.keras.layers.Conv3DTranspose(self.model_config.get_int(conf_prefix + "/channels"), (2, 2, 2), strides=2, use_bias=False))
            else:
                model.add(tf.keras.layers.UpSampling3D((2, 2, 2)))

    def train(self, evaluate=False, train_iterations=10, validation_iterations=100, eval_iterations=1):
        """ Trains and evaluates the network for the given amount of iterations.

        Args:
            evaluate: If false, validation and evaluation will be skipped.
            train_iterations:
            validation_iterations:
            eval_iterations:

        Returns:
            dict: The stats generated by train/validation/evaluation.
        """
        stats = {}
        self._run_training(train_iterations, stats)

        if evaluate:
            self._run_validation(validation_iterations, stats)
            self._run_evaluation(eval_iterations, stats)

        return stats

    def _run_training(self, train_iterations, stats):
        """ Trains the network for the given amount of iterations.

        Args:
            train_iterations: The number of iterations
            stats: The dict where to put in stats.
        """
        total_loss = 0
        for i in range(train_iterations):
            _, loss = self.sess.run([self.train_step, self.train_loss], feed_dict={K.learning_phase(): 1})

            total_loss += loss
        stats['loss'] = total_loss / train_iterations if train_iterations > 0 else 0

    def _run_validation(self, validation_iterations, stats):
        """ Validates the network for the given amount of iterations.

        Args:
            validation_iterations: The number of iterations
            stats: The dict where to put in stats.
        """
        total_loss = 0

        for i in range(validation_iterations):
            total_loss += self.sess.run([self.validation_loss], feed_dict={K.learning_phase(): 0})[0]

        stats['val_loss'] = total_loss / validation_iterations if validation_iterations > 0 else 0

    def _run_evaluation(self, eval_iterations, stats):
        """ Evaluates the network for the given amount of iterations.

        Args:
            eval_iterations: The number of iterations
            stats: The dict where to put in stats.
        """
        stats['eval_l1'] = 0
        stats['eval_weighted_l1'] = 0
        stats['eval_l1_boundary'] = 0
        stats['iou'] = 0
        number_of_samples = 0

        for e in range(eval_iterations):
            output, ground_truth = self.predict_not_split_input(True)
            output = np.squeeze(output, axis=-1)
            ground_truth = np.squeeze(ground_truth, axis=-1)

            for i in range(output.shape[0]):
                stats['eval_l1'] += np.mean(np.abs(output[i] - ground_truth[i]))

                boundary_mask = np.logical_or(np.abs(output[i]) < self.truncation_threshold, np.abs(ground_truth[i]) < self.truncation_threshold)
                stats['eval_l1_boundary'] += np.sum(np.abs(output[i] - ground_truth[i]) * boundary_mask) / boundary_mask.sum()

                intersection = np.sum((output[i] < 0) * (ground_truth[i] < 0))
                total = np.sum(output[i] < 0) + np.sum(ground_truth[i] < 0)
                stats['iou'] += intersection.astype(np.float) / (total - intersection)

                number_of_samples += 1
            break
        stats['eval_l1'] /= number_of_samples
        stats['eval_weighted_l1'] /= number_of_samples
        stats['eval_l1_boundary'] /= number_of_samples
        stats['iou'] /= number_of_samples

    def save(self, path=''):
        """ Saves the network weights in the given directory.
        Decoder weights will be stored in file called "decoder.h5",
        encoder weights will be stored in "encoder.h5".

        Args:
            path:
        """
        if not os.path.exists(path):
            os.makedirs(path)
        self.encoder.save_weights(os.path.join(path, "encoder.h5"))
        self.decoder.save_weights(os.path.join(path, "decoder.h5"))

    def load(self, path=''):
        """ Loads the network weights from the given directory.

        Args:
            path:
        """
        self.encoder.load_weights(os.path.join(path, "encoder.h5"))
        self.decoder.load_weights(os.path.join(path, "decoder.h5"))

    def predict_not_split_input(self, return_label=False):
        """ Encodes and decodes the given input and returns the result.

        Args:
            input: A numpy array which contains a "whole"/"non-split" voxelmap.
            focused_blocks: The number of focused blocks which should be encoded per forward pass. The higher, the less redundant computation has to be done.

        Returns:
            The output as a numpy array. Has the same resolution as the given input.
        """

        window_positions = self.dataset.possible_input_window_positions()
        if len(window_positions) % self.dataset.eval_batch_size != 0:
            raise SystemError()

        output = None
        label = None
        for i in range(len(window_positions) // self.dataset.eval_batch_size):
            if return_label:
                batch_output, batch_label = self.sess.run([self.eval_preds, self.eval_label], feed_dict={K.learning_phase(): 0})
                if label is None:
                    label = batch_label
                else:
                    label = np.concatenate((label, batch_label), 0)
            else:
                batch_output = self.sess.run(self.eval_preds, feed_dict={K.learning_phase(): 0})

            if output is None:
                output = batch_output
            else:
                output = np.concatenate((output, batch_output), 0)

        for i, splits in enumerate([self.dataset.data_size // self.dataset.output_size(), self.dataset.data_size // self.dataset.output_size(), self.dataset.data_size // self.dataset.output_size()]):
            output = np.concatenate(np.split(output, splits, 0), i + 1)

        if return_label:
            for i, splits in enumerate([self.dataset.data_size // self.dataset.output_size(), self.dataset.data_size // self.dataset.output_size(), self.dataset.data_size // self.dataset.output_size()]):
                label = np.concatenate(np.split(label, splits, 0), i + 1)

        if return_label:
            return output, label
        else:
            return output

    def encode_not_split_input(self, batch_container, empty_block_latent, full_block_latent):
        """ Encodes the given input and returns the compressed result. """

        indices, path = self.sess.run([self.eval_latent_mask, self.eval_path], feed_dict={K.learning_phase(): 0})

        number_of_ambiguous_blocks = (indices == 0).sum()
        print("Calc " + str(number_of_ambiguous_blocks) + " blocks")
        output = None
        for i in range(int(ceil(float(number_of_ambiguous_blocks) / self.dataset.batch_size))):
            batch_output = self.sess.run(self.eval_latent, feed_dict={K.learning_phase(): 0})

            if output is None:
                output = batch_output
            else:
                output = np.concatenate((output, batch_output), 0)

        batch_container[indices == 1] = empty_block_latent
        batch_container[indices == -1] = full_block_latent
        batch_container[indices == 0] = output

        output = batch_container

        for i, splits in enumerate([self.dataset.data_size // self.dataset.output_size(), self.dataset.data_size // self.dataset.output_size(), self.dataset.data_size // self.dataset.output_size()]):
            output = np.concatenate(np.split(output, splits, 0), i + 1)

        # Cut away the additional border which is normally "eaten up" by the valid conv layers in the decoder.
        return output[:, self.dataset.additional_blocks:output.shape[1] - self.dataset.additional_blocks, self.dataset.additional_blocks:output.shape[2] - self.dataset.additional_blocks, self.dataset.additional_blocks:output.shape[3] - self.dataset.additional_blocks, :], path

    def decode_not_split_input(self, batch_container):
        """ Decodes the given encoded input and returns the raw voxelgrid. """

        indices, path = self.sess.run([self.eval_latent_mask, self.eval_path], feed_dict={K.learning_phase(): 0})

        number_of_ambiguous_blocks = (indices == 0).sum()
        print("Calc " + str(number_of_ambiguous_blocks) + " blocks")
        output = None
        for i in range(int(ceil(float(number_of_ambiguous_blocks) / self.dataset.batch_size))):
            batch_output = self.sess.run(self.eval_preds_from_latent, feed_dict={K.learning_phase(): 0})

            if output is None:
                output = batch_output
            else:
                output = np.concatenate((output, batch_output), 0)

        batch_container[indices == 1] = self.dataset.truncation_threshold
        batch_container[indices == -1] = -self.dataset.truncation_threshold
        batch_container[indices == 0] = output

        output = batch_container
        for i, splits in enumerate([self.dataset.data_size // self.dataset.output_size(),
                                    self.dataset.data_size // self.dataset.output_size(),
                                    self.dataset.data_size // self.dataset.output_size()]):
            output = np.concatenate(np.split(output, splits, 0), i + 1)

        return output, path

    def encode_from_placeholder(self, input):
        return self.sess.run(self.eval_latent_from_placeholder,  feed_dict={K.learning_phase(): 0, self.eval_placeholder: input})

    def summary(self, print_fn=None):
        """ Prints the model summary

        Args:
            print_fn:
        """
        self.encoder.summary(print_fn=print_fn)
        self.decoder.summary(print_fn=print_fn)
