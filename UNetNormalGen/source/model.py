import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Add, Conv2DTranspose, MaxPooling2D, Activation, concatenate, BatchNormalization

class Model(object):
    """docstring for unet"""
    def __init__(self):
        self._name = 'unet_resnetblocks'
        self._encoder_layers = {}
        self._decoder_layers = {}
        self._arch = [(16, 2), (64, 3), (256, 3), (512, 3)]
        self._arch_decoder = [(256, 3), (64, 3), (16, 2), (8, 1)]

    def own_conv2d(self, incep, args):
        if incep:
            return self.incp_block(args[0], args[1])
        return self.convolution_block(args[0], args[1], args[2])

    def convolution_block(self, x, filters, size, strides=(1,1), padding='same', activation=True):
        x = Conv2D(filters, size, strides=strides, padding=padding, activation=None)(x)
        if activation:
            x = Activation(tf.nn.relu)(x)
        return x

    def incp_block(self, blockInput, num_filters):
        layers_list = []
        for dilation, fac_filter in [(1,0.5),(2,0.25),(3, 0.25)]:
            x = Conv2D(int(fac_filter * num_filters), (3,3), dilation_rate=[dilation, dilation], padding='same', activation=None)(blockInput)
            layers_list.append(x)

        x = concatenate(layers_list)
        x = Activation(tf.nn.relu)(x)
        return x

    def residual_block(self, blockInput, num_filters=16, dilation=True):
        x = blockInput
        if blockInput.shape[3] != num_filters:
            blockInput = Conv2D(num_filters, (1, 1), activation=tf.nn.relu, padding="same")(x)
        x = self.own_conv2d(dilation, [x, num_filters, (3,3)])
        x = self.own_conv2d(dilation, [x, num_filters, (3,3)])
        x = Add()([x, blockInput])
        return x

    def mini_encoder_block(self, input_layer, filters, res_blocks, divide=True):
        if divide:
            input_layer = MaxPooling2D((2, 2))(input_layer)
        for i in range(res_blocks):
            if filters > 250:
                input_layer = self.residual_block(input_layer, filters, dilation=False)
            else:
                input_layer = self.residual_block(input_layer, filters)
        return input_layer

    def mini_decoder_block(self, input_layer, filters, res_blocks, skipper=False):
        deconv = Conv2DTranspose(filters, (3,3), strides=(2, 2), padding='same', activation=tf.nn.relu)(input_layer)
        if skipper:
            deconv = concatenate([deconv, self._encoder_layers[str(filters)]])
            deconv = Conv2D(filters, (1,1), activation=tf.nn.relu)(deconv)
        for i in range(res_blocks):
            deconv = self.residual_block(deconv, filters, dilation=False)
        return deconv

    # Build model
    def create(self, input_layer, growth_rate=8):
        # Expand input channels 3->8
        x = Conv2D(8, (1, 1), activation=tf.nn.relu, padding="same")(input_layer)
        self._encoder_layers['8'] = x
        for filters, blocks in self._arch:
            x = self.mini_encoder_block(x, filters, blocks)
            self._encoder_layers[str(filters)] = x

        for filters, blocks in self._arch_decoder:
            x = self.mini_decoder_block(x, filters, blocks, True)
            self._decoder_layers[str(filters)] = x

        #self.visualize_layer_output(self._decoder_layers['64'], 'layer with 64 filters')
        #self.visualize_layer_output(self._decoder_layers['16'], 'layer with 16 filters')

        x = Conv2D(3, (1, 1), activation=None, padding="same")(x)

        self._last_layer = tf.nn.l2_normalize(x, axis=3)
        tf.summary.image('result image', (self._last_layer + 1.) / 2. * 255.)
        return self._last_layer

    def visualize_layer_output(self, layer, name):
        image = Conv2D(3, (1, 1), activation=tf.nn.tanh, strides=(1,1), padding="same")(layer)
        tf.summary.image(name, (image + 1.) / 2.)

    def get_results(self):
        return [self._last_layer, self._decoder_layers['16'], self._decoder_layers['64'], self._decoder_layers['256']]

    def compile(self, lr, loss):
        optimizer = tf.train.AdamOptimizer(lr)
        op = optimizer.minimize(loss)
        return op, loss
