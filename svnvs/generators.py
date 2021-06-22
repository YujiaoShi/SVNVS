import tensorflow as tf


def discrim_conv(batch_input, out_channels, stride):
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))



def gen_conv(batch_input, out_channels, separable_conv=False):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if separable_conv:
        return tf.layers.separable_conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)


def gen_deconv(batch_input, out_channels, separable_conv=False):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if separable_conv:
        _b, h, w, _c = batch_input.shape
        resized_input = tf.image.resize_images(batch_input, [h * 2, w * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.layers.separable_conv2d(resized_input, out_channels, kernel_size=4, strides=(1, 1), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)


def lrelu(x, a=0.2):
    with tf.name_scope("lrelu"):
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(inputs):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))



def create_discriminator(discrim_inputs, ndf=64):
    n_layers = 3
    layers = []

    # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
    with tf.variable_scope("layer_1", reuse=tf.AUTO_REUSE):
        convolved = discrim_conv(discrim_inputs, ndf, stride=2)
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)

    # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
    # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1), reuse=tf.AUTO_REUSE):
            out_channels = ndf * min(2**(i+1), 8)
            stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
            convolved = discrim_conv(layers[-1], out_channels, stride=stride)
            normalized = batchnorm(convolved)
            rectified = lrelu(normalized, 0.2)
            layers.append(rectified)

    # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    with tf.variable_scope("layer_%d" % (len(layers) + 1), reuse=tf.AUTO_REUSE):
        convolved = discrim_conv(rectified, out_channels=1, stride=1)
        output = tf.sigmoid(convolved)
        layers.append(output)

    return layers[-1]






def create_generator(generator_inputs_x, generator_inputs_y, ngf=64, name='generator'):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        outputs_channels = 3

        # ===========================================================
        # encoder for generator_inputs_x
        # ===========================================================
        layers_x = []
        with tf.variable_scope("encoder_x_1", reuse=tf.AUTO_REUSE):
            output = gen_conv(generator_inputs_x, ngf)
            layers_x.append(output)

        layer_specs = [
            ngf * 2,
            ngf * 4,
            ngf * 8,
            ngf * 8,
            ngf * 8,
            # ngf * 8,
            # ngf * 8,
        ]

        for out_channels in layer_specs:
            with tf.variable_scope("encoder_x_%d" % (len(layers_x) + 1), reuse=tf.AUTO_REUSE):
                rectified = lrelu(layers_x[-1], 0.2)
                # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                convolved = gen_conv(rectified, out_channels)
                output = batchnorm(convolved)
                layers_x.append(output)

        # ===========================================================
        # encoder for generator_inputs_y
        # ===========================================================
        layers_y = []
        with tf.variable_scope("encoder_y_1", reuse=tf.AUTO_REUSE):
            output = gen_conv(generator_inputs_y, ngf)
            layers_y.append(output)

        for out_channels in layer_specs:
            with tf.variable_scope("encoder_y_%d" % (len(layers_y) + 1), reuse=tf.AUTO_REUSE):
                rectified = lrelu(layers_y[-1], 0.2)
                # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                convolved = gen_conv(rectified, out_channels)
                output = batchnorm(convolved)
                layers_y.append(output)

        # ===========================================================
        # generate confidence score for generator_inputs_x and generator_inputs_y
        # ===========================================================
        confidences = []
        layers = []

        with tf.variable_scope('confidence_generation', reuse=tf.AUTO_REUSE):

            for i, (layer_x, layer_y) in enumerate(zip(layers_x, layers_y)):
                x = tf.concat([layer_x, layer_y], axis=-1)
                x = tf.layers.conv2d(x, int(x.get_shape().as_list()[-1]/4), 3, padding='same', name='confidence_conv1_'+str(i))
                x = batchnorm(x)
                x = tf.nn.relu(x)
                x = tf.layers.conv2d(x, 1, 3, padding='same', name='confidence_conv2_'+str(i))
                x = batchnorm(x)
                x = tf.nn.sigmoid(x)
                confidences.append(x)
                layer = x*layer_x + (1-x)*layer_y
                layers.append(layer)

        # ===========================================================
        # decoder
        # ===========================================================

        layer_specs = [
            # (ngf * 8, 0.5),   # decoder_8: [batch, 1, 4, ngf * 8] => [batch, 2, 8, ngf * 8 * 2]
            # (ngf * 8, 0.5),   # decoder_7: [batch, 2, 8, ngf * 8 * 2] => [batch, 4, 16, ngf * 8 * 2]
            (ngf * 8, 0.5),   # decoder_6: [batch, 4, 16, ngf * 8 * 2] => [batch, 8, 32, ngf * 8 * 2]
            (ngf * 8, 0.5),   # decoder_5: [batch, 8, 32, ngf * 8 * 2] => [batch, 16, 64, ngf * 8 * 2]
            (ngf * 4, 0.0),   # decoder_4: [batch, 16, 64, ngf * 8 * 2] => [batch, 32, 128, ngf * 4 * 2]
            (ngf * 2, 0.0),   # decoder_3: [batch, 32, 128, ngf * 4 * 2] => [batch, 64, 256, ngf * 2 * 2]
            (ngf, 0.0),       # decoder_2: [batch, 64, 256, ngf * 2 * 2] => [batch, 128, 512, ngf * 2 * 2]
        ]

        num_encoder_layers = len(layers)
        for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
            skip_layer = num_encoder_layers - decoder_layer - 1
            with tf.variable_scope("decoder_%d" % (skip_layer + 1), reuse=tf.AUTO_REUSE):
                if decoder_layer == 0:
                    # first decoder layer doesn't have skip connections
                    # since it is directly connected to the skip_layer
                    input = layers[-1]
                else:
                    input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

                rectified = tf.nn.relu(input)
                # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
                output = gen_deconv(rectified, out_channels)
                output = batchnorm(output)

                if dropout > 0.0:
                    output = tf.nn.dropout(output, keep_prob=1 - dropout)

                layers.append(output)

        # decoder to generate image
        with tf.variable_scope("decoder_1", reuse=tf.AUTO_REUSE):
            # input = tf.concat([layers[-1], layers[0]], axis=3)
            rectified = tf.nn.relu(layers[-1])
            output = gen_deconv(rectified, outputs_channels)
            output = tf.tanh(output)
            layers.append(output)

        # decoder to generate confidence
        with tf.variable_scope("decoder_confidence", reuse=tf.AUTO_REUSE):
            # input = tf.concat([layers[-1], layers[0]], axis=3)
            rectified = tf.nn.relu(layers[-2])
            self_confidence = gen_deconv(rectified, 1)
            self_confidence = tf.nn.sigmoid(self_confidence)

        return layers[-1], self_confidence



