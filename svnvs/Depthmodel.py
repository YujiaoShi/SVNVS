import sys

sys.path.append("../")
from cnn_wrapper.mvsnet import *
from svnvs.homography_warping import *

from svnvs.utils import *
from svnvs.lstm import ConvLSTMCell

FLAGS = tf.app.flags.FLAGS


def deconv_gn(input_tensor,
              kernel_size,
              filters,
              strides,
              name,
              relu=False,
              center=False,
              scale=False,
              channel_wise=True,
              group=32,
              group_channel=8,
              padding='same',
              biased=False,
              reuse=tf.AUTO_REUSE):
    assert len(input_tensor.get_shape()) == 4

    # deconvolution
    res = tf.layers.conv2d_transpose(input_tensor, kernel_size=kernel_size, filters=filters, padding=padding,
                                     strides=strides,
                                     reuse=reuse, name=name)
    # group normalization
    x = tf.transpose(res, [0, 3, 1, 2])
    shape = tf.shape(x)
    N = shape[0]
    C = x.get_shape().as_list()[1]
    H = shape[2]
    W = shape[3]
    if channel_wise:
        G = max(1, int(C / group_channel))
    else:
        G = min(group, C)

    # normalization
    x = tf.reshape(x, [N, G, int(C // G), H, W])
    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
    x = (x - mean) / tf.sqrt(var + 1e-5)

    # per channel scale and bias (gamma and beta)
    with tf.variable_scope(name + '/gn', reuse=reuse):
        if scale:
            gamma = tf.get_variable('gamma', [C], dtype=tf.float32, initializer=tf.ones_initializer())
        else:
            gamma = tf.constant(1.0, shape=[C])
        if center:
            beta = tf.get_variable('beta', [C], dtype=tf.float32, initializer=tf.zeros_initializer())
        else:
            beta = tf.constant(0.0, shape=[C])
    gamma = tf.reshape(gamma, [1, C, 1, 1])
    beta = tf.reshape(beta, [1, C, 1, 1])
    output = tf.reshape(x, [-1, C, H, W]) * gamma + beta

    # tranpose: [bs, c, h, w, c] to [bs, h, w, c] following the paper
    output = tf.transpose(output, [0, 2, 3, 1])

    if relu:
        output = tf.nn.relu(output, name + '/relu')
    return output


def extract_feature(x, trainable=True, name='extract_feature'):

    with tf.variable_scope(name, tf.AUTO_REUSE):

        net = SNetDS2BN_base_8({'data': x}, is_training=trainable, reuse=tf.AUTO_REUSE)
        feature = net.get_output()

    return feature




def DepthModelinv(src_images, src_cams, tgt_cam, depth_num, depth_start, depth_interval, trainable=True):

    # This function operates on half image size for memory efficiency

    depth_end = depth_start + depth_num * depth_interval

    half_h = int(FLAGS.max_h / 2)
    half_w = int(FLAGS.max_w / 2)
    src_images = tf.image.resize(tf.reshape(src_images, [-1, FLAGS.max_h, FLAGS.max_w, 3]),
                                 (half_h, half_w))
    src_images = tf.reshape(src_images, [FLAGS.batch_size, FLAGS.view_num, half_h, half_w, 3])


    # extract source view features for cost aggregation, and source view weights calculation
    view_towers = []
    for view in range(0, FLAGS.view_num):
        view_image = tf.squeeze(tf.slice(src_images, [0, view, 0, 0, 0], [-1, 1, -1, -1, -1]), axis=1)
        view_tower = extract_feature(view_image, trainable=trainable, name='cost_feature')
        view_towers.append(view_tower)

    C = view_towers[-1].get_shape().as_list()[-1]

    # get all homographies
    view_homographies = []
    for view in range(0, FLAGS.view_num):
        view_cam = tf.squeeze(tf.slice(src_cams, [0, view, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)
        homographies = get_homographies_inv_depth(tgt_cam, view_cam, depth_num=depth_num,
                                        depth_start=depth_start, depth_end=depth_end)

        view_homographies.append(homographies)


    # Construct four LSTM sells, the first four is for Source-view Visibility Estimation (SVE)
    # The last one is the Soft Ray-Casting (SRC)
    feature_shape = [FLAGS.batch_size, half_h, half_w, C]
    batch_size, height, width, channel = feature_shape
    cell0 = ConvLSTMCell(
        conv_ndims=2,
        input_shape=[height, width, 16],
        output_channels=8,
        kernel_shape=[3, 3],

        name="conv_lstm_cell0"
    )
    cell1 = ConvLSTMCell(
        conv_ndims=2,
        input_shape=[height / 2, width / 2, 8],
        output_channels=4,
        kernel_shape=[3, 3],
        name="conv_lstm_cell1"
    )

    cell2 = ConvLSTMCell(
        conv_ndims=2,
        input_shape=[height / 4, width / 4, 4],
        output_channels=4,
        kernel_shape=[3, 3],
        name="conv_lstm_cell2"
    )

    cell3 = ConvLSTMCell(
        conv_ndims=2,
        input_shape=[height / 2, width / 2, 8],
        output_channels=4,
        kernel_shape=[3, 3],
        name="conv_lstm_cell3"
    )

    cell4 = ConvLSTMCell(
        conv_ndims=2,
        input_shape=[height, width, 12],
        output_channels=4,
        kernel_shape=[3, 3],
        name="conv_lstm_cell4"
    )

    initial_state0 = cell0.zero_state(batch_size*(FLAGS.view_num), dtype=tf.float32)
    initial_state1 = cell1.zero_state(batch_size*(FLAGS.view_num), dtype=tf.float32)
    initial_state2 = cell2.zero_state(batch_size*(FLAGS.view_num), dtype=tf.float32)
    initial_state3 = cell3.zero_state(batch_size*(FLAGS.view_num), dtype=tf.float32)
    initial_state4 = cell4.zero_state(batch_size, dtype=tf.float32)

    with tf.name_scope('cost_volume_homography'):

        # forward cost volume
        src_weights = []
        depth_probs = []
        warped_feature_whole = []

        for d in range(depth_num):

            feature_list = []

            masks = []
            for view in range(0, FLAGS.view_num):
                homography = tf.slice(
                    view_homographies[view], begin=[0, d, 0, 0], size=[-1, 1, 3, 3])
                homography = tf.squeeze(homography, axis=1)
                warped_view_feature, view_mask = homography_warping_tfcontrib(view_towers[view], homography)

                feature_list.append(warped_view_feature)
                masks.append(view_mask)

            src_features = tf.stack(feature_list, axis=0)  # [N, B, H/2, W/2, C]
            warped_feature_whole.append(src_features)


            cost = tf.einsum('ncbhw, cmbhw->nmbhw', tf.transpose(src_features, [0,4,1,2,3]),
                             tf.transpose(src_features, [4,0,1,2,3])) # [N, N, B, H/2, W/2]
            view_cost = tf.expand_dims(tf.reduce_mean(cost, axis=0), axis=-1)  # [N, B, H/2, W/2, 1]
            # compute similarity, corresponds to Eq.(5) in the paper


            view_cost_mean = tf.tile(tf.reduce_mean(view_cost, axis=0, keepdims=True), [FLAGS.view_num, 1, 1, 1, 1])
            view_cost = tf.concat([src_features, view_cost, view_cost_mean], axis=-1)
            view_cost = tf.reshape(view_cost, [(FLAGS.view_num)*FLAGS.batch_size, feature_shape[1], feature_shape[2], C+2])
            # Construct input to our Souce-view Visibility Estimation (SVE) module
            # Corresponds to Eq.(6) in the paper


            with tf.variable_scope("rnn/", reuse=tf.AUTO_REUSE):

                # ================ starts Source-view Visibility Estimation (SVE) ===================================
                feature_out0, initial_state0 = cell0(view_cost, state=initial_state0)
                feature_out1 = tf.nn.max_pool(feature_out0, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

                feature_out1, initial_state1 = cell1(feature_out1, state=initial_state1)
                feature_out2 = tf.nn.max_pool(feature_out1, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

                feature_out2, initial_state2 = cell2(feature_out2, state=initial_state2)


                feature_out2 = deconv_gn(feature_out2, 16, 3, padding='same', strides=2, reuse=tf.AUTO_REUSE,
                                         name='cost_upconv0')
                feature_out2 = tf.concat([feature_out2, feature_out1], -1)
                feature_out3, initial_state3 = cell3(feature_out2, state=initial_state3)


                feature_out3 = deconv_gn(feature_out3, 16, 3, padding='same', strides=2, reuse=tf.AUTO_REUSE,
                                         name='cost_upconv1')
                feature_out3 = tf.concat([feature_out3, feature_out0], -1)

                feature_out3 = tf.layers.conv2d(
                                    feature_out3, 9, 3, padding='same', reuse=tf.AUTO_REUSE, name='refine_conv1',
                                            trainable=trainable)
                # ================ ends Source-view Visibility Estimation (SVE) ===================================
                # process output:
                feature_out3 = tf.reshape(feature_out3, [FLAGS.view_num, FLAGS.batch_size, feature_shape[1], feature_shape[2], 9])
                src_weight = feature_out3[..., 0]  # [N, B, H, W]
                # The first output channel is to compute the source view visibility (ie, weight)
                feature_out3 = tf.reduce_mean(feature_out3[..., 1:], axis=0) # shape= [B, H, W, 8]
                # The last eight channels are used to compute the consensus volume
                # Correspoding to Eq.(7) in the paper

                # ================ starts Soft Ray-Casting (SRC) ========================
                feature_out4, initial_state4 = cell4(feature_out3, state=initial_state4)
                features = tf.layers.conv2d(feature_out4, 1, 3, padding='same', reuse=tf.AUTO_REUSE, name='weights_conv',
                                            trainable=trainable)
                # ================ ends Soft Ray-Casting (SRC) ==========================

            src_weights.append(src_weight)
            depth_probs.append(features)

        src_weights = tf.stack(src_weights, axis=0)  # [D, N, B, H/4, W/4]
        depth_probs = tf.stack(depth_probs, axis=1)  # [B, D, H/4, W/4, 1]

        return depth_probs, src_weights


