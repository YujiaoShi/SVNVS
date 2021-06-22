import sys

sys.path.append("../")

from svnvs.generators import *
from svnvs.Depthmodel import *

FLAGS = tf.app.flags.FLAGS


def warp_image_inv_depth(src_images, src_cams, tgt_cam, depth_start, depth_interval):
    '''
    :param src_images: [B, N, H, W, 3]
    :param src_cams: [B, N, 2, 4, 4]
    :param tgt_cam: [B, 2, 4, 4]
    :param depth_num:
    :param depth_start:
    :param depth_interval:
    :return:
    '''

    depth_end = depth_start + FLAGS.max_d * depth_interval
    view_homographies = []
    for view in range(0, FLAGS.view_num):
        view_cam = tf.squeeze(tf.slice(src_cams, [0, view, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)
        homographies = get_homographies_inv_depth2(tgt_cam, view_cam, depth_num=FLAGS.max_d,
                                        depth_start=depth_start, depth_end=depth_end)
        view_homographies.append(homographies)

    warped_feature_whole = []
    masks_whole = []
    for d in range(FLAGS.max_d):
        view_list = []
        view_mask = []
        for view in range(FLAGS.view_num):
            homography = tf.slice(
                view_homographies[view], begin=[0, d, 0, 0], size=[-1, 1, 3, 3])
            homography = tf.squeeze(homography, axis=1)
            warped_view_img, mask = homography_warping_tfcontrib(src_images[:, view, ...], homography)  # [B, H/2, W/2, 3]
            warped_view_img = warped_view_img * mask
            view_list.append(warped_view_img)
            view_mask.append(mask)

        warped_feature_whole.append(tf.stack(view_list, axis=0))  # [N, B, H, W, 3]
        masks_whole.append(tf.stack(view_mask, axis=0))  # [N, B, H, W, 1]

    warped_feature_whole = tf.stack(warped_feature_whole, axis=0)  # [D, N, B, H, W, 3]
    masks_whole = tf.stack(masks_whole, axis=0) # [D, N, B, H, W, 1]
    return warped_feature_whole, masks_whole

    # warped_features = tf.stack(warped_feature_whole, axis=0)  # [D, N, B, H, W, 3]
    # masks = tf.stack(masks_whole, axis=0) # [D, N, B, H, W, 1]
    # return warped_features, masks


def NVSmodel(src_images, src_cams, tgt_cam, depth_prob_volume, src_weights, depth_num, depth_start, depth_interval):
    '''
    :param images:
    :param view_homographies:
    :param src_weights: # [D, N, B, H/4, W/4]
    :param depth_num:
    :param warped_feature_for_cost:
    :param warp:
    :param pe:
    :return:
    '''

    half_h = int(FLAGS.max_h // 2)
    half_w = int(FLAGS.max_w // 2)
    height = FLAGS.max_h
    width = FLAGS.max_w

    # ======== src weights ============
    src_weights_full_size = tf.reshape(tf.image.resize(tf.reshape(src_weights, [-1, half_h, half_w, 1]),
                                                       (height, width)),
                                       [depth_num, FLAGS.view_num, FLAGS.batch_size, height, width, 1])
    src_weights_softmax = stable_softmax(src_weights_full_size, axis=1)
    # [D, N, B, H, W, 1]


    # ======== depth prob ============
    depth_prob_volume_full_size = tf.reshape(tf.image.resize(tf.reshape(depth_prob_volume, [-1, half_h, half_w, 1]),
                                                        (height, width)),
                                        [depth_num, FLAGS.batch_size, height, width, 1])


    depth_prob_volume_softmax = stable_softmax(depth_prob_volume_full_size, axis=0)

    # =============================== warp images =========================================
    warped_imgs_srcs, src_masks = warp_image_inv_depth(src_images, src_cams, tgt_cam, depth_start, depth_interval)
    # [D, N, B, H, W, 3], # [D, N, B, H, W, 1]

    # =============== handle source weights with masks (valid warp pixels) ===========
    src_weights_softmax = src_weights_softmax * src_masks  # [D, N, B, H, W, 1]
    src_weights_softmax_sum = tf.reduce_sum(src_weights_softmax, axis=1, keepdims=True)
    src_weights_softmax_sum_zero_add = tf.cast(tf.equal(src_weights_softmax_sum, 0.0), tf.float32) * 1e-7
    src_weights_softmax_sum += src_weights_softmax_sum_zero_add
    src_weights_softmax = src_weights_softmax/(src_weights_softmax_sum)

    # =============== Compute aggregated images =====================================
    weighted_src_img = tf.reduce_sum(src_weights_softmax * warped_imgs_srcs, axis=1) # [D, B, H, W, 3]
    aggregated_img = tf.reduce_sum(weighted_src_img * depth_prob_volume_softmax, axis=0)
    # [B, H, W, 3]

    warped_imgs_srcs = tf.reduce_sum(warped_imgs_srcs * tf.expand_dims(depth_prob_volume_softmax, axis=1), axis=0)
    # [N, B, H, W, 3]

    # ======== generator =================================

    output_imgs = []
    self_confidences = []
    for src_index in range(FLAGS.view_num - 1):
        out_img, confidence = create_generator(aggregated_img, warped_imgs_srcs[src_index], name='generator')
        # [B, H/4, W/4, 3], [B, H/4, W/4, 1]
        output_imgs.append(out_img)
        self_confidences.append(confidence)

    outputs_imgs = tf.stack(output_imgs, axis=-1)  # [B, H/4, W/4, 3, N]
    img_confidences = tf.stack(self_confidences, axis=-1)  # [B, H/4, W/4, 1, N]
    img_confidences_norm = img_confidences / (tf.reduce_sum(img_confidences, axis=-1, keepdims=True) + 1e-7)
    final_output_img = tf.reduce_sum(outputs_imgs * img_confidences_norm, axis=-1)

    return aggregated_img, final_output_img, warped_imgs_srcs
           # [B, H, W, 3], [B, H, W, 3], [N, B, H, W, 3]


