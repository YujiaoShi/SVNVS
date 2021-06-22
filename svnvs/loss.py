#!/usr/bin/env python
"""
Copyright 2019, Yao Yao, HKUST.
Model architectures.
"""

import sys
import math
import tensorflow as tf
import numpy as np
from svnvs.utils import softargmax

EPS = 1e-12

FLAGS = tf.app.flags.FLAGS

from svnvs.utils import *



def compute_depth_map(depth_softmax, depth_start, depth_interval, depth_type):
    '''
    :param depth_logits: [B, H, W, D]
    :param depth_start:
    :param depth_interval:
    :param depth_type:
    :return:
    '''

    if depth_type=='uniform':

        depth_range = depth_start + tf.range(FLAGS.max_d, dtype=tf.float32)*depth_interval
    else:
        # inverse depth map
        depth_end = depth_start + FLAGS.max_d * depth_interval
        inv_depth_start = tf.reshape(tf.div(1.0, depth_start), [])  # [B]
        inv_depth_end = tf.reshape(tf.div(1.0, depth_end), [])  # [B]
        depth_range = tf.div(1.,
                           (inv_depth_end - inv_depth_start) * tf.range(FLAGS.max_d, dtype=tf.float32) / FLAGS.max_d + inv_depth_start)

    depth_map = tf.reduce_sum(depth_softmax * depth_range, axis=-1, keepdims=True)
    return depth_map



def inverse_depth_absolute_diff_loss(depth_softmax, gt_depth_image, depth_start, depth_interval):
    '''
    :param prob_volume: [B, H, W, D]
    :param gt_depth_image:
    :param depth_num:
    :param depth_start:
    :param depth_interval:
    :return:
    '''
    depth_map = compute_depth_map(depth_softmax, depth_start, depth_interval, depth_type='inverse')

    mask_true = tf.cast(tf.not_equal(gt_depth_image, 0.0), dtype='float32')
    denom = tf.reduce_sum(mask_true, axis=[1, 2, 3]) + 1e-7
    masked_abs_error = tf.abs(mask_true * (gt_depth_image - depth_map))
    masked_mae = tf.reduce_sum(masked_abs_error, axis=[1, 2, 3])/denom
    masked_mae = tf.reduce_mean(masked_mae)

    # less than one acc, less than three acc
    # less one accuracy
    less_one_accuracy = less_one_percentage(gt_depth_image, depth_map, tf.abs(depth_interval))
    # less three accuracy
    less_three_accuracy = less_three_percentage(gt_depth_image, depth_map, tf.abs(depth_interval))

    return masked_mae, less_one_accuracy, less_three_accuracy


def inverse_depth_absolute_diff_loss_original(prob_volume, gt_depth_image, depth_num, depth_start, depth_interval):
    '''
    :param prob_volume: [B, H, W, D]
    :param gt_depth_image:
    :param depth_num:
    :param depth_start:
    :param depth_interval:
    :return:
    '''
    depth_end = depth_start + depth_num*depth_interval
    depth_map = softargmax(prob_volume) # [B, H, W, 1]
    inv_depth_start = tf.reshape(tf.div(1.0, depth_start), []) #[B]
    inv_depth_end = tf.reshape(tf.div(1.0, depth_end), [])     #[B]
    depth_map = tf.div(tf.ones_like(depth_map), (inv_depth_end - inv_depth_start)*depth_map/depth_num + inv_depth_start)

    mask_true = tf.cast(tf.not_equal(gt_depth_image, 0.0), dtype='float32')
    denom = tf.reduce_sum(mask_true, axis=[1, 2, 3]) + 1e-7
    masked_abs_error = tf.abs(mask_true * (gt_depth_image - depth_map))
    masked_mae = tf.reduce_sum(masked_abs_error, axis=[1, 2, 3])/denom
    masked_mae = tf.reduce_mean(masked_mae)

    # less than one acc, less than three acc
    # less one accuracy
    less_one_accuracy = less_one_percentage(gt_depth_image, depth_map, tf.abs(depth_interval))
    # less three accuracy
    less_three_accuracy = less_three_percentage(gt_depth_image, depth_map, tf.abs(depth_interval))

    return masked_mae, depth_map, less_one_accuracy, less_three_accuracy





def non_zero_mean_absolute_diff(y_true, y_pred, interval):
    """ non zero mean absolute loss for one batch """
    with tf.name_scope('MAE'):
        shape = tf.shape(y_pred)
        interval = tf.reshape(interval, [shape[0]])
        mask_true = tf.cast(tf.not_equal(y_true, 0.0), dtype='float32')
        denom = tf.reduce_sum(mask_true, axis=[1, 2, 3]) + 1e-7
        masked_abs_error = tf.abs(mask_true * (y_true - y_pred))            # 4D
        masked_mae = tf.reduce_sum(masked_abs_error, axis=[1, 2, 3])        # 1D
        masked_mae = tf.reduce_sum((masked_mae / interval) / denom)         # 1
    return masked_mae

def less_one_percentage(y_true, y_pred, interval):
    """ less one accuracy for one batch """
    with tf.name_scope('less_one_error'):
        shape = tf.shape(y_pred)
        mask_true = tf.cast(tf.not_equal(y_true, 0.0), dtype='float32')
        denom = tf.reduce_sum(mask_true) + 1e-7
        interval_image = tf.tile(tf.reshape(interval, [shape[0], 1, 1, 1]), [1, shape[1], shape[2], 1])
        abs_diff_image = tf.abs(y_true - y_pred) / interval_image
        less_one_image = mask_true * tf.cast(tf.less_equal(abs_diff_image, 1.0), dtype='float32')
    return tf.reduce_sum(less_one_image) / denom

def less_three_percentage(y_true, y_pred, interval):
    """ less three accuracy for one batch """
    with tf.name_scope('less_three_error'):
        shape = tf.shape(y_pred)
        mask_true = tf.cast(tf.not_equal(y_true, 0.0), dtype='float32')
        denom = tf.reduce_sum(mask_true) + 1e-7
        interval_image = tf.tile(tf.reshape(interval, [shape[0], 1, 1, 1]), [1, shape[1], shape[2], 1])
        abs_diff_image = tf.abs(y_true - y_pred) / interval_image
        less_three_image = mask_true * tf.cast(tf.less_equal(abs_diff_image, 3.0), dtype='float32')
    return tf.reduce_sum(less_three_image) / denom

def mvsnet_regression_loss(estimated_depth_image, depth_image, depth_interval):
    """ compute loss and accuracy """
    # non zero mean absulote loss
    masked_mae = non_zero_mean_absolute_diff(depth_image, estimated_depth_image, depth_interval)
    # less one accuracy
    less_one_accuracy = less_one_percentage(depth_image, estimated_depth_image, depth_interval)
    # less three accuracy
    less_three_accuracy = less_three_percentage(depth_image, estimated_depth_image, depth_interval)

    return masked_mae, less_one_accuracy, less_three_accuracy


def mvsnet_classification_loss(prob_volume, gt_depth_image, depth_num, depth_start, depth_interval):
    """ compute loss and accuracy """

    # get depth mask
    mask_true = tf.cast(tf.not_equal(gt_depth_image, 0.0), dtype='float32')
    valid_pixel_num = tf.reduce_sum(mask_true, axis=[1, 2, 3]) + 1e-7
    # gt depth map -> gt index map
    shape = tf.shape(gt_depth_image)
    depth_end = depth_start + (tf.cast(depth_num, tf.float32) - 1) * depth_interval
    start_mat = tf.tile(tf.reshape(depth_start, [shape[0], 1, 1, 1]), [1, shape[1], shape[2], 1])

    interval_mat = tf.tile(tf.reshape(depth_interval, [shape[0], 1, 1, 1]), [1, shape[1], shape[2], 1])
    gt_index_image = tf.div(gt_depth_image - start_mat, interval_mat)
    gt_index_image = tf.multiply(mask_true, gt_index_image)
    gt_index_image = tf.cast(tf.round(gt_index_image), dtype='int32')
    # gt index map -> gt one hot volume (B x H x W x 1)
    gt_index_volume = tf.one_hot(gt_index_image, depth_num, axis=1) # shape=[B, D, H, W, 1]
    # cross entropy image (B x H x W x 1)
    cross_entropy_image = -tf.reduce_sum(gt_index_volume * tf.log(prob_volume), axis=1)
    # masked cross entropy loss
    masked_cross_entropy_image = tf.multiply(mask_true, cross_entropy_image)
    masked_cross_entropy = tf.reduce_sum(masked_cross_entropy_image, axis=[1, 2, 3])
    masked_cross_entropy = tf.reduce_sum(masked_cross_entropy / valid_pixel_num)

    # winner-take-all depth map
    wta_index_map = tf.cast(tf.argmax(prob_volume, axis=1), dtype='float32')
    wta_depth_map = wta_index_map * interval_mat + start_mat    

    # non zero mean absulote loss
    masked_mae = non_zero_mean_absolute_diff(gt_depth_image, wta_depth_map, tf.abs(depth_interval))
    # less one accuracy
    less_one_accuracy = less_one_percentage(gt_depth_image, wta_depth_map, tf.abs(depth_interval))
    # less three accuracy
    less_three_accuracy = less_three_percentage(gt_depth_image, wta_depth_map, tf.abs(depth_interval))

    return masked_cross_entropy, masked_mae, less_one_accuracy, less_three_accuracy, wta_index_map


#def inverse_depth_absolute_diff_loss(prob_volume, gt_depth_image, depth_num, depth_start, depth_interval):
    '''
    :param prob_volume: [B, H, W, D]
    :param gt_depth_image:
    :param depth_num:
    :param depth_start:
    :param depth_interval:
    :return:
    '''
    depth_end = depth_start + depth_num * depth_interval
    depth_map = softargmax(prob_volume)  # [B, H, W, 1]
    inv_depth_start = tf.reshape(tf.div(1.0, depth_start), [])  # [B]
    inv_depth_end = tf.reshape(tf.div(1.0, depth_end), [])  # [B]
    depth_map = tf.div(tf.ones_like(depth_map),
                       (inv_depth_end - inv_depth_start) * depth_map / depth_num + inv_depth_start)

    mask_true = tf.cast(tf.not_equal(gt_depth_image, 0.0), dtype='float32')
    denom = tf.reduce_sum(mask_true, axis=[1, 2, 3]) + 1e-7
    masked_abs_error = tf.abs(mask_true * (gt_depth_image - depth_map))
    masked_mae = tf.reduce_sum(masked_abs_error, axis=[1, 2, 3]) / denom
    masked_mae = tf.reduce_mean(masked_mae)

    # less than one acc, less than three acc
    # less one accuracy
    less_one_accuracy = less_one_percentage(gt_depth_image, depth_map, tf.abs(depth_interval/3))
    # less three accuracy
    less_three_accuracy = less_three_percentage(gt_depth_image, depth_map, tf.abs(depth_interval/3))

    return masked_mae, depth_map, less_one_accuracy, less_three_accuracy


def cls_loss(prob_volume, gt_depth_image, depth_num, depth_start, depth_interval):
    """ compute loss and accuracy """

    # get depth mask
    mask_true = tf.cast(tf.not_equal(gt_depth_image, 0.0), dtype='float32')
    valid_pixel_num = tf.reduce_sum(mask_true, axis=[1, 2, 3]) + 1e-7
    # gt depth map -> gt index map
    shape = tf.shape(gt_depth_image)
    depth_end = depth_start + (tf.cast(depth_num, tf.float32) - 1) * depth_interval
    start_mat = tf.tile(tf.reshape(depth_start, [shape[0], 1, 1, 1]), [1, shape[1], shape[2], 1])

    interval_mat = tf.tile(tf.reshape(depth_interval, [shape[0], 1, 1, 1]), [1, shape[1], shape[2], 1])
    gt_index_image = tf.div(gt_depth_image - start_mat, interval_mat)
    gt_index_image = tf.multiply(mask_true, gt_index_image)
    gt_index_image = tf.cast(tf.round(gt_index_image), dtype='int32')
    # gt index map -> gt one hot volume (B x H x W x 1)
    gt_index_volume = tf.one_hot(gt_index_image, depth_num, axis=1) # shape=[B, D, H, W, 1]
    # cross entropy image (B x H x W x 1)
    cross_entropy_image = -tf.reduce_sum(gt_index_volume * tf.log(prob_volume), axis=1)
    # masked cross entropy loss
    masked_cross_entropy_image = tf.multiply(mask_true, cross_entropy_image)
    masked_cross_entropy = tf.reduce_sum(masked_cross_entropy_image, axis=[1, 2, 3])
    masked_cross_entropy = tf.reduce_sum(masked_cross_entropy / valid_pixel_num)
    return masked_cross_entropy



def kl_loss(prob_volume, prob_volume_ref):
    '''
    :param prob_volume: shape = [B, D, H/4, W/4, 1]
    :param prob_volume_ref: shape = [B, D, H/4, W/4, 1]
    :param gt_depth_image:
    :param depth_num:
    :param depth_start:
    :param depth_interval:
    :return:
    '''
    # cross_entropy_image = -tf.reduce_sum(prob_volume_ref * tf.log(prob_volume), axis=1)
    kl_left = prob_volume * tf.log(prob_volume/(prob_volume_ref+EPS) + EPS)
    kl_right = prob_volume_ref * tf.log(prob_volume_ref/(prob_volume+EPS) + EPS)
    kl_all = tf.reduce_mean(kl_left + kl_right)

    return kl_all


def smoothness_loss(depth_map, image):
    def gradient_x(img):
        gx = img[:, :, :-1, :] - img[:,:,1:, :]
        return gx

    def gradient_y(img):
        gy = img[:, :-1, :, :] - img[:, 1:, :, :]
        return gy

    depth_gradient_x = gradient_x(depth_map)
    depth_gradient_y = gradient_y(depth_map)

    img_gradient_x = gradient_x(image)
    img_gradient_y = gradient_y(image)

    weight_x = tf.exp(-tf.reduce_mean(tf.abs(img_gradient_x), 3, keepdims=True))
    weight_y = tf.exp(-tf.reduce_mean(tf.abs(img_gradient_y), 3, keepdims=True))

    smoothness_x = depth_gradient_x * weight_x
    smoothness_y = depth_gradient_y * weight_y

    return tf.reduce_mean(tf.abs(smoothness_x)) + tf.reduce_mean(tf.abs(smoothness_y))


