#!/usr/bin/env python
"""
Copyright 2019, Yao Yao, HKUST.
Differentiable homography related.
"""

import tensorflow as tf
import numpy as np

def get_homographies(left_cam, right_cam, depth_num, depth_start, depth_interval):
    with tf.name_scope('get_homographies'):
        # cameras (K, R, t)
        R_left = tf.slice(left_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
        R_right = tf.slice(right_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
        t_left = tf.slice(left_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
        t_right = tf.slice(right_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
        K_left = tf.slice(left_cam, [0, 1, 0, 0], [-1, 1, 3, 3])
        K_right = tf.slice(right_cam, [0, 1, 0, 0], [-1, 1, 3, 3])

        # depth 
        depth_num = tf.reshape(tf.cast(depth_num, 'int32'), [])
        depth = depth_start + tf.cast(tf.range(depth_num), tf.float32) * depth_interval
        # preparation
        num_depth = tf.shape(depth)[0]
        K_left_inv = tf.matrix_inverse(tf.squeeze(K_left, axis=1))
        R_left_trans = tf.transpose(tf.squeeze(R_left, axis=1), perm=[0, 2, 1])
        R_right_trans = tf.transpose(tf.squeeze(R_right, axis=1), perm=[0, 2, 1])

        fronto_direction = tf.slice(tf.squeeze(R_left, axis=1), [0, 2, 0], [-1, 1, 3])          # (B, D, 1, 3)

        c_left = -tf.matmul(R_left_trans, tf.squeeze(t_left, axis=1))
        c_right = -tf.matmul(R_right_trans, tf.squeeze(t_right, axis=1))                        # (B, D, 3, 1)
        c_relative = tf.subtract(c_right, c_left)        

        # compute
        batch_size = tf.shape(R_left)[0]
        temp_vec = tf.matmul(c_relative, fronto_direction)
        depth_mat = tf.tile(tf.reshape(depth, [batch_size, num_depth, 1, 1]), [1, 1, 3, 3])

        temp_vec = tf.tile(tf.expand_dims(temp_vec, axis=1), [1, num_depth, 1, 1])

        middle_mat0 = tf.eye(3, batch_shape=[batch_size, num_depth]) - temp_vec / depth_mat
        middle_mat1 = tf.tile(tf.expand_dims(tf.matmul(R_left_trans, K_left_inv), axis=1), [1, num_depth, 1, 1])
        middle_mat2 = tf.matmul(middle_mat0, middle_mat1)

        homographies = tf.matmul(tf.tile(K_right, [1, num_depth, 1, 1])
                     , tf.matmul(tf.tile(R_right, [1, num_depth, 1, 1])
                     , middle_mat2))

    return homographies



def get_homographies2(left_cam, right_cam, depth_num, depth_start, depth_interval):
    with tf.name_scope('get_homographies'):
        # cameras (K, R, t)
        R_left = tf.slice(left_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
        R_right = tf.slice(right_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
        t_left = tf.slice(left_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
        t_right = tf.slice(right_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
        K_left_2 = tf.slice(left_cam, [0, 1, 0, 0], [-1, 1, 3, 3])
        K_right_2 = tf.slice(right_cam, [0, 1, 0, 0], [-1, 1, 3, 3])
        K_left = tf.concat([K_left_2[:,:, :2, :]*2, K_left_2[:, :, 2:3, :]], axis=2)
        K_right = tf.concat([K_right_2[:,:, :2, :]*2, K_right_2[:, :, 2:3, :]], axis=2)

        # depth
        depth_num = tf.reshape(tf.cast(depth_num, 'int32'), [])
        depth = depth_start + tf.cast(tf.range(depth_num), tf.float32) * depth_interval
        # preparation
        num_depth = tf.shape(depth)[0]
        K_left_inv = tf.matrix_inverse(tf.squeeze(K_left, axis=1))
        R_left_trans = tf.transpose(tf.squeeze(R_left, axis=1), perm=[0, 2, 1])
        R_right_trans = tf.transpose(tf.squeeze(R_right, axis=1), perm=[0, 2, 1])

        fronto_direction = tf.slice(tf.squeeze(R_left, axis=1), [0, 2, 0], [-1, 1, 3])          # (B, D, 1, 3)

        c_left = -tf.matmul(R_left_trans, tf.squeeze(t_left, axis=1))
        c_right = -tf.matmul(R_right_trans, tf.squeeze(t_right, axis=1))                        # (B, D, 3, 1)
        c_relative = tf.subtract(c_right, c_left)

        # compute
        batch_size = tf.shape(R_left)[0]
        temp_vec = tf.matmul(c_relative, fronto_direction)
        depth_mat = tf.tile(tf.reshape(depth, [batch_size, num_depth, 1, 1]), [1, 1, 3, 3])

        temp_vec = tf.tile(tf.expand_dims(temp_vec, axis=1), [1, num_depth, 1, 1])

        middle_mat0 = tf.eye(3, batch_shape=[batch_size, num_depth]) - temp_vec / depth_mat
        middle_mat1 = tf.tile(tf.expand_dims(tf.matmul(R_left_trans, K_left_inv), axis=1), [1, num_depth, 1, 1])
        middle_mat2 = tf.matmul(middle_mat0, middle_mat1)

        homographies = tf.matmul(tf.tile(K_right, [1, num_depth, 1, 1])
                     , tf.matmul(tf.tile(R_right, [1, num_depth, 1, 1])
                     , middle_mat2))

    return homographies


def get_homographies_inv_depth2(left_cam, right_cam, depth_num, depth_start, depth_end):

    with tf.name_scope('get_homographies'):
        # cameras (K, R, t)
        R_left = tf.slice(left_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
        R_right = tf.slice(right_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
        t_left = tf.slice(left_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
        t_right = tf.slice(right_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
        # K_left = tf.slice(left_cam, [0, 1, 0, 0], [-1, 1, 3, 3])
        # K_right = tf.slice(right_cam, [0, 1, 0, 0], [-1, 1, 3, 3])
        K_left_2 = tf.slice(left_cam, [0, 1, 0, 0], [-1, 1, 3, 3])
        K_right_2 = tf.slice(right_cam, [0, 1, 0, 0], [-1, 1, 3, 3])
        K_left = tf.concat([K_left_2[:, :, :2, :] * 2, K_left_2[:, :, 2:3, :]], axis=2)
        K_right = tf.concat([K_right_2[:, :, :2, :] * 2, K_right_2[:, :, 2:3, :]], axis=2)

        # depth
        depth_num = tf.reshape(tf.cast(depth_num, 'int32'), [])

        inv_depth_start = tf.reshape(tf.div(1.0, depth_start), [])
        inv_depth_end = tf.reshape(tf.div(1.0, depth_end), [])
        inv_depth = tf.lin_space(inv_depth_start, inv_depth_end, depth_num)
        depth = tf.div(1.0, inv_depth)

        # preparation
        num_depth = tf.shape(depth)[0]
        K_left_inv = tf.matrix_inverse(tf.squeeze(K_left, axis=1))
        R_left_trans = tf.transpose(tf.squeeze(R_left, axis=1), perm=[0, 2, 1])
        R_right_trans = tf.transpose(tf.squeeze(R_right, axis=1), perm=[0, 2, 1])

        fronto_direction = tf.slice(tf.squeeze(R_left, axis=1), [0, 2, 0], [-1, 1, 3])          # (B, D, 1, 3)

        c_left = -tf.matmul(R_left_trans, tf.squeeze(t_left, axis=1))
        c_right = -tf.matmul(R_right_trans, tf.squeeze(t_right, axis=1))                        # (B, D, 3, 1)
        c_relative = tf.subtract(c_right, c_left)

        # compute
        batch_size = tf.shape(R_left)[0]
        temp_vec = tf.matmul(c_relative, fronto_direction)
        depth_mat = tf.tile(tf.reshape(depth, [batch_size, num_depth, 1, 1]), [1, 1, 3, 3])

        temp_vec = tf.tile(tf.expand_dims(temp_vec, axis=1), [1, num_depth, 1, 1])

        middle_mat0 = tf.eye(3, batch_shape=[batch_size, num_depth]) - temp_vec / depth_mat
        middle_mat1 = tf.tile(tf.expand_dims(tf.matmul(R_left_trans, K_left_inv), axis=1), [1, num_depth, 1, 1])
        middle_mat2 = tf.matmul(middle_mat0, middle_mat1)

        homographies = tf.matmul(tf.tile(K_right, [1, num_depth, 1, 1])
                     , tf.matmul(tf.tile(R_right, [1, num_depth, 1, 1])
                     , middle_mat2))

    return homographies


def get_homographies_inv_depth(left_cam, right_cam, depth_num, depth_start, depth_end):

    with tf.name_scope('get_homographies'):
        # cameras (K, R, t)
        R_left = tf.slice(left_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
        R_right = tf.slice(right_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
        t_left = tf.slice(left_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
        t_right = tf.slice(right_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
        K_left = tf.slice(left_cam, [0, 1, 0, 0], [-1, 1, 3, 3])
        K_right = tf.slice(right_cam, [0, 1, 0, 0], [-1, 1, 3, 3])

        # depth 
        depth_num = tf.reshape(tf.cast(depth_num, 'int32'), [])

        inv_depth_start = tf.reshape(tf.div(1.0, depth_start), [])
        inv_depth_end = tf.reshape(tf.div(1.0, depth_end), [])
        inv_depth = tf.lin_space(inv_depth_start, inv_depth_end, depth_num)
        depth = tf.div(1.0, inv_depth)

        # preparation
        num_depth = tf.shape(depth)[0]
        K_left_inv = tf.matrix_inverse(tf.squeeze(K_left, axis=1))
        R_left_trans = tf.transpose(tf.squeeze(R_left, axis=1), perm=[0, 2, 1])
        R_right_trans = tf.transpose(tf.squeeze(R_right, axis=1), perm=[0, 2, 1])

        fronto_direction = tf.slice(tf.squeeze(R_left, axis=1), [0, 2, 0], [-1, 1, 3])          # (B, D, 1, 3)

        c_left = -tf.matmul(R_left_trans, tf.squeeze(t_left, axis=1))
        c_right = -tf.matmul(R_right_trans, tf.squeeze(t_right, axis=1))                        # (B, D, 3, 1)
        c_relative = tf.subtract(c_right, c_left)        

        # compute
        batch_size = tf.shape(R_left)[0]
        temp_vec = tf.matmul(c_relative, fronto_direction)
        depth_mat = tf.tile(tf.reshape(depth, [batch_size, num_depth, 1, 1]), [1, 1, 3, 3])

        temp_vec = tf.tile(tf.expand_dims(temp_vec, axis=1), [1, num_depth, 1, 1])

        middle_mat0 = tf.eye(3, batch_shape=[batch_size, num_depth]) - temp_vec / depth_mat
        middle_mat1 = tf.tile(tf.expand_dims(tf.matmul(R_left_trans, K_left_inv), axis=1), [1, num_depth, 1, 1])
        middle_mat2 = tf.matmul(middle_mat0, middle_mat1)

        homographies = tf.matmul(tf.tile(K_right, [1, num_depth, 1, 1])
                     , tf.matmul(tf.tile(R_right, [1, num_depth, 1, 1])
                     , middle_mat2))

    return homographies

def get_pixel_grids(height, width):
    # texture coordinate
    x_linspace = tf.linspace(0.5, tf.cast(width, 'float32') - 0.5, width)
    y_linspace = tf.linspace(0.5, tf.cast(height, 'float32') - 0.5, height)
    x_coordinates, y_coordinates = tf.meshgrid(x_linspace, y_linspace)
    x_coordinates = tf.reshape(x_coordinates, [-1])
    y_coordinates = tf.reshape(y_coordinates, [-1])
    ones = tf.ones_like(x_coordinates)
    indices_grid = tf.concat([x_coordinates, y_coordinates, ones], 0) # shape = [3*height*width]
    return indices_grid

def repeat_int(x, num_repeats):
    ones = tf.ones((1, num_repeats), dtype='int32')
    x = tf.reshape(x, shape=(-1, 1))
    x = tf.matmul(x, ones)
    return tf.reshape(x, [-1])

def repeat_float(x, num_repeats):
    ones = tf.ones((1, num_repeats), dtype='float')
    x = tf.reshape(x, shape=(-1, 1))
    x = tf.matmul(x, ones)
    return tf.reshape(x, [-1])

def interpolate(image, x, y):
    image_shape = tf.shape(image)
    batch_size = image_shape[0]
    height =image_shape[1]
    width = image_shape[2]

    # image coordinate to pixel coordinate
    x = x - 0.5
    y = y - 0.5
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1
    max_y = tf.cast(height - 1, dtype='int32')
    max_x = tf.cast(width - 1,  dtype='int32')
    x0 = tf.clip_by_value(x0, 0, max_x)
    x1 = tf.clip_by_value(x1, 0, max_x)
    y0 = tf.clip_by_value(y0, 0, max_y)
    y1 = tf.clip_by_value(y1, 0, max_y)
    b = repeat_int(tf.range(batch_size), height * width)

    indices_a = tf.stack([b, y0, x0], axis=1)
    indices_b = tf.stack([b, y0, x1], axis=1)
    indices_c = tf.stack([b, y1, x0], axis=1)
    indices_d = tf.stack([b, y1, x1], axis=1)

    pixel_values_a = tf.gather_nd(image, indices_a)
    pixel_values_b = tf.gather_nd(image, indices_b)
    pixel_values_c = tf.gather_nd(image, indices_c)
    pixel_values_d = tf.gather_nd(image, indices_d)

    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')
    area_a = tf.expand_dims(((y1 - y) * (x1 - x)), 1)
    area_b = tf.expand_dims(((y1 - y) * (x - x0)), 1)
    area_c = tf.expand_dims(((y - y0) * (x1 - x)), 1)
    area_d = tf.expand_dims(((y - y0) * (x - x0)), 1)
    output = tf.add_n([area_a * pixel_values_a,
                        area_b * pixel_values_b,
                        area_c * pixel_values_c,
                        area_d * pixel_values_d])
    return output

def homography_warping(input_image, homography):
    with tf.name_scope('warping_by_homography'):
        image_shape = tf.shape(input_image)
        batch_size = image_shape[0]
        height = image_shape[1]
        width = image_shape[2]

        # turn homography to affine_mat of size (B, 2, 3) and div_mat of size (B, 1, 3)
        affine_mat = tf.slice(homography, [0, 0, 0], [-1, 2, 3])
        div_mat = tf.slice(homography, [0, 2, 0], [-1, 1, 3])

        # generate pixel grids of size (B, 3, H x W)
        pixel_grids = get_pixel_grids(height, width)  # [3 x H x W]
        pixel_grids = tf.expand_dims(pixel_grids, 0)  # [1, 3 x H x W]
        pixel_grids = tf.tile(pixel_grids, [batch_size, 1])  # [B, 3 x H x W]
        pixel_grids = tf.reshape(pixel_grids, (batch_size, 3, -1))  # [B, 3, H x W]
        # return pixel_grids

        # affine + divide tranform, output (B, 2, H x W)
        grids_affine = tf.matmul(affine_mat, pixel_grids) # (B, 2, 3) * (B, 3, H x W) --> (B, 2, H x W)
        grids_div = tf.matmul(div_mat, pixel_grids)       # (B, 1, 3) * (B, 3, H x W) --> (B, 1, H x W)
        grids_zero_add = tf.cast(tf.equal(grids_div, 0.0), dtype='float32') * 1e-7 # handle div 0
        grids_div = grids_div + grids_zero_add
        grids_div = tf.tile(grids_div, [1, 2, 1])
        grids_inv_warped = tf.div(grids_affine, grids_div)
        x_warped, y_warped = tf.unstack(grids_inv_warped, axis=1)
        x_warped_flatten = tf.reshape(x_warped, [-1])
        y_warped_flatten = tf.reshape(y_warped, [-1])

        # interpolation
        warped_image = interpolate(input_image, x_warped_flatten, y_warped_flatten)
        warped_image = tf.reshape(warped_image, shape=image_shape, name='warped_feature')

    # return input_image
    return warped_image


def homography_warping_tfcontrib(input_image, homography):
    with tf.name_scope('warping_by_homography'):
        image_shape = tf.shape(input_image)
        batch_size = image_shape[0]
        height = image_shape[1]
        width = image_shape[2]

        # turn homography to affine_mat of size (B, 2, 3) and div_mat of size (B, 1, 3)
        affine_mat = tf.slice(homography, [0, 0, 0], [-1, 2, 3])
        div_mat = tf.slice(homography, [0, 2, 0], [-1, 1, 3])

        # generate pixel grids of size (B, 3, H x W)
        pixel_grids = get_pixel_grids(height, width)  # [3 x H x W]
        pixel_grids = tf.expand_dims(pixel_grids, 0)  # [1, 3 x H x W]
        pixel_grids = tf.tile(pixel_grids, [batch_size, 1])  # [B, 3 x H x W]
        pixel_grids = tf.reshape(pixel_grids, (batch_size, 3, -1))  # [B, 3, H x W]
        # return pixel_grids

        # affine + divide tranform, output (B, 2, H x W)
        grids_affine = tf.matmul(affine_mat, pixel_grids) # (B, 2, 3) * (B, 3, H x W) --> (B, 2, H x W)
        grids_div = tf.matmul(div_mat, pixel_grids)       # (B, 1, 3) * (B, 3, H x W) --> (B, 1, H x W)
        grids_zero_add = tf.cast(tf.equal(grids_div, 0.0), dtype='float32') * 1e-7 # handle div 0
        grids_div = grids_div + grids_zero_add
        grids_div = tf.tile(grids_div, [1, 2, 1])
        grids_inv_warped = tf.div(grids_affine, grids_div)  # [B, 2, H x W]
        # x_warped, y_warped = tf.unstack(grids_inv_warped, axis=1)
        # x_warped_flatten = tf.reshape(x_warped, [-1])
        # y_warped_flatten = tf.reshape(y_warped, [-1])
        grids_inv_warped = tf.reshape(tf.transpose(grids_inv_warped, [0, 2, 1]), [batch_size, height, width, 2])

        mask0 = tf.cast(tf.greater_equal(grids_inv_warped[..., 0:1], 0), tf.float32)
        mask1 = tf.cast(tf.less_equal(grids_inv_warped[..., 0:1], tf.cast(width, tf.float32)), tf.float32)
        mask2 = tf.cast(tf.greater_equal(grids_inv_warped[..., 1:], 0), tf.float32)
        mask3 = tf.cast(tf.less_equal(grids_inv_warped[..., 1:], tf.cast(height, tf.float32)), tf.float32)
        mask = mask0 * mask1 * mask2 * mask3  # indicator of valid value (1: valid, 0: invalid)

        warped_image = tf.contrib.resampler.resampler(input_image, grids_inv_warped)
        # warped_image = warped_image * mask

        # # interpolation
        # warped_image = interpolate(input_image, x_warped_flatten, y_warped_flatten)
        # warped_image = tf.reshape(warped_image, shape=image_shape, name='warped_feature')

    # return input_image
    return warped_image, mask


def tf_transform_homography(input_image, homography):

	# tf.contrib.image.transform is for pixel coordinate but our
	# homograph parameters are for image coordinate (x_p = x_i + 0.5).
	# So need to change the corresponding homography parameters 
    homography = tf.reshape(homography, [-1, 9])
    a0 = tf.slice(homography, [0, 0], [-1, 1])
    a1 = tf.slice(homography, [0, 1], [-1, 1])
    a2 = tf.slice(homography, [0, 2], [-1, 1])
    b0 = tf.slice(homography, [0, 3], [-1, 1])
    b1 = tf.slice(homography, [0, 4], [-1, 1])
    b2 = tf.slice(homography, [0, 5], [-1, 1])
    c0 = tf.slice(homography, [0, 6], [-1, 1])
    c1 = tf.slice(homography, [0, 7], [-1, 1])
    c2 = tf.slice(homography, [0, 8], [-1, 1])
    a_0 = a0 - c0 / 2
    a_1 = a1 - c1 / 2
    a_2 = (a0 + a1) / 2 + a2 - (c0 + c1) / 4 - c2 / 2
    b_0 = b0 - c0 / 2
    b_1 = b1 - c1 / 2
    b_2 = (b0 + b1) / 2 + b2 - (c0 + c1) / 4 - c2 / 2
    c_0 = c0
    c_1 = c1
    c_2 = c2 + (c0 + c1) / 2
    homo = []
    homo.append(a_0)
    homo.append(a_1)
    homo.append(a_2)
    homo.append(b_0)
    homo.append(b_1)
    homo.append(b_2)
    homo.append(c_0)
    homo.append(c_1)
    homo.append(c_2)
    homography = tf.stack(homo, axis=1)
    homography = tf.reshape(homography, [-1, 9])

    homography_linear = tf.slice(homography, begin=[0, 0], size=[-1, 8])
    homography_linear_div = tf.tile(tf.slice(homography, begin=[0, 8], size=[-1, 1]), [1, 8])
    homography_linear = tf.div(homography_linear, homography_linear_div)
    warped_image = tf.contrib.image.transform(
        input_image, homography_linear, interpolation='BILINEAR')

    # return input_image
    return warped_image

