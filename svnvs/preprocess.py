#!/usr/bin/env python
"""
Copyright 2019, Yao Yao, HKUST.
Training preprocesses.
"""

from __future__ import print_function

import os
import time
import glob
import random
import math
import re
import sys

import cv2
import numpy as np
import tensorflow as tf
import scipy.io
import urllib
from tensorflow.python.lib.io import file_io
FLAGS = tf.app.flags.FLAGS

# def center_image(img):
#     """ normalize image input """
#     img = img.astype(np.float32)
#     var = np.var(img, axis=(0,1), keepdims=True)
#     mean = np.mean(img, axis=(0,1), keepdims=True)
#     return (img - mean) / (np.sqrt(var) + 0.00000001)
#
# def scale_camera(cam, scale=1):
#     """ resize input in order to produce sampled depth map """
#     new_cam = np.copy(cam)
#     # focal:
#     new_cam[1][0][0] = cam[1][0][0] * scale
#     new_cam[1][1][1] = cam[1][1][1] * scale
#     # principle point:
#     new_cam[1][0][2] = cam[1][0][2] * scale
#     new_cam[1][1][2] = cam[1][1][2] * scale
#     return new_cam
#
# def scale_mvs_camera(cams, scale=1):
#     """ resize input in order to produce sampled depth map """
#     for view in range(FLAGS.view_num):
#         cams[view] = scale_camera(cams[view], scale=scale)
#     return cams
#
# def scale_image(image, scale=1, interpolation='linear'):
#     """ resize image using cv2 """
#     if interpolation == 'linear':
#         return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
#     if interpolation == 'nearest':
#         return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
#
# def scale_mvs_input(images, cams, depth_image=None, scale=1):
#     """ resize input to fit into the memory """
#     for view in range(FLAGS.view_num):
#         images[view] = scale_image(images[view], scale=scale)
#         cams[view] = scale_camera(cams[view], scale=scale)
#
#     if depth_image is None:
#         return images, cams
#     else:
#         depth_image = scale_image(depth_image, scale=scale, interpolation='nearest')
#         return images, cams, depth_image
#
# def crop_mvs_input(images, cams, depth_image=None, max_w=0, max_h=0):
#     """ resize images and cameras to fit the network (can be divided by base image size) """
#
#     # crop images and cameras
#     for view in range(FLAGS.view_num):
#         h, w = images[view].shape[0:2]
#         new_h = h
#         new_w = w
#         if new_h > FLAGS.max_h:
#             new_h = FLAGS.max_h
#         else:
#             new_h = int(math.ceil(h / FLAGS.base_image_size) * FLAGS.base_image_size)
#         if new_w > FLAGS.max_w:
#             new_w = FLAGS.max_w
#         else:
#             new_w = int(math.ceil(w / FLAGS.base_image_size) * FLAGS.base_image_size)
#
#         if max_w > 0:
#             new_w = max_w
#         if max_h > 0:
#             new_h = max_h
#
#         start_h = int(math.ceil((h - new_h) / 2))
#         start_w = int(math.ceil((w - new_w) / 2))
#         finish_h = start_h + new_h
#         finish_w = start_w + new_w
#         images[view] = images[view][start_h:finish_h, start_w:finish_w]
#         cams[view][1][0][2] = cams[view][1][0][2] - start_w
#         cams[view][1][1][2] = cams[view][1][1][2] - start_h
#
#         # crop depth image
#         if not depth_image is None and view == 0:
#             depth_image = depth_image[start_h:finish_h, start_w:finish_w]
#
#     if not depth_image is None:
#         return images, cams, depth_image
#     else:
#         return images, cams

# def mask_depth_image(depth_image, min_depth, max_depth):
#     """ mask out-of-range pixel to zero """
#     # print ('mask min max', min_depth, max_depth)
#     ret, depth_image = cv2.threshold(depth_image, min_depth, 100000, cv2.THRESH_TOZERO)
#     ret, depth_image = cv2.threshold(depth_image, max_depth, 100000, cv2.THRESH_TOZERO_INV)
#     depth_image = np.expand_dims(depth_image, 2)
#     return depth_image

def load_cam(file, interval_scale=1):
    """ read camera txt file """
    cam = np.zeros((2, 4, 4))
    words = file.read().split()
    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            cam[0][i][j] = words[extrinsic_index]

    # read intrinsic
    for i in range(0, 3):
        for j in range(0, 3):
            intrinsic_index = 3 * i + j + 18
            cam[1][i][j] = words[intrinsic_index]

    if len(words) == 29:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = FLAGS.max_d
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
    elif len(words) == 30:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
    elif len(words) == 31:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = words[30]
    else:
        cam[1][3][0] = 0
        cam[1][3][1] = 0
        cam[1][3][2] = 0
        cam[1][3][3] = 0

    return cam

# def write_cam(file, cam):
#     # f = open(file, "w")
#     f = file_io.FileIO(file, "w")
#
#     f.write('extrinsic\n')
#     for i in range(0, 4):
#         for j in range(0, 4):
#             f.write(str(cam[0][i][j]) + ' ')
#         f.write('\n')
#     f.write('\n')
#
#     f.write('intrinsic\n')
#     for i in range(0, 3):
#         for j in range(0, 3):
#             f.write(str(cam[1][i][j]) + ' ')
#         f.write('\n')
#
#     f.write('\n' + str(cam[1][3][0]) + ' ' + str(cam[1][3][1]) + ' ' + str(cam[1][3][2]) + ' ' + str(cam[1][3][3]) + '\n')
#
#     f.close()

# def load_pfm(file):
#     color = None
#     width = None
#     height = None
#     scale = None
#     data_type = None
#     header = file.readline().decode('UTF-8').rstrip()
#
#     if header == 'PF':
#         color = True
#     elif header == 'Pf':
#         color = False
#     else:
#         raise Exception('Not a PFM file.')
#     dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('UTF-8'))
#     if dim_match:
#         width, height = map(int, dim_match.groups())
#     else:
#         raise Exception('Malformed PFM header.')
#     # scale = float(file.readline().rstrip())
#     scale = float((file.readline()).decode('UTF-8').rstrip())
#     if scale < 0: # little-endian
#         data_type = '<f'
#     else:
#         data_type = '>f' # big-endian
#     data_string = file.read()
#     data = np.fromstring(data_string, data_type)
#     shape = (height, width, 3) if color else (height, width)
#     data = np.reshape(data, shape)
#     data = cv2.flip(data, 0)
#     return data
#
# def write_pfm(file, image, scale=1):
#     file = file_io.FileIO(file, mode='wb')
#     color = None
#
#     if image.dtype.name != 'float32':
#         raise Exception('Image dtype must be float32.')
#
#     image = np.flipud(image)
#
#     if len(image.shape) == 3 and image.shape[2] == 3: # color image
#         color = True
#     elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
#         color = False
#     else:
#         raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')
#
#     file.write('PF\n' if color else 'Pf\n')
#     file.write('%d %d\n' % (image.shape[1], image.shape[0]))
#
#     endian = image.dtype.byteorder
#
#     if endian == '<' or endian == '=' and sys.byteorder == 'little':
#         scale = -scale
#
#     file.write('%f\n' % scale)
#
#     image_string = image.tostring()
#     file.write(image_string)
#
#     file.close()

def gen_dtu_resized_path(dtu_data_folder, mode='training'):
    """ generate data paths for dtu dataset """
    sample_list = []

    # parse camera pairs
    cluster_file_path = dtu_data_folder + '/Cameras/pair.txt'

    # cluster_list = open(cluster_file_path).read().split()
    cluster_list = file_io.FileIO(cluster_file_path, mode='r').read().split()

    # 3 sets
    training_set = [2, 6, 7, 8, 14, 16, 18, 19, 20, 22, 30, 31, 36, 39, 41, 42, 44,
                    45, 46, 47, 50, 51, 52, 53, 55, 57, 58, 60, 61, 63, 64, 65, 68, 69, 70, 71, 72,
                    74, 76, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                    101, 102, 103, 104, 105, 107, 108, 109, 111, 112, 113, 115, 116, 119, 120,
                    121, 122, 123, 124, 125, 126, 127, 128]
    validation_set = [3, 5, 17, 21, 28, 35, 37, 38, 40, 43, 56, 59, 66, 67, 82, 86, 106, 117]

    view_index_all = [i for i in range(49)]

    data_set = []
    if mode == 'training':
        data_set = training_set
    elif mode == 'validation':
        data_set = validation_set


    # for each dataset
    for i in data_set:

        image_folder = os.path.join(dtu_data_folder, ('Rectified/scan%d_train' % i))
        cam_folder = os.path.join(dtu_data_folder, 'Cameras/train')
        # depth_folder = os.path.join(dtu_data_folder, ('Depths/scan%d_train' % i))

        if mode == 'training':
            # for each lighting
            for j in range(0, 7):
                # for each reference image
                for p in range(0, int(cluster_list[0])):
                    paths = []
                    # ref image
                    ref_index = int(cluster_list[22 * p + 1])
                    ref_image_path = os.path.join(
                        image_folder, ('rect_%03d_%d_r5000.png' % ((ref_index + 1), j)))
                    ref_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % ref_index))
                    paths.append(ref_image_path)
                    paths.append(ref_cam_path)

                    indexes = []
                    indexes.append(ref_index)

                    # view images
                    for view in range(FLAGS.view_num):
                        if view < 10:
                            view_index = int(cluster_list[22 * p + 2 * view + 3])
                            indexes.append(view_index)
                        else:
                            view_list = [index for index in view_index_all if index not in indexes]
                            view_index = view_list[0]
                            indexes.append(view_index)
                        view_image_path = os.path.join(
                            image_folder, ('rect_%03d_%d_r5000.png' % ((view_index + 1), j)))
                        view_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % view_index))
                        paths.append(view_image_path)
                        paths.append(view_cam_path)

                    # # depth path
                    # depth_image_path = os.path.join(depth_folder, ('depth_map_%04d.pfm' % ref_index))
                    # paths.append(depth_image_path)
                    sample_list.append(paths)
        elif mode == 'validation':
            j = 3
            # for each reference image
            for p in range(0, int(cluster_list[0])):
                paths = []
                # ref image
                ref_index = int(cluster_list[22 * p + 1])
                ref_image_path = os.path.join(
                    image_folder, ('rect_%03d_%d_r5000.png' % ((ref_index + 1), j)))
                ref_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % ref_index))
                paths.append(ref_image_path)
                paths.append(ref_cam_path)

                # view images
                for view in range(FLAGS.view_num):
                    view_index = int(cluster_list[22 * p + 2 * view + 3])
                    view_image_path = os.path.join(
                        image_folder, ('rect_%03d_%d_r5000.png' % ((view_index + 1), j)))
                    view_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % view_index))
                    paths.append(view_image_path)
                    paths.append(view_cam_path)

                # # depth path
                # depth_image_path = os.path.join(depth_folder, ('depth_map_%04d.pfm' % ref_index))
                # paths.append(depth_image_path)
                sample_list.append(paths)

    return sample_list


def gen_TanksandTemples_list(data_folder='/media/yujiao/6TB/dataset/MVS/FreeViewSynthesis/ibr3d_tat', mode='training'):
    '''
    :param data_folder:
    :param mode:
    :return:
    '''
    validation = {'training/Truck':
                      [172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196],
                  'intermediate/M60':
                      [94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129],
                  'intermediate/Playground':
                      [221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252],
                  'intermediate/Train':
                      [174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248]}

    sample_list = []

    if mode=='training':

        dirs1 = os.listdir(data_folder)   #['intermediate', 'training', 'advanced']
        for dir1 in dirs1:
            root_dir1 = os.path.join(data_folder, dir1)
            dirs2 = os.listdir(root_dir1)
            for data_name in dirs2:

                if os.path.join(dir1, data_name) not in validation.keys():

                    file_folder = os.path.join(root_dir1, data_name, 'dense', 'ibr3d_pw_0.25')

                    Ks = os.path.join(file_folder, 'Ks.npy')
                    Rs = os.path.join(file_folder, 'Rs.npy')
                    ts = os.path.join(file_folder, 'ts.npy')

                    count_dict = {}
                    im_dict = {}

                    files = os.listdir(file_folder)
                    for file in files:
                        if file not in ['Ks.npy', 'Rs.npy', 'ts.npy']:
                            name, index = (file.split('.')[0]).split('_')
                            if name=='count':
                                count_dict[index] = os.path.join(file_folder, file)
                            elif name=='im':
                                im_dict[index] = os.path.join(file_folder, file)

                    assert im_dict.keys() == count_dict.keys()

                    indexes = im_dict.keys()
                    for index in indexes:
                        tgt_img_dict = {index: im_dict[index]}
                        tgt_count_dict = {index: count_dict[index]}

                        src_img_dict = {key: value for key, value in im_dict.items() if key!= index}

                        sample_list.append([tgt_img_dict, tgt_count_dict, src_img_dict, Ks, Rs, ts])


    else:

        if mode == 'Truck':
            data_name = 'training/Truck'

        elif mode == 'M60':
            data_name = 'intermediate/M60'

        elif mode == 'Playground':
            data_name = 'intermediate/Playground'

        elif mode == 'Train':
            data_name = 'intermediate/Train'

        file_folder = os.path.join(data_folder, data_name, 'dense', 'ibr3d_pw_0.25')

        count_dict = {}
        dm_dict = {}
        im_dict = {}

        files = os.listdir(file_folder)
        for file in files:
            if file not in ['Ks.npy', 'Rs.npy', 'ts.npy']:
                name, index = (file.split('.')[0]).split('_')
                if name == 'count':
                    count_dict[index] = os.path.join(file_folder, file)
                elif name == 'im':
                    im_dict[index] = os.path.join(file_folder, file)

        Ks = os.path.join(file_folder, 'Ks.npy')
        Rs = os.path.join(file_folder, 'Rs.npy')
        ts = os.path.join(file_folder, 'ts.npy')

        for item in validation[data_name]:
            index = '%08d'%item
            tgt_img_dict = {index: im_dict[index]}
            tgt_count_dict = {index: count_dict[index]}

            src_img_dict = {key: value for key, value in im_dict.items() if key != index}

            sample_list.append([tgt_img_dict, tgt_count_dict, src_img_dict, Ks, Rs, ts])

    return sample_list



# def gen_dtu_mvs_path(dtu_data_folder, mode='training'):
#     """ generate data paths for dtu dataset """
#     sample_list = []
#
#     # parse camera pairs
#     cluster_file_path = dtu_data_folder + '/Cameras/pair.txt'
#     cluster_list = open(cluster_file_path).read().split()
#
#     # 3 sets
#     training_set = [2, 6, 7, 8, 14, 16, 18, 19, 20, 22, 30, 31, 36, 39, 41, 42, 44,
#                     45, 46, 47, 50, 51, 52, 53, 55, 57, 58, 60, 61, 63, 64, 65, 68, 69, 70, 71, 72,
#                     74, 76, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
#                     101, 102, 103, 104, 105, 107, 108, 109, 111, 112, 113, 115, 116, 119, 120,
#                     121, 122, 123, 124, 125, 126, 127, 128]
#     validation_set = [3, 5, 17, 21, 28, 35, 37, 38, 40, 43, 56, 59, 66, 67, 82, 86, 106, 117]
#     evaluation_set = [1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77,
#                       110, 114, 118]
#
#     # for each dataset
#     data_set = []
#     if mode == 'training':
#         data_set = training_set
#     elif mode == 'validation':
#         data_set = validation_set
#     elif mode == 'evaluation':
#         data_set = evaluation_set
#
#     # for each dataset
#     for i in data_set:
#
#         image_folder = os.path.join(dtu_data_folder, ('Rectified/scan%d' % i))
#         cam_folder = os.path.join(dtu_data_folder, 'Cameras')
#         depth_folder = os.path.join(dtu_data_folder, ('Depths/scan%d' % i))
#
#         if mode == 'training':
#             # for each lighting
#             for j in range(0, 7):
#                 # for each reference image
#                 for p in range(0, int(cluster_list[0])):
#                     paths = []
#                     # ref image
#                     ref_index = int(cluster_list[22 * p + 1])
#                     ref_image_path = os.path.join(
#                         image_folder, ('rect_%03d_%d_r5000.png' % ((ref_index + 1), j)))
#                     ref_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % ref_index))
#                     paths.append(ref_image_path)
#                     paths.append(ref_cam_path)
#                     # view images
#                     for view in range(FLAGS.view_num):
#                         view_index = int(cluster_list[22 * p + 2 * view + 3])
#                         view_image_path = os.path.join(
#                             image_folder, ('rect_%03d_%d_r5000.png' % ((view_index + 1), j)))
#                         view_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % view_index))
#                         paths.append(view_image_path)
#                         paths.append(view_cam_path)
#                     # depth path
#                     depth_image_path = os.path.join(depth_folder, ('depth_map_%04d.pfm' % ref_index))
#                     paths.append(depth_image_path)
#                     sample_list.append(paths)
#         else:
#             # for each reference image
#             j = 5
#             for p in range(0, int(cluster_list[0])):
#                 paths = []
#                 # ref image
#                 ref_index = int(cluster_list[22 * p + 1])
#                 ref_image_path = os.path.join(
#                     image_folder, ('rect_%03d_%d_r5000.png' % ((ref_index + 1), j)))
#                 ref_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % ref_index))
#                 paths.append(ref_image_path)
#                 paths.append(ref_cam_path)
#                 # view images
#                 for view in range(FLAGS.view_num):
#                     view_index = int(cluster_list[22 * p + 2 * view + 3])
#                     view_image_path = os.path.join(
#                         image_folder, ('rect_%03d_%d_r5000.png' % ((view_index + 1), j)))
#                     view_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % view_index))
#                     paths.append(view_image_path)
#                     paths.append(view_cam_path)
#                 # depth path
#                 depth_image_path = os.path.join(depth_folder, ('depth_map_%04d.pfm' % ref_index))
#                 paths.append(depth_image_path)
#                 sample_list.append(paths)
#
#     return sample_list

#
# from PIL import Image
#
# def write_depth_img(filename,depth):
#
#     if not os.path.exists(os.path.dirname(filename)):
#         try:
#             os.makedirs(os.path.dirname(filename))
#         except OSError as exc: # Guard against race condition
#             if exc.errno != errno.EEXIST:
#                 raise
#
#     image = Image.fromarray(((depth-0.5)/(100-0.5)*255).astype(np.uint8)).convert("L")
#     image.save(filename)
#     return 1

