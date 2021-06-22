import numpy as np
import time
from svnvs.preprocess import load_cam
from PIL import Image
import tensorflow as tf
import cv2

FLAGS = tf.app.flags.FLAGS


# def compute_depth_range(list_data, depth_num):
#     hist, bins = np.histogram(list_data, bins=depth_num)
#     n_data = len(list_data)
#     threshold_max = n_data * 0.98
#     threshold_min = n_data * 0.02
#     sum_hist = 0
#     min_depth = np.min(list_data)
#     max_depth = np.max(list_data)
#     # print('min: %f / max: %f (before histogram)' % (min_depth, max_depth))
#
#     min_found = False
#     for bin_idx, hist_val in enumerate(hist):
#         sum_hist += hist_val
#         if not min_found and sum_hist > threshold_min:
#             if bin_idx >= 1:
#                 min_depth = bins[bin_idx - 1]
#             else:
#                 min_depth = bins[bin_idx]
#             min_found = True
#
#         if sum_hist > threshold_max:
#             max_depth = bins[bin_idx + 1]
#             break
#
#     depth_interval = (max_depth - min_depth)/(depth_num)
#
#     return min_depth, max_depth, depth_interval



def scale_img_cam(img, cam, new_height, new_width):
    '''
    input:
    img and depth have the same shape, cam intrinsics corresponds to the current image shape
    output:
    resize img and depth to a new shape, cam intrinsics corresponds to half new image shape

    :param img: PIL.Image object
    :param cam: shape=[2, 4, 4]
    :param new_height: 256
    :param new_width: 320
    :return:
    '''

    old_width, old_height = img.size

    img = img.resize((new_width, new_height))
    img = np.array(img, np.float32)

    scale_w = new_width/old_width
    scale_h = new_height/old_height

    new_cam = np.copy(cam)
    # focal:
    new_cam[1][0][0] = cam[1][0][0] * scale_w * 0.5
    new_cam[1][1][1] = cam[1][1][1] * scale_h * 0.5
    # principle point:
    new_cam[1][0][2] = cam[1][0][2] * scale_w * 0.5
    new_cam[1][1][2] = cam[1][1][2] * scale_h * 0.5

    return img, new_cam


class DTUGenerator:
    """ data generator class, tf only accept generator without param """

    def __init__(self, sample_list, view_num):
        self.sample_list = sample_list
        self.view_num = view_num
        self.sample_num = len(sample_list)
        self.counter = 0

    def __iter__(self):
        while True:
            for data in self.sample_list:

                ###### read input data ######
                src_images = []
                src_cams = []

                tgt_image = Image.open(data[0]).resize((320, 256))
                tgt_image = np.asarray(tgt_image)
                tgt_image = tgt_image / 255. * 2 - 1
                tgt_cam = load_cam(open(data[1]))

                for view in range(self.view_num):
                    image = Image.open(data[2 * (view + 1)]).resize((320, 256))
                    image = np.asarray(image)
                    image = image/255.*2 - 1
                    cam = load_cam(open(data[2 * (view + 1) + 1]))
                    src_images.append(image)
                    src_cams.append(cam)

                # set depth range to [425, 937]
                tgt_cam[1, 3, 0] = 425
                tgt_cam[1, 3, 3] = 937

                # skip invalid views
                if tgt_cam[1, 3, 0] <= 0 or tgt_cam[1, 3, 3] <= 0:
                    continue

                # fix depth range and adapt depth sample number
                tgt_cam[1, 3, 2] = FLAGS.max_d
                tgt_cam[1, 3, 1] = (tgt_cam[1, 3, 3] - tgt_cam[1, 3, 0]) / FLAGS.max_d

                # return mvs input
                self.counter += 1
                src_images = np.stack(src_images, axis=0)
                src_cams = np.stack(src_cams, axis=0)

                yield (src_images, src_cams, tgt_image, tgt_cam)


class TanksandTemplesGenerator:

    def __init__(self, sample_list, view_num):

        self.sample_list = sample_list
        self.view_num = view_num

    def __iter__(self):
        while True:

            for data in self.sample_list:
                tgt_img_dict, tgt_count_dict, src_img_dict, Ks, Rs, Ts = data[:]

                ks = np.load(Ks)  # shape=[N, 3, 3]
                rs = np.load(Rs)  # shape=[N, 3, 3]
                ts = np.load(Ts)  # shape=[N, 3]

                for key, value in tgt_img_dict.items():
                    tgt_index = int(key)
                    tgt_k = ks[tgt_index]
                    tgt_r = rs[tgt_index]
                    tgt_t = ts[tgt_index]

                    tgt_cam = np.zeros((2, 4, 4))
                    tgt_cam[0, 0:3, 0:3] = tgt_r
                    tgt_cam[0, 0:3, 3] = tgt_t
                    tgt_cam[0, 3, 3] = 1.
                    tgt_cam[1, 0:3, 0:3] = tgt_k
                    tgt_cam = tgt_cam.astype(np.float32)

                    tgt_img = Image.open(value)

                    tgt_image, tgt_camra = scale_img_cam(tgt_img, tgt_cam, FLAGS.max_h, FLAGS.max_w)
                    # shape = [height, width, 3], shape = [height, width], shape=[2, 4, 4]

                    tgt_image = tgt_image / 255.*2. - 1.  #[0, 255] --> [-1, 1]

                    # =================== Load source images ================================================
                    list_data = []
                    count = np.argsort(np.load(tgt_count_dict[key]))[::-1]

                    count = count[:self.view_num]
                    count = count[np.random.permutation(self.view_num)]

                    src_imgs = []
                    src_cams = []

                    for i in range(self.view_num):
                        src_index = count[i]
                        src_key = '%08d' % src_index

                        if src_key==key:
                            continue

                        src_k = ks[src_index]
                        src_r = rs[src_index]
                        src_t = ts[src_index]

                        src_cam = np.zeros((2, 4, 4))
                        src_cam[0, 0:3, 0:3] = src_r
                        src_cam[0, 0:3, 3] = src_t
                        src_cam[0, 3, 3] = 1.
                        src_cam[1, 0:3, 0:3] = src_k
                        src_cam[1, 3, 3] = 1.
                        src_cam = src_cam.astype(np.float32)

                        src_img = Image.open(src_img_dict[src_key])
                        src_img, src_cam = scale_img_cam(src_img, src_cam, FLAGS.max_h, FLAGS.max_w)

                        src_imgs.append(src_img)
                        src_cams.append(src_cam)

                    src_image = np.stack(src_imgs, axis=0)  # shape = [N, height, width, 3]
                    src_image = src_image / 255. * 2 - 1
                    src_camera = np.stack(src_cams, axis=0)  # shape = [N, 2, 4, 4]

                    depth_min = 0.5
                    depth_max = 100
                    depth_interval = (depth_max - depth_min)/FLAGS.max_d

                    tgt_camra[1, 3, 0] = depth_min
                    tgt_camra[1, 3, 1] = depth_interval
                    tgt_camra[1, 3, 2] = FLAGS.max_d
                    tgt_camra[1, 3, 3] = depth_max


                    yield src_image, src_camera, tgt_image, tgt_camra


