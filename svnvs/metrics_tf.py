import numpy as np
# from skimage.measure import compare_ssim, compare_psnr, compare_mse, compare_nrmse
import cv2

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
from tensorflow.python.ops import math_ops

import argparse

dataset_name = 'dtu'

tf.app.flags.DEFINE_string('dataset_name', 'TanksandTemples',
                            """TanksandTemples, DTU""")
tf.app.flags.DEFINE_string('output_dir', './output',
                           """Path to save the model.""")
# input parameters
tf.app.flags.DEFINE_integer('view_num', 6,
                            """Number of images (1 ref image and view_num - 1 view images).""")
tf.app.flags.DEFINE_integer('max_d', 48,
                            """Maximum depth step when training.""")
tf.app.flags.DEFINE_integer('max_w', 448,
                            """Maximum image width when training.""")
tf.app.flags.DEFINE_integer('max_h', 256,
                            """Maximum image height when training.""")

tf.app.flags.DEFINE_string('dataset', 'Truck',
                           """Truck, M60, Playground, Train;
                           .""")

FLAGS = tf.app.flags.FLAGS


tgtimg_dir = os.path.join(FLAGS.output_dir, FLAGS.dataset_name, 'tgtimgs',
                          str(FLAGS.view_num) + '_' + str(FLAGS.max_d) + '_' + FLAGS.dataset)


outimg_dir = os.path.join(FLAGS.output_dir, FLAGS.dataset_name, 'outimgs',
                          str(FLAGS.view_num) + '_' + str(FLAGS.max_d) + '_' + FLAGS.dataset)


print(outimg_dir)
print(tgtimg_dir)

def safe_divide(numerator, denominator, name='safe_divide'):

    return tf.where(math_ops.greater(denominator, 0),
                    math_ops.divide(numerator, denominator),
                    tf.zeros_like(numerator), name=name)


def RMSE(input, target):
    return tf.sqrt(tf.reduce_mean((input - target)**2, axis=(1, 2, 3)))



def input_data_generator(input_dir, target_dir, batch_size=1):
    img_list = os.listdir(input_dir)

    num_batches = len(img_list)//batch_size

    for i in range(num_batches + 1):

        input_list = []
        target_list = []

        img_num_per_batch = batch_size if i<num_batches else len(img_list)%batch_size

        for j in range(img_num_per_batch):
            input_list.append((cv2.imread(os.path.join(input_dir, img_list[i*batch_size+j]))).astype(np.float32)) # [64:, ...]

            target_img = cv2.imread(os.path.join(target_dir, img_list[i*batch_size+j]))
            target_list.append((cv2.resize(target_img, (FLAGS.max_w, FLAGS.max_h))).astype(np.float32))  #
            # target_list.append((cv2.resize(target_img, (1024, 256))).astype(np.float32)[:, :256, :])  # [64:, ...]

        yield np.stack(input_list, axis=0), np.stack(target_list, axis=0)


def get_evaluation_metrics(inputs, targets):

    # inputs = tf.stack(input_list, axis=0)
    # targets = tf.stack(target_list, axis=0)

    ssim = tf.reduce_sum(tf.image.ssim(inputs, targets, max_val=255))
    psnr = tf.reduce_sum(tf.image.psnr(inputs, targets, max_val=255))
    rmse = tf.reduce_sum(RMSE(inputs, targets))

    return ssim, psnr, rmse


if __name__=="__main__":
    batch_size = 128
   
    data_generator = input_data_generator(input_dir=outimg_dir, target_dir=tgtimg_dir, batch_size=batch_size)

    inputs = tf.placeholder(tf.float32, [None, 256, FLAGS.max_w, 3], name='inputs')
    targets = tf.placeholder(tf.float32, [None, 256, FLAGS.max_w, 3], name='targets')
    ssim, psnr, rmse = get_evaluation_metrics(inputs, targets)

    ssim_sum = 0
    psnr_sum = 0
    rmse_sum = 0

    i = 0

    sess = tf.Session()
    for batch_inputs, batch_targets in data_generator:
        feed_dict = {inputs: batch_inputs, targets:batch_targets}
        ssim_val, psnr_val, rmse_val = sess.run([ssim, psnr, rmse], feed_dict=feed_dict)

        ssim_sum += ssim_val
        psnr_sum += psnr_val
        rmse_sum += rmse_val
        i += batch_inputs.shape[0]
    ssim_mean = ssim_sum/i
    psnr_mean = psnr_sum/i
    rmse_mean = rmse_sum/i

    print('===============================================================================')
    print(ssim_mean, psnr_mean, rmse_mean, i)
    print('===============================================================================')

