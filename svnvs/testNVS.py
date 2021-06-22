from __future__ import print_function

import os

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import time
import sys

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import matplotlib.pyplot as plt

sys.path.append("../")
from tools.common import Notify

from svnvs.preprocess import *
from svnvs.FNVSmodel import *
from svnvs.loss import *
from svnvs.data_generator import *

from Perloss.perceptual_loss import perceptual_loss

from PIL import Image


# paths
tf.app.flags.DEFINE_string('data_root_dir', '/media/yujiao/6TB/dataset/MVS/',
                            """dataset root dir""")
tf.app.flags.DEFINE_string('dataset_name', 'TanksandTemples',
                            """TanksandTemples, DTU""")
tf.app.flags.DEFINE_string('log_folder', 'tf_log',
                           """Path to store the log.""")
tf.app.flags.DEFINE_string('model_folder', 'tf_model',
                           """Path to save the model.""")
tf.app.flags.DEFINE_string('output_dir', './output',
                           """Path to save the model.""")

# input parameters
tf.app.flags.DEFINE_integer('view_num', 6,
                            """Number of images.""")
tf.app.flags.DEFINE_integer('max_d', 48,
                            """Maximum depth step when training.""")
tf.app.flags.DEFINE_integer('max_w', 448,
                            """Maximum image width when training.""")
tf.app.flags.DEFINE_integer('max_h', 256,
                            """Maximum image height when training.""")


tf.app.flags.DEFINE_integer('batch_size', 1,
                            """Training batch size.""")
tf.app.flags.DEFINE_integer('display', 20,
                            """Interval of loginfo display.""")

tf.app.flags.DEFINE_string('dataset', 'Truck',
                           """Train, Truck, M60, Playground; scan65, scan106, scan118""")


FLAGS = tf.app.flags.FLAGS


def makedirs(director):
    if not os.path.exists(director):
        os.makedirs(director)


def test():
    if FLAGS.dataset_name == 'DTU':
        data_folder = os.path.join(FLAGS.data_root_dir, 'yaoyao/dtu-yao')
        testing_list = gen_dtu_resized_path(data_folder, mode='validation')
    elif FLAGS.dataset_name == 'TanksandTemples':
        data_folder = os.path.join(FLAGS.data_root_dir, 'FreeViewSynthesis/ibr3d_tat')
        testing_list = gen_TanksandTemples_list(data_folder, mode=FLAGS.dataset)

    generator_name = FLAGS.dataset_name + 'Generator'

    testing_sample_size = len(testing_list)
    print('Testing sample number: ', testing_sample_size)


    model_folder = os.path.join('./NVSmodel', FLAGS.dataset_name, FLAGS.model_folder,
                                str(FLAGS.view_num) + '_' + str(FLAGS.max_d))

    tgtimg_dir = os.path.join(FLAGS.output_dir, FLAGS.dataset_name, 'tgtimgs',
                              str(FLAGS.view_num) + '_' + str(FLAGS.max_d) + '_' + FLAGS.dataset)

    outimg_dir = os.path.join(FLAGS.output_dir, FLAGS.dataset_name, 'outimgs',
                              str(FLAGS.view_num) + '_' + str(FLAGS.max_d) + '_' + FLAGS.dataset)

    warpimg_dir = os.path.join(FLAGS.output_dir, FLAGS.dataset_name, 'warpimgs',
                              str(FLAGS.view_num) + '_' + str(FLAGS.max_d) + '_' + FLAGS.dataset)

    warpsrc_dir = os.path.join(FLAGS.output_dir, FLAGS.dataset_name, 'warpsrcs',
                              str(FLAGS.view_num) + '_' + str(FLAGS.max_d) + '_' + FLAGS.dataset)

    makedirs(tgtimg_dir)
    makedirs(outimg_dir)
    makedirs(warpimg_dir)
    makedirs(warpsrc_dir)


    with tf.Graph().as_default():

        ########## data iterator #########
        # training generators
        testing_generator = iter(eval(generator_name)(testing_list, FLAGS.view_num))
        generator_data_type = (tf.float32, tf.float32, tf.float32, tf.float32)
        # dataset from generator
        testing_set = tf.data.Dataset.from_generator(lambda: testing_generator, generator_data_type)
        testing_set = testing_set.batch(FLAGS.batch_size)
        testing_set = testing_set.prefetch(buffer_size=1)
        # iterators
        testing_iterator = testing_set.make_initializable_iterator()

        ########## optimization options ##########
        global_step = tf.Variable(0, trainable=False, name='global_step')

        with tf.name_scope('Model_tower') as scope:
            # get data
            src_images, src_cams, tgt_image, tgt_cam = testing_iterator.get_next()

            src_images.set_shape(tf.TensorShape([None, FLAGS.view_num, None, None, 3]))
            src_cams.set_shape(tf.TensorShape([None, FLAGS.view_num, 2, 4, 4]))

            tgt_image.set_shape(tf.TensorShape([None, None, None, 3]))
            tgt_cam.set_shape(tf.TensorShape([None, 2, 4, 4]))

            depth_start = tf.reshape(
                tf.slice(tgt_cam, [0, 1, 3, 0], [FLAGS.batch_size, 1, 1, 1]), [FLAGS.batch_size])
            depth_interval = tf.reshape(
                tf.slice(tgt_cam, [0, 1, 3, 1], [FLAGS.batch_size, 1, 1, 1]), [FLAGS.batch_size])

            depth_probs, src_weights = \
                DepthModelinv(src_images, src_cams, tgt_cam, FLAGS.max_d, depth_start, depth_interval)

            aggregated_img, final_output_img, warped_imgs_srcs \
                = \
                NVSmodel(src_images, src_cams, tgt_cam, depth_probs, src_weights,
                         FLAGS.max_d, depth_start, depth_interval)
            # [B, H, W, 3], [B, H, W, 3], [N, B, H, W, 3]

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:


            saver.restore(sess, os.path.join(model_folder, 'model.ckpt'))


            step = 0
            sess.run(testing_iterator.initializer)
            for ii in range(int(testing_sample_size)):

                # run one batch
                start_time = time.time()
                try:
                    oaggregated_image, otgt_image, output_images, oaggregated_img_srcs, osrc_images, \
                        = sess.run([aggregated_img, tgt_image, final_output_img, warped_imgs_srcs, src_images])
                    # [N, B, H, W, D], [N, B, H, W, 1], [N, B, H, W, 1]

                    duration = time.time() - start_time

                    print(Notify.INFO, '(%.3f sec/step)' % duration, Notify.ENDC)

                    warpedimg = ((oaggregated_image[0]+1)/2*255).astype(np.uint8)
                    output_images = ((output_images+1)/2*255).astype(np.uint8)
                    warpedsrc = ((oaggregated_img_srcs[:, 0,...]+1)/2*255).astype(np.uint8) # [N, H, W, 3]

                    img = Image.fromarray(warpedimg)
                    img.save(os.path.join(warpimg_dir, str(step) + '.png'))

                    otgt_image = ((otgt_image + 1)/2*255).astype(np.uint8)
                    img = Image.fromarray(otgt_image[0]).resize((FLAGS.max_w, FLAGS.max_h))
                    img.save(os.path.join(tgtimg_dir, str(step) + '.png'))

                    img = Image.fromarray(output_images[0])
                    img.save(os.path.join(outimg_dir, str(step) + '.png'))

                    for i in range(FLAGS.view_num):
                        img = Image.fromarray(warpedsrc[i])
                        img.save(os.path.join(warpsrc_dir, str(step) + '_' + str(i) + '.png'))


                except tf.errors.OutOfRangeError:
                    print("End of dataset")  # ==> "End of dataset"
                    break

                # print info
                if step % FLAGS.display == 0:
                    print(step)

                step += 1


def main(argv=None):  # pylint: disable=unused-argument

    test()


if __name__ == '__main__':
    tf.app.run()

