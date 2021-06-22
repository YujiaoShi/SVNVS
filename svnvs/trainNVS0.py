'''
unsupervised depth training
'''

from __future__ import print_function

import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


import sys
import random
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


sys.path.append("../")
from tools.common import Notify

from svnvs.preprocess import gen_dtu_resized_path, gen_TanksandTemples_list
from svnvs.FNVSmodel import *
# from svnvs.loss import *
from svnvs.data_generator import *

from Perloss.perceptual_loss import perceptual_loss


# paths
tf.app.flags.DEFINE_string('data_root_dir', '/media/yujiao/6TB/dataset/MVS/',
                            """dataset root dir""")
tf.app.flags.DEFINE_string('dataset_name', 'TanksandTemples',
                            """TanksandTemples, DTU""")
tf.app.flags.DEFINE_string('log_folder', 'tf_log',
                           """Path to store the log.""")
tf.app.flags.DEFINE_string('model_folder', 'tf_model',
                           """Path to save the model.""")

# input parameters
tf.app.flags.DEFINE_integer('view_num', 4,
                            """Number of images.""")
tf.app.flags.DEFINE_integer('max_d', 48,
                            """Maximum depth step when training.""")
tf.app.flags.DEFINE_integer('max_w', 448,
                            """Maximum image width when training.""")
tf.app.flags.DEFINE_integer('max_h', 256,
                            """Maximum image height when training.""")

# training parameters
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """Training batch size.""")
tf.app.flags.DEFINE_integer('epoch', 10,
                            """Training epoch number.""")
# tf.app.flags.DEFINE_float('base_lr', 0.001,
#                           """Base learning rate.""")
tf.app.flags.DEFINE_integer('display', 20,
                            """Interval of loginfo display.""")
# tf.app.flags.DEFINE_integer('stepvalue', 10000,
#                             """Step interval to decay learning rate.""")
tf.app.flags.DEFINE_integer('snapshot', 2000,
                            """Step interval to save the model.""")
# tf.app.flags.DEFINE_float('gamma', 0.9,
#                           """Learning rate decay rate.""")

# training parameters for adam optimizer
tf.app.flags.DEFINE_float('lr', 0.0002,
                          """initial learning rate for adam.""")
tf.app.flags.DEFINE_float('beta1', 0.5,
                          """momentum term of adam.""")

FLAGS = tf.app.flags.FLAGS



def train():

    if FLAGS.dataset_name == 'DTU':
        data_folder = os.path.join(FLAGS.data_root_dir, '/dtu-yao')
        traning_list = gen_dtu_resized_path(data_folder, mode='training')
    elif FLAGS.dataset_name == 'TanksandTemples':
        data_folder = os.path.join(FLAGS.data_root_dir, 'FreeViewSynthesis/ibr3d_tat')
        traning_list = gen_TanksandTemples_list(data_folder, mode='training')

    for i in range(20):
        random.shuffle(traning_list)

    generator_name = FLAGS.dataset_name + 'Generator'

    training_sample_size = len(traning_list)
    print('Training sample number: ', training_sample_size)

    pretrained_model_folder = os.path.join('./NVSmodel', FLAGS.dataset_name, FLAGS.model_folder,
                                str(FLAGS.view_num) + '_' + str(FLAGS.max_d))

    model_folder = os.path.join('./NVSmodel', FLAGS.dataset_name, FLAGS.model_folder,
                                           str(FLAGS.view_num) + '_' + str(FLAGS.max_d))

    log_folder = os.path.join('./NVSmodel', FLAGS.dataset_name, FLAGS.log_folder,
                              str(FLAGS.view_num) + '_' + str(FLAGS.max_d))


    with tf.Graph().as_default():

        ########## data iterator #########
        # training generators
        training_generator = iter(eval(generator_name)(traning_list, FLAGS.view_num))
        generator_data_type = (tf.float32, tf.float32, tf.float32, tf.float32)
        # dataset from generator
        training_set = tf.data.Dataset.from_generator(lambda: training_generator, generator_data_type)
        training_set = training_set.batch(FLAGS.batch_size)
        training_set = training_set.prefetch(buffer_size=1)
        # iterators
        training_iterator = training_set.make_initializable_iterator()


        with tf.name_scope('Model_tower') as scope:
            # get data
            src_images, src_cams, tgt_image, tgt_cam = training_iterator.get_next()

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

            with tf.name_scope("real_discriminator"):
                with tf.variable_scope("discriminator"):
                    predict_real = create_discriminator(tgt_image)

            with tf.name_scope("fake_discriminator_grd"):
                with tf.variable_scope("discriminator", reuse=True):
                    predict_fake = create_discriminator(final_output_img)

            with tf.name_scope("discriminator_loss"):
                # minimizing -tf.log will try to get inputs to 1
                # predict_real => 1
                # predict_fake => 0
                # discrim_loss = 0
                # for predict_fake in predict_fakes:
                discrim_loss = 0.5 * (
                    tf.reduce_mean(-(tf.log(predict_real + 1e-12) + tf.log(1 - predict_fake + 1e-12))))


            with tf.name_scope("generator_loss"):
                # predict_fake => 1
                # abs(targets - outputs) => 0
                gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + 1e-12))
                gen_loss_L1 = tf.reduce_mean(tf.abs(tgt_image - final_output_img))
                gen_loss_perceptual = perceptual_loss(tgt_image, final_output_img)
                gen_loss = gen_loss_GAN + gen_loss_perceptual


            with tf.name_scope("discriminator_train"):
                discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
                discrim_optim = tf.train.AdamOptimizer(FLAGS.lr, FLAGS.beta1)
                discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
                discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

            with tf.name_scope("generator_train"):
                with tf.control_dependencies([discrim_train]):
                    gen_tvars = [var for var in tf.trainable_variables() if var not in discrim_tvars]
                    gen_optim = tf.train.AdamOptimizer(FLAGS.lr, FLAGS.beta1)
                    gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
                    gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

            ema = tf.train.ExponentialMovingAverage(decay=0.99)
            update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1, gen_loss_perceptual])

            # retain the summaries from the final tower.
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

            with tf.name_scope("targets_img_summary"):
                summaries.append(tf.summary.image("tgt_image", tf.image.convert_image_dtype((tgt_image+1)/2,
                                                                                            dtype=tf.uint8)))

            with tf.name_scope("outputs_img_summary"):
                summaries.append(tf.summary.image("out_image", tf.image.convert_image_dtype((final_output_img+1)/2,
                                                                                            dtype=tf.uint8)))

            with tf.name_scope("warped_img_summary"):
                summaries.append(tf.summary.image("warped_image", tf.image.convert_image_dtype((aggregated_img+1)/2,
                                                                                            dtype=tf.uint8)))

                for src_index in range(FLAGS.view_num):
                    summaries.append(
                        tf.summary.image("warped_image_" + str(src_index), tf.image.convert_image_dtype((warped_imgs_srcs[src_index] + 1) / 2,
                                                                                      dtype=tf.uint8)))



            for src_i in range(0, FLAGS.view_num):
                summaries.append(
                    tf.summary.image("src_image_" + str(src_i),
                                     tf.image.convert_image_dtype((src_images[:, src_i, ...]+1)/2.,
                                                                  dtype=tf.uint8)))

        train_opt = tf.group(update_losses, gen_train)

        # summary
        summaries.append(tf.summary.scalar('discriminator_loss', discrim_loss))
        summaries.append(tf.summary.scalar('generator_loss_GAN', gen_loss_GAN))
        summaries.append(tf.summary.scalar('generator_loss_L1', gen_loss_L1))
        summaries.append(tf.summary.scalar('gen_loss_perceptual', gen_loss_perceptual))

        weights_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for var in weights_list:
            summaries.append(tf.summary.histogram(var.op.name, var))
        for grad, var in discrim_grads_and_vars:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
        for grad, var in gen_grads_and_vars:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        # saver

        # depth_saver = tf.train.Saver(g_vars_depth)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
        summary_op = tf.summary.merge(summaries)

        # initialization option
        init_op = tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            # initialization
            total_step = 0
            sess.run(init_op)
            summary_writer = tf.summary.FileWriter(log_folder, sess.graph)

            if os.path.exists(pretrained_model_folder):
                pretrained_model_path = os.path.join(pretrained_model_folder, 'model.ckpt')
                restorer = tf.train.Saver(tf.global_variables())
                restorer.restore(sess, pretrained_model_path)
                print('restore pretrained model from ', pretrained_model_path)

            # training several epochs
            for epoch in range(FLAGS.epoch):

                # training of one epoch
                step = 0
                sess.run(training_iterator.initializer)
                for _ in range(int(training_sample_size)):

                    # run one batch
                    start_time = time.time()
                    try:

                        out_summary_op, out_opt, out_discrim_loss, out_gen_loss_GAN, out_gen_loss_L1\
                            = sess.run(
                            [summary_op, train_opt, discrim_loss, gen_loss_GAN, gen_loss_L1])
                    except tf.errors.OutOfRangeError:
                        print("End of dataset")  # ==> "End of dataset"
                        break
                    duration = time.time() - start_time

                    # print info
                    if step % FLAGS.display == 0:
                        print(Notify.INFO,
                              'epoch, %d, step %d, discrim_loss = %.4f, '
                              'gen_loss_GAN = %.4f, gen_loss_L1 = %.4f (%.3f sec/step)' %
                              (epoch, step, out_discrim_loss, out_gen_loss_GAN, out_gen_loss_L1, duration), Notify.ENDC)

                    # write summary
                    if step % (FLAGS.display) == 0:
                        summary_writer.add_summary(out_summary_op, total_step)

                    # save the model checkpoint periodically
                    if (total_step % FLAGS.snapshot == 0 or step == (training_sample_size - 1)):
                        model_folder = os.path.join(model_folder)
                        if not os.path.exists(model_folder):
                            os.makedirs(model_folder)
                        ckpt_path = os.path.join(model_folder, 'model.ckpt')
                        print(Notify.INFO, 'Saving model to %s' % ckpt_path, Notify.ENDC)
                        saver.save(sess, ckpt_path)
                    step += FLAGS.batch_size
                    total_step += FLAGS.batch_size


def main(argv=None):

    train()


if __name__ == '__main__':
    print('Training SVNVS with totally %d views inputs' % FLAGS.view_num)
    tf.app.run()

