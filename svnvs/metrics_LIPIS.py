
import numpy as np
import cv2

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
from tensorflow.python.ops import math_ops

import argparse


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


import cv2
import lpips
import torch

def load_image(path):
    return cv2.imread(path)[:,:,::-1]


def im2tensor(image, imtype=np.uint8, cent=1., factor=255./2.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))


# loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
loss_fn_vgg = lpips.LPIPS(net='vgg')
loss_fn_vgg = loss_fn_vgg.cuda()  # best

files = os.listdir(outimg_dir)
result = 0

for file in files:
    # Load images
    img0 = im2tensor(load_image(os.path.join(outimg_dir,file))) # RGB image from [-1,1]
    img1 = im2tensor(load_image(os.path.join(tgtimg_dir,file)))

    img0 = img0.cuda()
    img1 = img1.cuda()

    # Compute distance
    dist01 = loss_fn_vgg.forward(img0,img1)

    result += dist01.cpu().detach().numpy()
    # print('%s: %.3f'%(file,dist01))
    # f.writelines('%s: %.6f\n'%(file,dist01))
print('=====================================')
print(result/len(files))
print('=====================================')

