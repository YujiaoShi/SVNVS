import tensorflow as tf
import numpy as np


def tf_shape(x, rank):
    static_shape = x.get_shape().with_rank(rank).as_list()
    dynamic_shape = tf.unstack(tf.shape(x), rank)
    return [s if s is not None else d for s,d in zip(static_shape, dynamic_shape)]


def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2



def softargmax(x):
    x = x - tf.reduce_max(x, axis=-1, keepdims=True)
    x = x - tf.math.reduce_logsumexp(x, axis=-1, keepdims=True)
    x = tf.exp(x)
    x_range = tf.range(x.shape.as_list()[-1], dtype=tf.float32)
    return tf.reduce_sum(x * x_range, axis=-1, keepdims=True)


# def softargmax(x, beta=1):
#     x_range = tf.range(x.shape.as_list()[-1], dtype=tf.float32)
#     return tf.reduce_sum(tf.nn.softmax(x * beta) * x_range, axis=-1, keepdims=True)


# def softargmax(x):
#     '''
#     :param x: [B, H, W, C]
#     :return:
#     '''
#     x_range = tf.range(x.shape.as_list()[-1], dtype=tf.float32)
#     exp_x = tf.exp(x)
#     exp_x_sum = tf.reduce_sum(exp_x, axis=-1, keepdims=True)
#     exp_x_sum_zero_add = tf.cast(tf.equal(exp_x_sum, 0.0), tf.float32) * 1e-12
#     exp_x_sum = exp_x_sum + exp_x_sum_zero_add
#     softmax_x = exp_x / exp_x_sum
#     return tf.reduce_sum(softmax_x * x_range, axis=-1, keepdims=True)

def stable_softmax(x, axis=-1):
    x = x - tf.reduce_max(x, axis, keepdims=True)
    x = x - tf.math.reduce_logsumexp(x, axis, keepdims=True)
    x = tf.exp(x)
    return x

def stable_softmax1(x, axis=-1):
    minx = tf.reduce_min(x, axis, keepdims=True)
    maxx = tf.reduce_max(x, axis, keepdims=True)
    x = (x - minx)/(maxx - minx + 1e-6)
    exp_x = tf.exp(x)
    exp_x_sum = tf.reduce_sum(exp_x, axis=axis, keepdims=True) + 1e-6
    return exp_x/exp_x_sum





def Smooth_l1_loss(labels,predictions,scope=tf.GraphKeys.LOSSES):
    with tf.variable_scope(scope):
        diff=tf.abs(labels-predictions)
        less_than_one=tf.cast(tf.less(diff,1.0),tf.float32)   #Bool to float32
        smooth_l1_loss=(less_than_one*0.5*diff**2)+(1.0-less_than_one)*(diff-0.5)
        return tf.reduce_mean(smooth_l1_loss)


def compute_delta_d(depth_map):
    batch, height, width, _ = depth_map.get_shape().as_list()

    xx, yy = tf.meshgrid(tf.range(width, dtype=tf.float32), tf.range(height, dtype=tf.float32), indexing='ij')

    xxx = tf.reshape(xx, [-1])
    yyy = tf.reshape(yy, [-1])

    X = tf.stack([xxx, yyy, tf.ones_like(xxx)], axis=0)

    for batch_index in range(batch):
        D1 = tf.transpose(depth_map[batch_index, :, :, 0])
        D2 = D1 + 1

        X1 = X*D1
        X2 = X*D2

        ray1 = tf.matmul()

