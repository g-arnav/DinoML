import tensorflow as tf
import numpy as np
from time import time

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


with tf.variable_scope('P'):
    x_in = tf.placeholder(tf.float32, shape=[None, 10, 40, 1])
    exp_out = tf.placeholder(tf.float32, shape=[None, 1])

    W_conv1 = weight_variable([3, 3, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_in, W_conv1) + b_conv1)

    h_conv1_flat = tf.reshape(h_conv1, [-1, 10 * 40 * 32])


    W_fc1_0 = weight_variable([10 * 40 * 32, 128])
    h_fc1_0 = tf.nn.relu(tf.matmul(h_conv1_flat, W_fc1_0))

    W_fc2_0 = weight_variable([128, 1])
    y_out_0 = tf.matmul(h_fc1_0, W_fc2_0)

    cross_entropy_0 = tf.losses.mean_squared_error(exp_out, y_out_0)
    train_step_0 = tf.train.AdamOptimizer(1e-6).minimize(cross_entropy_0)


    W_fc1_1 = weight_variable([10 * 40 * 32, 128])
    h_fc1_1 = tf.nn.relu(tf.matmul(h_conv1_flat, W_fc1_1))

    W_fc2_1 = weight_variable([128, 1])
    y_out_1 = tf.matmul(h_fc1_1, W_fc2_1)

    cross_entropy_1 = tf.losses.mean_squared_error(exp_out, y_out_1)
    train_step_1 = tf.train.AdamOptimizer(1e-6).minimize(cross_entropy_1)


    W_fc1_2 = weight_variable([10 * 40 * 32, 128])
    h_fc1_2 = tf.nn.relu(tf.matmul(h_conv1_flat, W_fc1_2))

    W_fc2_2 = weight_variable([128, 1])
    y_out_2 = tf.matmul(h_fc1_2, W_fc2_2)

    cross_entropy_2 = tf.losses.mean_squared_error(exp_out, y_out_2)
    train_step_2 = tf.train.AdamOptimizer(1e-6).minimize(cross_entropy_2)

    output = tf.reshape(tf.concat([y_out_0, y_out_1, y_out_2], 0), (1, 3))


def train(sess, x, y, a):
    x = np.reshape(x, (-1, 10, 40, 1))
    y = np.reshape(y, (-1, 1))
    if a == 0:
        sess.run(train_step_0, feed_dict={x_in: x, exp_out: y})
    elif a == 1:
        sess.run(train_step_1, feed_dict={x_in: x, exp_out: y})
    elif a == 2:
        sess.run(train_step_2, feed_dict={x_in: x, exp_out: y})

def predict(sess, s):
    s = np.reshape(s, (-1, 10, 40, 1))
    return sess.run(output, feed_dict={x_in: s})

