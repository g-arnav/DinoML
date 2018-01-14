import tensorflow as tf
import numpy as np
from os import listdir
import cv2
import time
import os
import Input
from pynput.keyboard import Controller


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=.5)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


with tf.variable_scope('Supervised'):
    x_in = tf.placeholder(tf.float32, shape=[None, 10, 40, 1])
    exp_y = tf.placeholder(tf.float32, shape=[None, 1])

    W_conv1 = weight_variable([2, 2, 1, 16])
    b_conv1 = bias_variable([16])
    h_conv1 = tf.nn.relu(conv2d(x_in, W_conv1) + b_conv1)

    W_conv2 = weight_variable([2, 2, 16, 32])
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

    W_conv3 = weight_variable([2, 2, 32, 64])
    b_conv3 = bias_variable([64])
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)

    h_conv3_flat = tf.reshape(h_conv3, [-1, 10 * 40 * 64])

    W_fc1 = weight_variable([10 * 40 * 64, 256])
    b_fc1 = bias_variable([256])
    h_fc1 = tf.nn.sigmoid(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    W_fc2 = weight_variable([256, 500])
    b_fc2 = bias_variable([500])
    h_fc2 = tf.nn.sigmoid(tf.matmul(h_fc1, W_fc2) + b_fc2)

    W_fc3 = weight_variable([500, 1])
    b_fc3 = bias_variable([1])
    y_out = tf.nn.sigmoid(tf.matmul(h_fc2, W_fc3) + b_fc3)

    cross_entropy = tf.losses.mean_squared_error(exp_y, y_out)
    train_step = tf.train.AdamOptimizer(5e-5).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.round(y_out), exp_y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    output = tf.round(y_out)


saver = tf.train.Saver()
keyboard = Controller()

def train():
    with tf.Session() as sess:
        print 'Gathering Data...'
        data = []
        labels = []
        s = time.time()
        for n in range(1, 11):
            for d in range(0, len(listdir('GamePics/game{}/'.format(n)))):
                im = cv2.imread('GamePics/game{}/'.format(n) + str(d) + '.png', 0)
                im = cv2.resize(im, (40, 10), interpolation=cv2.INTER_AREA)
                im = im / 255.0
                data.append(im)

            nl = np.load('labels/game{}.npy'.format(n))
            labels = np.append(labels, nl)
            print 'labels/game{}.npy'.format(n), nl.shape
        print time.time() - s

        data = np.array(data).reshape(-1, 10, 40, 1)
        labels = labels.reshape(-1, 1)

        print labels.shape, data.shape

        try:
            saver.restore(sess, "models/supervised/model.ckpt")
        except:
            print 'did not restore'
            sess.run(tf.global_variables_initializer())

        s = time.time()
        ce = 1.0
        i = 0
        try:
            while(1):
                if i % 25 == 0:
                    ce, acc, ts = sess.run([cross_entropy, accuracy, train_step], feed_dict={x_in: data, exp_y: labels})
                    print 'step {} - cross entropy: {}    accuracy: {}    time: {}'.format(i, ce, acc, time.time() - s)
                    s = time.time()
                # if i % 10 == 0:
                #     print sess.run(output, feed_dict={x_in: data, exp_y: labels, m: len(data)})
                else:
                    sess.run(train_step, feed_dict={x_in: data, exp_y: labels})
                i += 1
        finally:
            saver.save(sess, "models/supervised/model.ckpt")
            print time.time() - s

def getScrn():
    os.system("screencapture -R60,125,600,150 holder.png")
    im = cv2.imread('holder.png', 0)
    im = cv2.resize(im, (40, 10), interpolation=cv2.INTER_AREA)
    im = im / 255.0
    return im

def run():
    time.sleep(1)
    with tf.Session() as sess:
        saver.restore(sess, "models/supervised/model.ckpt")
        print 'Starting...'
        Input.Reset(keyboard)
        while 1:
            im = np.reshape(getScrn(), [1, 10, 40, 1])
            a = sess.run(output, feed_dict={x_in : im})
            if a == [[0]]:
                Input.Wait(keyboard)
            else:
                Input.Up(keyboard)
            print a
