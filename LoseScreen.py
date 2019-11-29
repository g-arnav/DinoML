import tensorflow as tf
from time import time
from os import listdir
import cv2
import numpy as np
import os



def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=.5)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


with tf.variable_scope('L'):
    x_in = tf.placeholder(tf.float32, shape=[None, 10, 40, 1])
    exp_y = tf.placeholder(tf.float32, shape=[None, 3])


    x_in_flat = tf.reshape(x_in, (-1, 10*40))

    W_fc1 = weight_variable([10 * 40, 64])
    b_fc1 = bias_variable([64])
    h_fc1 = tf.nn.sigmoid(tf.matmul(x_in_flat, W_fc1) + b_fc1)

    W_fc2 = weight_variable([64, 3])
    b_fc2 = bias_variable([3])
    y_out = tf.nn.sigmoid(tf.matmul(h_fc1, W_fc2) + b_fc2)

    cross_entropy = tf.losses.mean_squared_error(exp_y, y_out)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.round(y_out), exp_y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    output = tf.argmax(y_out, 1)


def check(im, sess):
    im = np.array(np.reshape(im, (1, 10, 40, 1)), dtype=float)
    return sess.run(output, feed_dict={x_in: im})


def train(sess, saver):
    data = []
    labels = []
    for d in listdir('GamePics/'):
        print('GamePics/' + d)
        im = cv2.imread('GamePics/' + d, 0)
        im = cv2.resize(im, (40, 10), interpolation=cv2.INTER_AREA)
        im = im / 255.0
        data.append(im)
        labels.append([1, 0, 0])
    n = len(labels)

    for d in listdir('LossPics'):
        print('LossPics/' + d)
        im = cv2.imread('LossPics/' + d, 0)
        im = cv2.resize(im, (40, 10), interpolation=cv2.INTER_AREA)
        im = im / 255.0
        data.append(im)
        labels.append([0, 1, 0])

    for d in listdir('LossPics'):
        print('LossPics/' + d)
        im = cv2.imread('LossPics/' + d, 0)
        im = cv2.resize(im, (40, 10), interpolation=cv2.INTER_AREA)
        im = im / 255.0
        data.append(im)
        labels.append([0, 1, 0])

    for n in range(750):
        d = listdir('NotRunning')[0]
        print('NotRunning/' + str(n))
        im = cv2.imread('NotRunning/' + d, 0)
        im = cv2.resize(im, (40, 10), interpolation=cv2.INTER_AREA)
        im = im / 255.0
        data.append(im)
        labels.append([0, 0, 1])
    p = len(labels) - n

    print n, p

    # DataAndLabels = np.concatenate((data, labels), axis = 1)
    # np.random.shuffle(DataAndLabels)
    # data, labels = np.hsplit(DataAndLabels, 2)
    # print data.shape
    # print labels.shape

    data = np.reshape(data, (len(data), 10, 40, 1))
    labels = np.reshape(labels, [len(labels), 3])
    print labels.shape

    s = time()
    i = 0
    ce = 1.0
    try:
        while(1):
            i += 1
            if i % 1 == 0:
                ce, train_accuracy = sess.run([ cross_entropy, accuracy], feed_dict={x_in: data, exp_y: labels})
                print('step {}- cross entropy: {}   accuracy: {}'.format(i, ce, train_accuracy))
            # if i % 10 == 0:
            #     print sess.run(output, feed_dict={x_in: data, exp_y: labels, m: len(data)})
            sess.run(train_step, feed_dict={x_in: data, exp_y: labels})
            y = sess.run(y_out, feed_dict={x_in: data, exp_y: labels})

    except KeyboardInterrupt:
        print('test accuracy %g' % sess.run(accuracy, feed_dict={x_in: data, exp_y: labels}))
        saver.save(sess, "models/unsupervised/loseScreen.ckpt")
        print time() - s

def test(sess):
    c = True
    while 1:
        os.system("screencapture -R60,125,600,150 holder.png")
        im = cv2.imread('holder.png', 0)
        im = cv2.resize(im, (40, 10), interpolation=cv2.INTER_AREA)
        im = im / 255.0
        s = time()
        c = check(im, sess)
        print c, time() - s

def restore(sess, saver):
    saver.restore(sess, "models/unsupervised/loseScreen.ckpt")

