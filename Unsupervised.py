from pynput.keyboard import Controller
import tensorflow as tf
import numpy as np
import LoseScreen
import random
import Input
import time
import math
import cv2
import os


def initialize_uninitialized(sess):
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=.5)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


with tf.variable_scope('Unsupervised'):
    x_in = tf.placeholder(tf.float32, shape=[None, 10, 40, 1])
    t = tf.placeholder(tf.float32, shape=[None, 1])
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
    h_conv3_flat_w_time = tf.concat([h_conv3_flat, t], 1)

    W_fc1 = weight_variable([(10 * 40 * 64) + 1, 256])
    b_fc1 = bias_variable([256])
    h_fc1 = tf.nn.sigmoid(tf.matmul(h_conv3_flat_w_time, W_fc1) + b_fc1)

    W_fc2 = weight_variable([256, 500])
    b_fc2 = bias_variable([500])
    h_fc2 = tf.nn.sigmoid(tf.matmul(h_fc1, W_fc2) + b_fc2)

    W_fc3 = weight_variable([500, 1])
    b_fc3 = bias_variable([1])
    y_out = tf.nn.sigmoid(tf.matmul(h_fc2, W_fc3) + b_fc3)

cross_entropy = tf.losses.mean_squared_error(exp_y, y_out)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.round(y_out), exp_y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
output = tf.round(y_out)


def get_out(sess, im, tim):
    im = np.reshape(im, (1, 10, 40, 1))
    tim = np.reshape(np.array(tim), [1, 1])
    return sess.run(output, feed_dict={x_in: im, t: tim})[0][0]

def getScrn(sess):
    s = time.time()
    os.system("screencapture -R60,125,600,150 holder.png")
    im = cv2.imread('holder.png', 0)
    im = cv2.resize(im, (40, 10), interpolation=cv2.INTER_AREA)
    im = im / 255.0
    running = LoseScreen.check(im, sess)
    return im, running

def act(sess, im, ctime, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 1)
    else:
        return get_out(sess, im, ctime)

Lsaver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='L'))
saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Unsupervised'))
keyboard = Controller()
sess = tf.Session()

LoseScreen.restore(sess, Lsaver)
initialize_uninitialized(sess)

MEMORY_CAPACITY = 10

GAMMA = 0.8

MAX_EPSILON = 1
MIN_EPSILON = 0.1
LAMBDA = 0.005
memory = {}
lens = []
steps = 0

while(1):
    running = True
    epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * steps)
    cgame = []
    steps += 1

    Input.Reload(keyboard)

    start = time.time()
    while(running):
        im, running = getScrn(sess)
        ctime = time.time() - start
        a = act(sess, im, ctime, epsilon)
        Input.Act(keyboard, a)
        cgame.append([im, ctime, a])

    for i in range(5):
        try:
            del cgame[-1]
        except:
            pass

    score = len(cgame)

    same = False
    for i in lens:
        if len(cgame) == i:
            same = True

    if not same:
        memory[len(cgame)] = cgame
        lens.append(len(cgame))
        lens.sort()

    print "Try:", steps, "   Score:", score, "   Highest Score:", lens[-1], "   Epsilon:", epsilon, "    ", lens

    while len(lens) > MEMORY_CAPACITY:
        del memory[lens[0]]
        del lens[0]

    if steps > 10:
        ims = []
        times = []
        actions = []
        for i in lens:
            for j in range(i):
                ims = np.append(ims, memory[i][j][0])
                times.append(memory[i][j][1])
                actions.append(memory[i][j][2])

        ims = np.reshape(ims, (-1, 10, 40, 1))
        times = np.reshape(times, (-1, 1))
        actions = np.reshape(actions, (-1, 1))

        for i in range(30):
            ce, ts = sess.run([cross_entropy, train_step], feed_dict={x_in: ims, t: times, exp_y: actions})
