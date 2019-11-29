import time
import tensorflow as tf
import numpy as np
import os
import cv2
import random
import math
import LoseScreen
import Input
import Pnet
from pynput.keyboard import Controller


MEMORY_CAPACITY = 512
BATCH_SIZE = 64

GAMMA = 0.8

MAX_EPSILON = 1
MIN_EPSILON = 0.1
LAMBDA = 0.000005  # speed of decay

keyboard = Controller()

t = time.time()

def getScrn():
    os.system("screencapture -R60,125,600,150 holder.png")
    im = cv2.imread('holder.png', 0)
    im = cv2.resize(im, (40, 10), interpolation=cv2.INTER_AREA)
    im = np.round(im / 400.0)
    running = LoseScreen.check(im, sess)
    im = list(im.flatten())
    return im, running

class Brain:
    def train(self, x, y, a, epoch):
        for w in range(epoch):
            Pnet.train(sess, x, y, a)

    def predict(self, s):
        return Pnet.predict(sess, s)

class Memory:
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, states, actions, rewards):
        self.samples.append((states, actions, rewards))

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)


class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self):
        self.brain = Brain()
        self.memory = Memory(MEMORY_CAPACITY)

    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        else:
            a = np.argmax(self.brain.predict(s))
            print 'predicted', self.brain.predict(s), a
            return a

    def step(self):
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def observe(self, states, actions, rewards):
        self.memory.add(states, actions, rewards)

    def replay(self):
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)

        for i in range(batchLen):
            o = batch[i]
            s = o[0]
            a = o[1]
            r = o[2]
            self.brain.train(s, r, a, 1)

class Environment:
    def __init__(self, i):
        self.i = i

    def run(self, agent):
        Input.Reset(keyboard)
        time.sleep(3)
        start = time.time()
        states = []
        actions = []
        s, running = getScrn()

        while True:
            a = agent.act(s)

            actions.append(a)
            states.append([s])

            Input.Act(keyboard, a)

            s_, running = getScrn()

            if not running:
                s_ = list(np.zeros((10, 40)).flatten())

            agent.step()

            s = s_

            if not running:
                break

        R = (time.time() - start - 1.75) * 100
        print 'Reward On Try {}:'.format(self.i), R

        r = R / len(states)
        for j in range(len(states)):
            agent.observe(states[j], actions[j], r * (((len(states) - j)**(1/8)) / (len(states)**(1/8))))
        agent.replay()
        time.sleep(0.5)


sess = tf.Session()
saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())
LoseScreen.run(sess)
c = True
Input.Reset(keyboard)
while c:
    im, c = getScrn()
    t = time.time()
    print c, time.time() - t


agent = Agent()

time.sleep(1)
try:
    i = 1
    while True:
        env = Environment(i)
        env.run(agent)
        i += 1
finally:
    saver.save(sess, "models/model.ckpt")
    sess.close()


