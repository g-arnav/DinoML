from getScrn import getScrn
from Input import Up, Wait, Reload, Reset
import numpy as np
from random import *
import time
import math
import LoseScreen
import tensorflow as tf
from pynput.keyboard import Controller
import copy
import pickle

def save_obj(obj, name):
    with open('models/NEAT/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('models/NEAT/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

pop_size = 20
new_neuron_chance = 0.05
new_connection_chance = 0.005
del_connection_chance = 0.05
mutate_weight_chance = 0.4
input_nodes = 10 * 40
output_nodes = 1
max_score = 55
keyboard = Controller()

def sigmoid(x):
    return 1 / (1 + (math.e ** -x))

def inp_frame(sess):
    im = getScrn()
    running = LoseScreen.check(im, sess)
    im = list(im.reshape(-1))
    for i in range(len(im)):
        if im[i] > .85:
            im[i] = 0.
        else:
            im[i] = 1.
    return im, running

class Member:
    def __init__(self, id, neurons=None):
        self.id = id
        if neurons != None:
            self.neurons = neurons
        else:
            self.neurons = {}

            for in_node in range(input_nodes):
                self.neurons[in_node] = [0., []]
                if random() < new_connection_chance:
                    self.neurons[in_node][1] = [self.app_connection()]

    def app_connection(self, output_node=-1):
        return [output_node, 2 * random() - 1]

    def mutate_weight(self, neuron, connection):
        self.neurons[neuron][1][connection][1] = self.neurons[neuron][1][connection][1] + np.random.normal(0, 0.25)


    def mutate(self, id):
        for n in list(self.neurons.iterkeys()):

            cnctd = []
            for i in range(len(self.neurons[n][1])):
                if random() < mutate_weight_chance:
                    self.mutate_weight(n, i)
                if random() < new_neuron_chance:
                    new_neuron = max(self.neurons) + 1
                    if self.neurons[n][1][i][0] == -1:
                        layer = (self.neurons[n][0] + 1) / 2
                    else:
                        layer = (self.neurons[self.neurons[n][1][i][0]][0] + self.neurons[n][0]) / 2
                    self.neurons[new_neuron] = [layer, [[self.neurons[n][1][i][0], self.neurons[n][1][i][1]]]]
                    self.neurons[n][1][i][0] = new_neuron
                    self.mutate_weight(n, i)
                    self.mutate_weight(new_neuron, 0)
                cnctd.append(self.neurons[n][1][i][0])

                if random() < del_connection_chance:
                    del(self.neurons[n][1][i])
                    del(cnctd[-1])

            layer = self.neurons[n][0]
            pos_out = [-1]
            for out in self.neurons:
                if self.neurons[out][0] > layer:
                    pos_out.append(out)
            for pos_cnct in list(np.setdiff1d(pos_out, cnctd)):
                if random() < new_connection_chance:
                    self.neurons[n][1].append(self.app_connection(pos_cnct))
        self.id = int(str(self.id) + str(id))

    def generate_graph(self):
        pass

    def run(self, inp):
        graph = {-1: 0}
        q = 0
        for n in range(len(inp)):
            graph[n] = inp[n]
        for n in range(len(inp), len(self.neurons)):
            graph[n] = 0
        for nrn in range(len(self.neurons)):
            if nrn < input_nodes + 1:
                for cnctn in self.neurons[nrn][1]:
                    q += 1
                    graph[cnctn[0]] += graph[nrn] * cnctn[1]
            else:
                graph[nrn] = sigmoid(graph[nrn])
                for cnctn in self.neurons[nrn][1]:
                    q += 1
                    graph[cnctn[0]] += graph[nrn] + cnctn[1]
        return sigmoid(graph[-1])


    def eval_fitness(self, sess):
        Reload(keyboard)
        time.sleep(.5)
        inp, state = inp_frame(sess)
        while state == 2:
            Reset(keyboard)
            time.sleep(1)
            inp, state = inp_frame(sess)
        time.sleep(3)
        fitness = 0
        start = time.time()
        while state != 1:
            inp, state = inp_frame(sess)
            out = self.run(inp)
            if out > 0.5:
                Up(keyboard)
            else:
                Wait(keyboard)
            while (time.time() - start) % 0.1 > 0.01:
                pass
            fitness += 1
        return fitness

    def new_member(self):
        return self.id, self.neurons

def evolve(pop, fit, orig_members):
    print 'Evolving'
    fit = sorted(fit.iteritems(), key=lambda (k, v): (v, k))
    fit.reverse()
    print fit
    new_pop = []
    for i in range(15):
        new_pop.append(copy.deepcopy(pop[fit[i][0]]))
        for j in range(4 - i/3):
            new_pop.append(copy.deepcopy(pop[fit[i][0]]))
            new_pop[-1].mutate(j + 1)
    for i in range(1, 6):
        new_pop.append(Member((orig_members + i) * 10))
    orig_members += 5
    return new_pop, orig_members


def initialize_population():
    population = []
    for i in range(pop_size):
        population.append(Member((i + 1) * 10))
    return population

sess = tf.Session()
saver = tf.train.Saver()
LoseScreen.restore(sess, saver)

try:
    population, i = load_obj("2")
    print i
except:
    population = initialize_population()
    i = 0
orig_members = 50

try:
    while(1):
        fitness = {}
        for n in range(len(population)):
            fitness[n] = population[n].eval_fitness(sess)
            print "{0:1d} {1:2d} {2:3d} {3:4d}".format(population[n].id, i + 1, n + 1, fitness[n])
        print "Generation:", i, "   Max:", max(fitness.itervalues()), "   Average:", sum(fitness.itervalues()) / pop_size
        population, orig_members = evolve(population, fitness, orig_members)
        i += 1
except KeyboardInterrupt:
    print i
    save_obj([population, i], "2")


# LoseScreen.test(sess)