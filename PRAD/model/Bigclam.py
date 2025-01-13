import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from PRAD.tools.agm import AGM, community
import numpy as np
import random

def bigclam(graph, community_num, step_size, threshold):  # max_iter
    # initialize a random matrix F
    F = np.matrix(np.ones((len(graph.nodes()), community_num)))
    print(np.shape(F))
    for i in range(np.shape(F)[0]):
        for j in range(np.shape(F)[1]):
            F[i, j] = random.random()

    # MLE
    iter = 0
    last_norm = 10000000
    while True:
        # print('start', last_norm)
        # we cannot use last_F = F cause this is an assignment and will pass the reference of the class F, not copy
        last_F = F.copy()
        iter += 1
        for node in graph.nodes():
            gradient = np.matrix(np.zeros((1, community_num)))
            u = F[node, :]
            for neighbor in graph.neighbors(node):
                v = F[neighbor, :]
                gradient += v * float(np.exp(-u * v.transpose()) / (1 - np.exp(-u * v.transpose()))) + v
            gradient += u - np.sum(F, axis=0)
            # update row u
            F[node, :] = F[node, :] + 0.1 / iter * gradient
            # check for non-negative
            for i in range(community_num):
                if F[node, i] <= 0:
                    F[node, i] = 0.001
        f_norm = np.sum(np.multiply(F - last_F, F - last_F)) ** 0.5
        # print( type(f_norm))
        # if f_norm > last_norm:
        # 	# print('in > ', last_norm)
        # 	# print 'iter is {0} Frpbemois norm is {1:.5} last norm is {2:5} step_size is {3}'.format(iter, float(f_norm), last_norm,step_size)
        # 	F = last_F.copy()
        # 	#print f_norm
        # 	step_size = step_size*0.1
        # 	continue

        # print('end', last_norm)
        print('iter is {0} Frpbemois norm is {1:.5} last norm is {2:5} step_size is {3}'.format(iter, float(f_norm),
                                                                                                last_norm, step_size))
        last_norm = f_norm

        if f_norm <= threshold:
            break
    print(iter)
    return F

def Reshape(F):
    for i in range(np.shape(F)[0]):
        for j in range(np.shape(F)[1]):
            if F[i, j] < 0.2:
                F[i, j] = 0
            if F[i, j] >= 0.8:
                F[i, j] = 0.9999
    return F