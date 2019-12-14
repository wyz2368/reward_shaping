import numpy as np
from attackgraph import file_op as fp
# from attackgraph.uniform_str_init import act_def, act_att
# import math
import os
import copy
import glob
from psutil import virtual_memory
from queue import PriorityQueue as pq

# # import networkx as nx
import random
# # import itertools
import time
import datetime
# import training
# # import tensorflow as tf
# import pickle as pk
#
# class Value:
#     def __init__(self, v=None):
#         self.v = v
#
# a = {}
# a[1] = Value(0)
# a[2] = a[1]
# a['2'] = 3
#
# print(a)


# b = np.zeros(6,)
# a = []
# a.append(np.array([0.3, 0.7, 0  , 0  , 0  , 0]))
# a.append(np.array([0.1, 0.3, 0.6, 0  , 0  , 0]))
# a.append(np.array([0  , 0.1, 0.5, 0.4, 0  , 0]))
# a.append(np.array([0  , 0.1, 0.3, 0  , 0.6, 0]))
# a.append(np.array([0  , 0.1, 0.1, 0.1, 0.4, 0.3]))
#
# gamma = 0.3
# epoch = 5
# den = 0
# for i in a:
#     b += i * gamma**epoch
#     den += gamma**epoch
#     epoch -=1
#     print(epoch, ":", i * gamma**epoch, gamma**epoch)
#
# print(b/np.sum(b))
# print(b/den)

# a = pq()
# a.put((0,"a"))
# a.put((1,"b"))
# def ji(a):
#     a.put((2, "c"))
# ji(a)
# print(a.get())
# print(a.get())
# print(a.get())

# a = np.array([[1,2,3],[4,5,6],[7,8,9]])
# print(a[0:2,0:2])

# a = glob.glob1(os.getcwd()+'/gambit_data/',"*")
# print(type(a))

a = {0:np.array([0.3,0.2,0]),1:np.array([0.3,0.2,0.5])}
a[0][a[0]>0] = 1
print(a)