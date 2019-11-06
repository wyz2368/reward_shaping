import numpy as np
from attackgraph import file_op as fp
# from attackgraph.uniform_str_init import act_def, act_att
# import math
import os
import copy
from psutil import virtual_memory


# # import networkx as nx
# # import random
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


a=np.array([1,2,3,4,5])
for i, item in enumerate(a):
    if item < 3:
        a[i] = 0

print(a)

