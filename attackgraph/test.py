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


# a = {}
# a[1]=1
# a[2]=2
# a[3]=3
# print(1 in a.keys())


a = np.array([1,2,3,4,5])
b = np.reshape(a,newshape=[5,1])
a = np.array([])
print(a)

# epoch = 10
# gamma = 0.7
# mix_str_def = np.zeros(epoch)
# mix_str_att = np.zeros(epoch)
# for i in np.arange(1, epoch+1):
#     temp = np.random.dirichlet(np.ones(i),size=1)[0]
#     mix_str_def[:len(temp)] += temp * gamma**(epoch-i)
#     print(i, gamma**(epoch-i))
#     temp = np.random.dirichlet(np.ones(i),size=1)[0]
#     mix_str_att[:len(temp)] += temp * gamma**(epoch-i)
# mix_str_def = mix_str_def / np.sum(mix_str_def)
# mix_str_att = mix_str_att / np.sum(mix_str_att)
# print(mix_str_def)
# print(mix_str_att)
# print(np.sum(mix_str_def))
# print(np.sum(mix_str_att))

# epoch = 10
# mem_size = 3
# for i in np.arange(epoch-mem_size+1, epoch + 1):
#     print(i)
