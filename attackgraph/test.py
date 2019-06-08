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
class Value:
    def __init__(self, v=None):
        self.v = v

a = {}
a[1] = Value(0)
a[2] = a[1]
a['2'] = 3

print(a)


