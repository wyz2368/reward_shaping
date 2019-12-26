import numpy as np
from attackgraph import file_op as fp
# from attackgraph.uniform_str_init import act_def, act_att
# import math
import os
import copy
import glob
from psutil import virtual_memory
from queue import PriorityQueue as pq
# from attackgraph.cournot import extract_submatrix

# # import networkx as nx
import random
# # import itertools
import time
import datetime
# import training
# # import tensorflow as tf
# import pickle as pk
#

a = pq()
a.put(('a',2))
a.put(('b',1))

print(a.get()[0])
