import os
from attackgraph import file_op as fp

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

from attackgraph import evaluation_combined as ec


path_def = os.getcwd() + '/drawing/matrix/' + 'payoff_matrix_def.pkl'
path_att = os.getcwd() + '/drawing/matrix/' + 'payoff_matrix_att.pkl'
payoff_matrix_def = fp.load_pkl(path_def)
payoff_matrix_att = fp.load_pkl(path_att)



child_patition = {'RS': 80, 'SP': 50}
ec.do_evaluation(payoff_matrix_def, payoff_matrix_att, child_patition)