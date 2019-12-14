from simulator import *
from utils import *
# from evaluation import *
from attackgraph import file_op as fp
import os

# Test screen
methods_list = ['RS','BR_selfplay', 'BR_fic']
# paths = create_paths(methods_list)
# print(paths)
# str_dict_def, str_dict_att = screen(paths)
# print(str_dict_def)

payoff_matrix_att, payoff_matrix_def, child_partition = scan_and_sim(methods_list)

print("att_matrix:", '\n', payoff_matrix_att)
print("def_matrix:", '\n', payoff_matrix_def)

