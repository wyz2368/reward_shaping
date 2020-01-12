from attackgraph.NE_search_et import eligibility_trace, create_ne_dict, ne_search_with_etrace, ne_search_wo_etrace
import numpy as np
from attackgraph import file_op as fp
import os

method_list = ["DO", "DO_SP"]
child_partition = {"DO": 80, "DO_SP":80}
ne_dict = create_ne_dict(method_list)

# for method in ne_dict:
#     print(method_list)
#     for epoch in ne_dict[method]:
#         print(epoch)

pq_def, pq_att = eligibility_trace(ne_dict, child_partition)


# for i in np.arange(10):
#     print(pq_def["DO"].get())
#
# print('-------------------------')
#
# for i in np.arange(10):
#     print(pq_def["DO_SP"].get())
#
#
# print('==========================')
#
#
# for i in np.arange(10):
#     print(pq_att["DO"].get())
#
# print('-------------------------')
#
# for i in np.arange(10):
#     print(pq_att["DO_SP"].get())

payoff_path = os.getcwd() + '/combined_game/'
payoff_matrix_def = fp.load_pkl(payoff_path + 'payoff_matrix_def.pkl')
payoff_matrix_att = fp.load_pkl(payoff_path + 'payoff_matrix_att.pkl')

nash_def, nash_att, indicator_matrix = ne_search_wo_etrace(payoff_matrix_def, payoff_matrix_att, child_partition)

print(indicator_matrix)