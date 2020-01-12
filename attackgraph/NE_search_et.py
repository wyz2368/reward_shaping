import numpy as np
from attackgraph.utils_combined import find_heuristic_position
from attackgraph.gambit_analysis import do_gambit_analysis
from attackgraph.cournot import extract_submatrix as es
from attackgraph import file_op as fp
from queue import PriorityQueue as pq
import os
import copy

def eligibility_trace(nasheq_dict, child_partition, gamma=0.7):
    '''
    This function calculates the eligibility trace of strategies based on game.nasheq.
    :param nasheq_dict: {"baselines": game.nasheq}
    :param gamma:
    :return:
    '''
    position = find_heuristic_position(child_partition)
    total_num_str = 0
    for method in child_partition:
        total_num_str += child_partition[method]

    et_dict_def = {}
    et_dict_att = {}

    nash_thred = 0.05

    # Construct eligibility trace.
    for method in nasheq_dict:
        et_dict_def[method] = np.zeros(child_partition[method])
        et_dict_att[method] = np.zeros(child_partition[method])
        nasheq = nasheq_dict[method]
        for epoch in np.arange(1, child_partition[method]+1):
            et_dict_att[method] *= gamma
            ne_att = nasheq[epoch][1][1:]
            ne_att[ne_att <= nash_thred] = 0
            ne_att[ne_att > nash_thred] = -2
            et_dict_att[method][:len(ne_att)] += ne_att

            et_dict_def[method] *= gamma
            ne_def = nasheq[epoch][0][1:]
            ne_def[ne_def <= nash_thred] = 0
            ne_def[ne_def > nash_thred] = -2
            et_dict_def[method][:len(ne_def)] += ne_def

    # Put strategies into the queue with the eligibility trace as priority.
    pq_def = {}
    pq_att = {}
    for method in et_dict_def:
        pq_def[method] = pq()
        pq_att[method] = pq()
        start, end = position[method]
        idx_str = start + np.arange(child_partition[method])
        idx_et_pair_def = zip(et_dict_def[method], idx_str)
        idx_et_pair_att = zip(et_dict_att[method], idx_str)
        for pair in idx_et_pair_def:
            pq_def[method].put(pair)

        for pair in idx_et_pair_att:
            pq_att[method].put(pair)

    return pq_def, pq_att


def create_ne_dict(method_list):
    load_path = os.getcwd() + '/combined_game/games/'

    ne_dict = {}
    for method in method_list:
        path = load_path + method + '/game.pkl'
        game = fp.load_pkl(path)
        ne_dict[method] = game.nasheq

    return ne_dict


def ne_search_wo_etrace(payoff_matrix_def, payoff_matrix_att, child_partition):
    position = find_heuristic_position(child_partition)

    total_num_str = 0
    init_flag = False

    # Assume 2 methods. Find candidate NE in the first subgame.
    for method in child_partition:
        if not init_flag:
            nash_att, nash_def = do_gambit_analysis(payoff_matrix_def[:child_partition[method], :child_partition[method]],
                                                    payoff_matrix_att[:child_partition[method], :child_partition[method]],
                                                    maxent=False,
                                                    minent=False)
            # Strategies of current game
            strategy_set_def = list(range(child_partition[method]))
            strategy_set_att = list(range(child_partition[method]))
            init_flag = True

        total_num_str += child_partition[method]


    # Extend the NE to the length of the combined game.
    zeros_def = np.zeros(total_num_str)
    zeros_att = np.zeros(total_num_str)
    zeros_def[:len(nash_def)] = nash_def
    zeros_att[:len(nash_def)] = nash_att
    nash_def = zeros_def
    nash_att = zeros_att

    # indicator_matrix records which cell has been simulated in the payoff matrix.
    indicator_matrix = np.zeros((total_num_str, total_num_str))
    for method in position:
        start, end = position[method]
        indicator_matrix[start:end, start:end] = 1

    nash_def_T = np.reshape(nash_def, newshape=(len(nash_def), 1))


    payoff_def = np.sum(nash_def_T * payoff_matrix_def * nash_att)
    payoff_att = np.sum(nash_def_T * payoff_matrix_att * nash_att)

    support_idx_def = np.where(nash_def>0)[0]
    support_idx_att = np.where(nash_att>0)[0]

    # Change to simulation mode when simulation is needed.
    while True:

        for x in support_idx_def:
            indicator_matrix[x, :] = 1
        for y in support_idx_att:
            indicator_matrix[:, y] = 1

        dev_payoff_def = np.max(np.sum(payoff_matrix_def * nash_att, axis=1))
        dev_payoff_att = np.max(np.sum(nash_def_T * payoff_matrix_att, axis=0))

        dev_def = np.argmax(np.sum(payoff_matrix_def * nash_att, axis=1))
        dev_att = np.argmax(np.sum(nash_def * payoff_matrix_att, axis=0))

        if dev_payoff_def <= payoff_def and dev_payoff_att <= payoff_att:
            break
        if dev_payoff_def > payoff_def:
            strategy_set_def.append(dev_def)
            strategy_set_def.sort()
            indicator_matrix[dev_def, :] = 1
        else:
            strategy_set_def.append(dev_def)
            strategy_set_def.sort()
            indicator_matrix[dev_def, :] = 1

        if dev_payoff_att > payoff_att:
            strategy_set_att.append(dev_att)
            strategy_set_att.sort()
            indicator_matrix[:, dev_att] = 1
        else:
            strategy_set_att.append(dev_att)
            strategy_set_att.sort()
            indicator_matrix[:, dev_att] = 1

        subgame_def = es(strategy_set_def, strategy_set_att, payoff_matrix_def)
        subgame_att = es(strategy_set_def, strategy_set_att, payoff_matrix_att)

        # print(strategy_set_def, strategy_set_att)
        # print(np.shape(subgame_def), np.shape(subgame_att))

        nash_att, nash_def = do_gambit_analysis(subgame_def, subgame_att, maxent=False, minent=False)
        nash_def_T = np.reshape(nash_def, newshape=(len(nash_def), 1))

        payoff_def = np.sum(nash_def_T * subgame_def * nash_att)
        payoff_att = np.sum(nash_def_T * subgame_att * nash_att)

        zeros_def = np.zeros(total_num_str)
        zeros_att = np.zeros(total_num_str)
        for pos, value in zip(strategy_set_att, nash_att):
            zeros_att[pos] = value
        for pos, value in zip(strategy_set_def, nash_def):
            zeros_def[pos] = value

        nash_def = zeros_def
        nash_att = zeros_att

        support_idx_def = np.where(nash_def > 0)[0]
        support_idx_att = np.where(nash_att > 0)[0]


    # Payoff matrix of subgames denotes 5.
    for method in position:
        start, end = position[method]
        indicator_matrix[start:end, start:end] = 5

    return nash_def, nash_att, indicator_matrix


def ne_search_with_etrace(payoff_matrix_def, payoff_matrix_att, child_partition, pq_def, pq_att):
    position = find_heuristic_position(child_partition)

    total_num_str = 0
    init_flag = False

    # Assume 2 methods. Find candidate NE in the first subgame.
    for method in child_partition:
        if not init_flag:
            nash_att, nash_def = do_gambit_analysis(
                payoff_matrix_def[:child_partition[method], : child_partition[method]],
                payoff_matrix_att[:child_partition[method], : child_partition[method]],
                maxent=False,
                minent=False)
            # Strategies of current game
            strategy_set_def = list(range(child_partition[method]))
            strategy_set_att = list(range(child_partition[method]))
            init_flag = True
        second_method = method
        total_num_str += child_partition[method]

    # Extend the NE to the length of the combined game.
    zeros_def = np.zeros(total_num_str)
    zeros_att = np.zeros(total_num_str)
    zeros_def[:len(nash_def)] = nash_def
    zeros_att[:len(nash_def)] = nash_att
    nash_def = zeros_def
    nash_att = zeros_att

    # indicator_matrix records which cell has been simulated in the payoff matrix.
    indicator_matrix = np.zeros((total_num_str, total_num_str))
    for method in position:
        start, end = position[method]
        indicator_matrix[start:end, start:end] = 1

    nash_def_T = np.reshape(nash_def, newshape=(len(nash_def), 1))

    payoff_def = np.sum(nash_def_T * payoff_matrix_def * nash_att)
    payoff_att = np.sum(nash_def_T * payoff_matrix_att * nash_att)

    support_idx_def = np.where(nash_def > 0)[0]
    support_idx_att = np.where(nash_att > 0)[0]

    # Change to simulation mode when simulation is needed.
    confirmed = False
    while True:
        sample_pq_def = copy.copy(pq_def)
        sample_pq_att = copy.copy(pq_att)

        dev_flag = False
        while True:

            if sample_pq_def[second_method].empty() and sample_pq_att[second_method].empty():
                confirmed = True
                break

            while True:
                if sample_pq_def[second_method].empty():
                    break
                _, dev_def = sample_pq_def[second_method].get()
                if dev_def not in strategy_set_def:
                    break

            while True:
                if sample_pq_att[second_method].empty():
                    break
                _, dev_att = sample_pq_att[second_method].get()
                if dev_att not in strategy_set_att:
                    break

            for x in support_idx_def:
                indicator_matrix[x, dev_att] = 1
            for y in support_idx_att:
                indicator_matrix[dev_def, y] = 1

            dev_payoff_def = np.sum(payoff_matrix_def[dev_def,:] * nash_att)
            dev_payoff_att = np.sum(nash_def_T * payoff_matrix_att[:,dev_att])

            if dev_payoff_def > payoff_def:
                strategy_set_def.append(dev_def)
                strategy_set_def.sort()
                indicator_matrix[dev_def, :] = 1
                dev_flag = True
            if dev_payoff_att > payoff_att:
                strategy_set_att.append(dev_att)
                strategy_set_att.sort()
                indicator_matrix[:, dev_att] = 1
                dev_flag = True

            if dev_flag:
                break

        if confirmed:
            break

        subgame_def = es(strategy_set_def, strategy_set_att, payoff_matrix_def)
        subgame_att = es(strategy_set_def, strategy_set_att, payoff_matrix_att)

        nash_att, nash_def = do_gambit_analysis(subgame_def, subgame_att, maxent=False, minent=False)
        nash_def_T = np.reshape(nash_def, newshape=(len(nash_def), 1))

        payoff_def = np.sum(nash_def_T * subgame_def * nash_att)
        payoff_att = np.sum(nash_def_T * subgame_att * nash_att)

        zeros_def = np.zeros(total_num_str)
        zeros_att = np.zeros(total_num_str)
        for pos, value in zip(strategy_set_att, nash_att):
            zeros_att[pos] = value
        for pos, value in zip(strategy_set_def, nash_def):
            zeros_def[pos] = value

        nash_def = zeros_def
        nash_att = zeros_att

        support_idx_def = np.where(nash_def > 0)[0]
        support_idx_att = np.where(nash_att > 0)[0]

    # Payoff matrix of subgames denotes 5.
    for method in position:
        start, end = position[method]
        indicator_matrix[start:end, start:end] = 5

    return nash_def, nash_att, indicator_matrix



# if __name__ == '__main__':
#





















