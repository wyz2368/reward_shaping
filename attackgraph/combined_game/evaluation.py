import numpy as np


def regret(nash_att, nash_def, payoffmatrix_att, payoffmatrix_def):
    """
    Calculate the regret vectors for both players.
    :param nash_att: attacker's NE
    :param nash_def: defender's NE
    :param payoffmatrix_att: attacker's payoff matrix
    :param payoffmatrix_def: defender's payoff matrix
    :return: regret vectors
    """
    num_str = len(nash_att)
    x1, y1 = np.shape(payoffmatrix_def)
    x2, y2 = np.shape(payoffmatrix_att)
    if x1 != y1 or x1 != x2 or x2 != y2 or x1 != num_str:
        raise ValueError("Dim of NE does not match payoff matrix.")

    nash_def = np.reshape(nash_def, newshape=(num_str, 1))

    dPayoff = np.round(np.sum(nash_def * payoffmatrix_def * nash_att), decimals=2)
    aPayoff = np.round(np.sum(nash_def * payoffmatrix_att * nash_att), decimals=2)

    utils_def = np.round(np.sum(payoffmatrix_def * nash_att, axis=1), decimals=2)
    utils_att = np.round(np.sum(nash_def * payoffmatrix_att, axis=0), decimals=2)

    regret_def = utils_def - dPayoff
    regret_att = utils_att - aPayoff

    regret_def = np.reshape(regret_def, newshape=np.shape(regret_att))

    regret_att = -regret_att
    regret_def = -regret_def

    return regret_att, regret_def


def NE_regret(regret_vect, ne_dict, identity):
    """
    Calculate the regret of each heuristic with respect to the combined game. The strategies of each heuristic only\
    include those in the NE of each heuristic.
    :param regret_vect: regret vector calculated from combined game.
    :param ne_dict: {"baseline": {0: np.array([1,0,1,0...]), 1: np.array([1,0,1,0...])},
    "RS": np.array([0,0,1,0...])} when a strategy is in a NE, that strategy is indicated by 1.
    :return:
    """

    indicator_dict = {}
    regret_dict = {}
    length_dict = {}
    indicator = np.array([])
    for method in ne_dict:
        length_dict[method] = len(ne_dict[method][identity])
        indicator = np.concatenate((indicator, ne_dict[method][identity]))

    idx = 0
    for method in ne_dict:
        indicator_dict[method] = indicator
        indicator_dict[method][:idx] = 0
        indicator_dict[method][idx + length_dict[method]:] = 0
        idx += length_dict[method]

    for method in ne_dict:
        regret_dict[method] = regret_vect * indicator_dict[method]

    return regret_dict


# Measure the regret of subgames during strategy exploration.
def regret_curve(ne_dict):
    pass


# Replaceability Implementation.
def strategy_replaceability():
    pass


def heuristic_replaceability():
    pass


def vector_convertor(ne_dict):
    '''
    Convert a NE to a indicator vector which indicates which strategy is in the support.
    :param ne_dict: {0:[0.3,0.2,0.5],1:[0.3,0.2,0.5]}
    :return:
    '''
    ones_def = ne_dict[0]
    ones_att = ne_dict[1]

    ones_def[ones_def>0] = 1
    ones_att[ones_att>0] = 1

    return ones_att, ones_def































