import numpy as np
from attackgraph import subproc
from attackgraph import file_op as fp
import os

# Input arguments: payoff matrix for the defender, poDef; payoff matrix for the attacker, poAtt.
# In a payoff matrix example: 1 2
#                               3 5
#                               6 7
# There are 3 defender strategies (3 rows) and 2 attacker strategies (2 columns).
# NFG File format: Payoff version

gambit_DIR = os.getcwd() + '/gambit_data/payoffmatrix.nfg'

def encode_gambit_file(poDef, poAtt):
    try:
        if poDef.shape != poAtt.shape:
            raise Exception("Inputted payoff matrix for defender and attacker must be of same shape.")
    except Exception as error:
        print(repr(error))
        return -1
    # Write header
    with open(gambit_DIR, "w") as nfgFile:
        nfgFile.write('NFG 1 R "Attackgroup"\n{ "Defender" "Attacker" } ')
        # Write strategies
        nfgFile.write('{ ' + str(poDef.shape[0]) + ' ' + str(poDef.shape[1]) +' }\n\n')
        # Write outcomes
        for i in range(poDef.shape[1]):
            for j in range(poDef.shape[0]):
                nfgFile.write(str(poDef[j][i]) + " ")
                nfgFile.write(str(poAtt[j][i]) + " ")

    # Gambit passing and NE calculation to come later.

def gambit_analysis(timeout):
    if not fp.isExist(gambit_DIR):
        raise ValueError(".nfg file does not exist!")
    command_str = "gambit-lcp -q " + os.getcwd() + "/gambit_data/payoffmatrix.nfg -d 8 > " + os.getcwd() + "/gambit_data/nash.txt"
    subproc.call_and_wait_with_timeout(command_str, timeout)


def decode_gambit_file():
    nash_DIR = os.getcwd() + '/gambit_data/nash.txt'
    if not fp.isExist(nash_DIR):
        raise ValueError("nash.txt file does not exist!")
    with open(nash_DIR,'r') as f:
        nash = f.readline()
        if len(nash.strip()) == 0:
            return 0,0

    nash = nash[3:]
    nash = nash.split(',')
    new_nash = []
    for i in range(len(nash)):
        new_nash.append(convert(nash[i]))

    new_nash = np.array(new_nash)
    new_nash = np.round(new_nash, decimals=8)
    nash_def = new_nash[:int(len(new_nash)/2)]
    nash_att = new_nash[int(len(new_nash)/2):]

    return nash_att, nash_def

def do_gambit_analysis(poDef, poAtt, maxent=False, minent=False, num_nash=5):
    timeout = 600
    encode_gambit_file(poDef, poAtt) #TODO:change timeout adaptive
    while True:
        gambit_analysis(timeout)
        if not maxent and not minent:
            # pick random NE.
            nash_att, nash_def = decode_gambit_file()
        else:
            # pick required NE.
            nash_att, nash_def = decode_gambit_file_multiple_NEs(maxent, minent, num_nash)
        timeout += 120
        if timeout > 7200:
            print("Gambit has been running for more than 2 hour.!")
        if isinstance(nash_def,np.ndarray) and isinstance(nash_att,np.ndarray):
            break
        print("Timeout has been added by 120s.")
    print('gambit_analysis done!')
    return nash_att, nash_def


def convert(s):
    try:
        return float(s)
    except ValueError:
        num, denom = s.split('/')
        return float(num) / float(denom)

# ne is a dic。 nash is a numpy. 0: def, 1: att
def add_new_NE(game, nash_att, nash_def, epoch):
    if not isinstance(nash_att,np.ndarray):
        raise ValueError("nash_att is not numpy array.")
    if not isinstance(nash_def,np.ndarray):
        raise ValueError("nash_def is not numpy array.")
    if not isinstance(epoch,int):
        raise ValueError("Epoch is not an integer.")
    ne = {}
    ne[0] = nash_def
    ne[1] = nash_att
    game.add_nasheq(epoch, ne)


# Calculate the entropy of NE.
def maxent_NE(nash_list):
    H_list = np.array([])
    if len(nash_list) == 0:
        raise ValueError("The length of Nash list is zero.")
    for nash in nash_list:
        np.append(H_list, entropy_NE(nash))

    nash_selected = nash_list[np.argmax(H_list)]
    return nash_selected


def minent_NE(nash_list):
    H_list = np.array([])
    if len(nash_list) == 0:
        raise ValueError("The length of Nash list is zero.")
    for nash in nash_list:
        np.append(H_list, entropy_NE(nash))

    nash_selected = nash_list[np.argmin(H_list)]
    return nash_selected


def entropy_NE(nash):
    # This function calculates an approximated entropy since
    # it sets a shreshold for probability.
    H = 0
    for p in nash:
        if p <= 0.05:
            continue
        H -= p*np.log(p)
    return H

def decode_gambit_file_multiple_NEs(maxent, minent, num_nash):
    nash_DIR = os.getcwd() + '/gambit_data/nash.txt'
    if not fp.isExist(nash_DIR):
        raise ValueError("nash.txt file does not exist!")
    num_lines = file_len(nash_DIR)
    print("Number of NE is ", num_lines)
    if num_lines >= num_nash:
        num_lines = num_nash
        print("Number of NE is constrained by the num_nash.")
    nash_att_list = []
    nash_def_list = []
    with open(nash_DIR,'r') as f:
        for i in np.arange(num_lines):
            nash = f.readline()
            if len(nash.strip()) == 0:
                continue
            nash = nash[3:]
            nash = nash.split(',')
            new_nash = []
            for j in range(len(nash)):
                new_nash.append(convert(nash[j]))

            new_nash = np.array(new_nash)
            new_nash = np.round(new_nash, decimals=8)
            nash_def = new_nash[:int(len(new_nash)/2)]
            nash_att = new_nash[int(len(new_nash)/2):]
            nash_att_list.append(nash_att)
            nash_def_list.append(nash_def)

    if maxent:
        nash_att = maxent_NE(nash_att_list)
        nash_def = maxent_NE(nash_def_list)
    elif minent:
        nash_att = minent_NE(nash_att_list)
        nash_def = minent_NE(nash_def_list)
    else:
        raise ValueError("Falsely enter the multiple NE function.")

    return nash_att, nash_def

# Get number of lines in a text file.
def file_len(fname):
    num_lines = sum(1 for line in open(fname))
    return num_lines

path = os.getcwd() + '/gambit_data/nash.txt'
with open(path,'r') as f:
    for i in np.arange(5):
        nash = f.readline()
        print(nash)

