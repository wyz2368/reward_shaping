from attackgraph import json_op as jp
from baselines.common import models
from baselines.deepq.deepq import learn_multi_nets, Learner
import os
# import copy

DIR_def = os.getcwd() + '/defender_strategies/'
DIR_att = os.getcwd() + '/attacker_strategies/'

def training_att(game, mix_str_def, epoch, retrain = False):
    if len(mix_str_def) != len(game.def_str):
        raise ValueError("The length of mix_str_def and def_str does not match while training")

    # env = copy.deepcopy(game.env)
    print("training_att mix_str_def is ", mix_str_def)

    env = game.env
    env.reset_everything()

    env.set_training_flag(1)

    env.defender.set_mix_strategy(mix_str_def)
    env.defender.set_str_set(game.def_str)

    param_path = os.getcwd() + '/network_parameters/param.json'
    param = jp.load_json_data(param_path)

    if retrain:
        scope = 'att_str_retrain' + str(0) + '.pkl' + '/'
    else:
        scope = 'att_str_epoch' + str(epoch) + '.pkl' + '/'

    learner = Learner()
    with learner.graph.as_default():
        with learner.sess.as_default():
            act_att, a_BD = learner.learn_multi_nets(
                env,
                network = models.mlp(num_hidden=param['num_hidden'], num_layers=param['num_layers']),
                lr =param['lr'],
                total_timesteps=param['total_timesteps_att'],
                exploration_fraction=param['exploration_fraction'],
                exploration_final_eps=param['exploration_final_eps'],
                print_freq=param['print_freq'],
                param_noise=param['param_noise'],
                gamma=param['gamma'],
                prioritized_replay=param['prioritized_replay'],
                checkpoint_freq=param['checkpoint_freq'],
                scope = scope,
                epoch=epoch
            )
            print("Saving attacker's model to pickle.")
            if retrain:
                act_att.save(os.getcwd() + '/retrain_att/' + 'att_str_retrain' + str(0) + '.pkl', 'att_str_retrain' + str(0) + '.pkl' + '/')
            else:
                act_att.save(DIR_att + "att_str_epoch" + str(epoch) + ".pkl", 'att_str_epoch' + str(epoch) + '.pkl' + '/')
    learner.sess.close()
    return a_BD




def training_def(game, mix_str_att, epoch, retrain = False):
    if len(mix_str_att) != len(game.att_str):
        raise ValueError("The length of mix_str_att and att_str does not match while retraining")

    print("training_def mix_str_att is ", mix_str_att)

    # env = copy.deepcopy(game.env)
    env = game.env
    env.reset_everything()

    env.set_training_flag(0)

    env.attacker.set_mix_strategy(mix_str_att)
    env.attacker.set_str_set(game.att_str)

    param_path = os.getcwd() + '/network_parameters/param.json'
    param = jp.load_json_data(param_path)

    if retrain:
        scope = 'def_str_retrain' + str(0) + '.pkl' + '/'
    else:
        scope = 'def_str_epoch' + str(epoch) + '.pkl' + '/'

    learner = Learner()
    with learner.graph.as_default():
        with learner.sess.as_default():
            act_def, d_BD = learner.learn_multi_nets(
                env,
                network=models.mlp(num_hidden=param['num_hidden'], num_layers=param['num_layers']),
                lr=param['lr'],
                total_timesteps=param['total_timesteps_def'],
                exploration_fraction=param['exploration_fraction'],
                exploration_final_eps=param['exploration_final_eps'],
                print_freq=param['print_freq'],
                param_noise=param['param_noise'],
                gamma=param['gamma'],
                prioritized_replay=param['prioritized_replay'],
                checkpoint_freq=param['checkpoint_freq'],
                scope = scope,
                epoch=epoch
            )
            print("Saving defender's model to pickle.")
            if retrain:
                act_def.save(os.getcwd() + '/retrain_def/' + 'def_str_retrain' + str(0) + '.pkl', 'def_str_retrain' + str(0) + '.pkl' + '/')
            else:
                act_def.save(DIR_def + "def_str_epoch" + str(epoch) + ".pkl", "def_str_epoch" + str(epoch) + '.pkl' + '/')
    learner.sess.close()
    return d_BD



# for all strategies learned by retraining, the scope index is 0.
def training_hado_att(game):
    param = game.param
    mix_str_def = game.hado_str(identity=0, param=param)

    if len(mix_str_def) != len(game.def_str):
        raise ValueError("The length of mix_str_def and def_str does not match while retraining")

    # env = copy.deepcopy(game.env)
    env = game.env
    env.reset_everything()

    env.set_training_flag(1)

    env.defender.set_mix_strategy(mix_str_def)
    env.defender.set_str_set(game.def_str)

    param_path = os.getcwd() + '/network_parameters/param.json'
    param = jp.load_json_data(param_path)

    learner = Learner(retrain=True, freq=param['retrain_freq'])
    # TODO: add epoch???
    with learner.graph.as_default():
        with learner.sess.as_default():
            act_att, _ = learner.learn_multi_nets(
                env,
                network = models.mlp(num_hidden=param['num_hidden'], num_layers=param['num_layers']),
                lr =param['lr'],
                total_timesteps=param['retrain_timesteps'],
                exploration_fraction=param['exploration_fraction'],
                exploration_final_eps=param['exploration_final_eps'],
                print_freq=param['print_freq'],
                param_noise=param['param_noise'],
                gamma=param['gamma'],
                prioritized_replay=param['prioritized_replay'],
                checkpoint_freq=param['checkpoint_freq'],
                scope = 'att_str_retrain' + str(0) + '.pkl' + '/',
                load_path=os.getcwd() + '/retrain_att/' + 'att_str_retrain' + str(0) + '.pkl'
            )
            # print("Saving attacker's model to pickle.")
            # act_att.save(os.getcwd() + '/retrain_att/' + 'att_str_retrain' + str(epoch) + ".pkl", 'att_str_epoch' + str(epoch) + '.pkl' + '/')
    learner.sess.close()


def training_hado_def(game):
    param = game.param
    mix_str_att = game.hado_str(identity=1, param=param)

    if len(mix_str_att) != len(game.att_str):
        raise ValueError("The length of mix_str_att and att_str does not match while training")

    # env = copy.deepcopy(game.env)
    env = game.env
    env.reset_everything()

    env.set_training_flag(0)

    env.attacker.set_mix_strategy(mix_str_att)
    env.attacker.set_str_set(game.att_str)

    param_path = os.getcwd() + '/network_parameters/param.json'
    param = jp.load_json_data(param_path)

    learner = Learner(retrain=True, freq=param['retrain_freq'])
    with learner.graph.as_default():
        with learner.sess.as_default():
            act_def, _ = learner.learn_multi_nets(
                env,
                network=models.mlp(num_hidden=param['num_hidden'], num_layers=param['num_layers']),
                lr=param['lr'],
                total_timesteps=param['retrain_timesteps'],
                exploration_fraction=param['exploration_fraction'],
                exploration_final_eps=param['exploration_final_eps'],
                print_freq=param['print_freq'],
                param_noise=param['param_noise'],
                gamma=param['gamma'],
                prioritized_replay=param['prioritized_replay'],
                checkpoint_freq=param['checkpoint_freq'],
                scope = 'def_str_retrain' + str(0) + '.pkl' + '/',
                load_path = os.getcwd() + '/retrain_def/' + 'def_str_retrain' + str(0) + '.pkl'
            )
            # print("Saving defender's model to pickle.")
            # act_def.save(os.getcwd() + '/retrain_def/' + 'def_str_retrain' + str(epoch) + ".pkl", "def_str_epoch" + str(epoch) + '.pkl' + '/')
    learner.sess.close()