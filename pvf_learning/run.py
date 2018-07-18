import sys

import numpy as np
from joblib import Parallel, delayed

from mushroom.core import Core
from mushroom.environments.generators.taxi import generate_taxi
from mushroom.utils.callbacks import CollectDataset
from mushroom.utils.dataset import parse_dataset
from mushroom.utils.parameters import ExponentialDecayParameter, Parameter
from mushroom.utils.spaces import *

import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from envs.loop import  generate_loop
from envs.chain import  generate_chain

import gym
import argparse

from pvf_learning import PVFMaxMeanLearning, PVFWeightedLearning
sys.path.append('..')
from policy import VPIPolicy, WeightedPolicy


def experiment(n_approximators, policy, agent_alg, name):
    np.random.seed()
   
    
    # MDP
    if name=="Taxi":
        mdp = generate_taxi('../grid.txt')
        vmax=100
        nsteps_train=20000
        nsteps_test=20000
    elif name=="NChain-v0":
        mdp=generate_chain()
        vmax=500
        nsteps_train=1000
        nsteps_test=1000
    elif name=="Loop":
        mdp=generate_loop()
        vmax=60
        nsteps_train=1000
        nsteps_test=1000
    else:
        raise NotImplementedError
        

    # Policy
    # epsilon = ExponentialDecayParameter(value=1., decay_exp=.5,
    #                                     size=mdp.info.observation_space.size)
    epsilon = Parameter(0.)
    pi = policy(n_approximators, epsilon=epsilon)

    # Agent
    learning_rate = ExponentialDecayParameter(value=1., decay_exp=.2,
                                              size=mdp.info.size)
    algorithm_params = dict(
                learning_rate=learning_rate, 
                VMax=vmax
                )
    agent = agent_alg(pi, mdp.info, **algorithm_params)

    # Algorithm
    collect_dataset = CollectDataset()
    callbacks = [collect_dataset]
    core = Core(agent, mdp, callbacks)

    # Train
    n_steps = nsteps_train
    core.learn(n_steps=n_steps, n_steps_per_fit=1, quiet=False)

    dataset = collect_dataset.get()
    _, _, reward, _, _, _ = parse_dataset(dataset)
    pi.set_eval(True)
    dataset = core.evaluate(n_steps=nsteps_test, quiet=False)
    reward_test = [r[2] for r in dataset]

    return reward, reward_test


if __name__ == '__main__':
    n_experiment = 10
    n_approximators = 32
    parser = argparse.ArgumentParser()

    arg_game = parser.add_argument_group('Game')
    arg_game.add_argument("--name",
                          choices=['Taxi',
                                  'NChain-v0',
                                  'Loop'],
                         default='Taxi',
                          help='name of the environment to test in')

    args = parser.parse_args()
    env=args.name
    policy_name = {VPIPolicy: 'VPI', WeightedPolicy: 'Weighted'}
    update_rule={PVFMaxMeanLearning:"MaxMean", PVFWeightedLearning:"Weighted"}
    for p in [VPIPolicy, WeightedPolicy]:
        for a in [PVFMaxMeanLearning, PVFWeightedLearning]:
            out = Parallel(n_jobs=-1)(delayed(experiment)(
                n_approximators, p, a, env) for _ in range(n_experiment))

            r = [x[0] for x in out]
            r_test = [x[1] for x in out]
            np.save('%s/r_%s_%s.npy' % (env, policy_name[p], update_rule[a]), r)
            np.save('%s/r_test_%s_%s.npy' % (env, policy_name[p], update_rule[a]), r_test)
