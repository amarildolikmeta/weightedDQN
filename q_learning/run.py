import sys
from mushroom.utils.dataset import compute_scores
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from mushroom.core import Core
from mushroom.environments.generators.taxi import generate_taxi
from mushroom.utils.callbacks import CollectDataset
from mushroom.utils.dataset import parse_dataset
from mushroom.utils.parameters import ExponentialDecayParameter, Parameter
import pandas as pd
from tabulate import tabulate
from mushroom.algorithms.value.td import QLearning
sys.path.append('..')
from mushroom.policy.td_policy import EpsGreedy
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from envs.loop import  generate_loop
from envs.chain import  generate_chain
import argparse

def experiment( policy, name, alg_version):
    np.random.seed()

    # MDP
    
    if name=="Taxi":
        mdp = generate_taxi('../grid.txt')
        max_steps=100000
        evaluation_frequency=2000
        test_samples=10000
    elif name=="NChain-v0":
        mdp=generate_chain(horizon=1000)
        max_steps=5000
        evaluation_frequency=100
        test_samples=10000
    elif name=="Loop":
        mdp=generate_loop(horizon=1000)
        max_steps=5000
        evaluation_frequency=100
        test_samples=10000
    else:
        raise NotImplementedError
    # Policy
    # epsilon = ExponentialDecayParameter(value=1., decay_exp=.5,
    #                                     size=mdp.info.observation_space.size)
    epsilon_train = ExponentialDecayParameter(value=1., decay_exp=.5,
                                        size=mdp.info.observation_space.size)
    epsilon_test=Parameter(0)
    pi = policy( epsilon=epsilon_train)

    # Agent
    learning_rate = ExponentialDecayParameter(value=1., decay_exp=.2,
                                              size=mdp.info.size)
    algorithm_params = dict(learning_rate=learning_rate)
    agent = QLearning(pi, mdp.info, **algorithm_params)

    # Algorithm
    collect_dataset = CollectDataset()
    callbacks = [collect_dataset]
    core = Core(agent, mdp, callbacks)
    scores = list()
    scores_train = list()
    # Train
    for n_epoch in range(1, max_steps // evaluation_frequency + 1):
            print('- Learning:')
            # learning step
            pi.set_epsilon(epsilon_train)
            core.learn(n_steps=evaluation_frequency,
                       n_steps_per_fit=1, quiet=False)
            dataset = collect_dataset.get()
            if name=="Taxi":
                scores_train.append(get_stats(dataset))
            else:
                scores_train.append(compute_scores_Loop(dataset))        
            collect_dataset.clean()
            mdp.reset()
            print('- Evaluation:')
            # evaluation step
            pi.set_epsilon(epsilon_test)
            dataset = core.evaluate(n_steps=test_samples,
                                    quiet=False)
            mdp.reset()
            scores.append(get_stats(dataset))
            #np.save(env + '/'+alg_version+'_scores.npy', scores)

    return scores_train, scores
def get_stats(dataset):
    score = compute_scores(dataset)
    return score

def compute_scores_Loop(dataset, horizon=1000):
  
    scores = list()

    score = 0.
    episode_steps = 0
    n_episodes = 0
    for i in range(len(dataset)):
        score += dataset[i][2]
        episode_steps += 1
        if episode_steps==horizon:
            scores.append(score)
            score = 0.
            episode_steps = 0
            n_episodes += 1

    if len(scores) > 0:
        return np.min(scores), np.max(scores), np.mean(scores), n_episodes
    else:
        return 0, 0, 0, 0
if __name__ == '__main__':
    n_experiment = 10
    parser = argparse.ArgumentParser()

    arg_game = parser.add_argument_group('Game')
    arg_game.add_argument("--name",
                          choices=['Taxi',
                                  'NChain-v0',
                                  'Loop'],
                         default='Taxi',
                          help='name of the environment to test in')

    args = parser.parse_args()
    #env=args.name
    policy_name = {EpsGreedy: 'EpsGreedy'}
    policies=[EpsGreedy]
    num_algs=len(policies)
    tableData={"Algorithm":[""]*num_algs,"Num Experiments":[1.]*num_algs, "Phase Length":[1.]*num_algs,  "Avg Score Phase 1":[1.]*num_algs, "Std Dev Phase 1":[1.]*num_algs ,"Avg Score Phase 2":[1.]*num_algs,"Std Dev Phase 2":[1.]*num_algs}
    count=0
    envs=['NChain-v0','Loop']
    for p in policies:
        for env in envs:
            alg_version=policy_name[p]
            out = Parallel(n_jobs=-1)(delayed(experiment)(
                 p,  env,alg_version) for _ in range(n_experiment))
            scores_train= [x[0] for x in out]
            scores = [x[1] for x in out]
            train=np.mean(scores_train, axis=0)
            test=np.mean(scores, axis=0)
            np.save(env + '/'+alg_version+'_train_scores.npy', train)
            np.save(env + '/'+alg_version+'_eval_scores.npy', test)
        
