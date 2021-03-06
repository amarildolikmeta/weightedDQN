import sys
from mushroom.utils.dataset import compute_scores
import numpy as np
from joblib import Parallel, delayed
import pandas as pd
from tabulate import tabulate
from mushroom.core import Core
from mushroom.environments.generators.taxi import generate_taxi
from mushroom.utils.callbacks import CollectDataset
from mushroom.utils.dataset import parse_dataset
from mushroom.utils.parameters import ExponentialDecayParameter, Parameter
from mushroom.utils.spaces import *
import matplotlib.pyplot as plt
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from envs.loop import  generate_loop
from envs.chain import  generate_chain
from envs.river_swim import  generate_river
from envs.six_arms import  generate_arms
import gym
import argparse

from pvf_learning import ParticleQLearning, ParticleDoubleQLearning
sys.path.append('..')
from policy import VPIPolicy, WeightedPolicy


def experiment(n_approximators, policy, update_type, name,exponent, alg_version, double):
    np.random.seed()
   
    
    # MDP
    if name=="Taxi":
        mdp = generate_taxi('../grid.txt')
        vmax=10
        max_steps=100000
        evaluation_frequency=2000
        test_samples=10000
    elif name=="NChain-v0":
        mdp=generate_chain(horizon=1000)
        vmax=500
        max_steps=5000
        evaluation_frequency=100
        test_samples=10000
    elif name=="Loop":
        mdp=generate_loop(horizon=1000)
        vmax=60
        max_steps=5000
        evaluation_frequency=100
        test_samples=10000
    elif name=="SixArms":
        mdp=generate_arms(horizon=1000)
        vmax=10e10
        max_steps=25000
        evaluation_frequency=500
        test_samples=10000
    elif name=="RiverSwim":
        mdp=generate_river(horizon=1000)
        vmax=100000
        max_steps=25000
        evaluation_frequency=500
        test_samples=10000
    else:
        raise NotImplementedError
        

    # Policy
    # epsilon = ExponentialDecayParameter(value=1., decay_exp=.5,
    #                                     size=mdp.info.observation_space.size)
    epsilon = Parameter(0.)
    pi = policy(n_approximators, epsilon=epsilon)

    # Agent
    learning_rate = ExponentialDecayParameter(value=1., decay_exp=exponent,
                                              size=mdp.info.size)
    algorithm_params = dict(
                learning_rate=learning_rate, 
                q_max=vmax, 
                n_approximators=n_approximators, 
                update_type=update_type
                )
    if double:
        agent = ParticleDoubleQLearning(pi, mdp.info, **algorithm_params)
    else:
        agent = ParticleQLearning(pi, mdp.info, **algorithm_params)
    # Algorithm
    collect_dataset = CollectDataset()
    callbacks = [collect_dataset]
    core = Core(agent, mdp, callbacks)
    scores = list()
    scores_train=list()
    # Train
    for n_epoch in range(1, max_steps // evaluation_frequency + 1):
            print('- Learning:')
            # learning step
            pi.set_eval(False)
            core.learn(n_steps=evaluation_frequency,
                       n_steps_per_fit=1, quiet=False)
            dataset = collect_dataset.get()
            if name=="Taxi":
                scores_train.append(get_stats(dataset))
            elif name in ["SixArms"]:
                scores_train.append(compute_scores_Loop(dataset, horizon=500))
            else:
                scores_train.append(compute_scores_Loop(dataset))
            collect_dataset.clean()
            mdp.reset()
            print('- Evaluation:')
            # evaluation step
            pi.set_eval(True)
            dataset = core.evaluate(n_steps=test_samples,
                                    quiet=False)
            mdp.reset()
            scores.append(get_stats(dataset))
            #np.save(env + '/'+alg_version+'_scores.npy', scores)

    return scores_train, scores

def get_stats(dataset):
    score = compute_scores(dataset)
    return score

def compute_scores_Loop(dataset, horizon=100):
  
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
    n_approximators = 20
    parser = argparse.ArgumentParser()

    arg_game = parser.add_argument_group('Game')
    arg_game.add_argument("--name",
                          choices=['Taxi',
                                  'NChain-v0',
                                  'Loop',
                                  'SixArms', 
                                  'RiverSwim', 
                                  ''],
                         default='',
                          help='name of the environment to test in')
    arg_game.add_argument("--evaluation-frequency", type=int, default=1000,
                         help='Number of learning step before each evaluation.'
                              'This number represents an epoch.')
    arg_game.add_argument("--test-samples", type=int, default=1000,
                         help='Number of steps for each evaluation.')
    arg_game.add_argument("--max-steps", type=int, default=50000,
                         help='Total number of learning steps.')
    arg_game.add_argument("--double", action='store_true')
    args = parser.parse_args()
    #env=args.name
    exponent=0.1
    delta=0.1
    max=1
    policy_name = {VPIPolicy: 'VPIPolicy', WeightedPolicy: 'WeightedPolicy'}
    update_rule={"mean":"MaxMean", "weighted":"WeightedMax"}
    policies=[VPIPolicy, WeightedPolicy]
    updates=["mean", "weighted"]
    num_algs=len(policies)*len(updates) 
    tableData={"Algorithm":[""]*num_algs,"Num Experiments":[1.]*num_algs, "Phase Length":[1.]*num_algs,  "Avg Score Phase 1":[1.]*num_algs, "Std Dev Phase 1":[1.]*num_algs ,"Avg Score Phase 2":[1.]*num_algs,"Std Dev Phase 2":[1.]*num_algs}
    count=0
    exponent=0.5
    colors=["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#ffff33","#a65628","#f781bf"]
    double=args.double
    if args.name !='':
        envs=[args.name]
    else:
        envs=['NChain-v0','Loop', "Taxi", "SixArms", "RiverSwim"]   
    for p in policies:
        for a in updates:
            for env in envs:
                alg_version=policy_name[p]+"_"+update_rule[a]
                out = Parallel(n_jobs=-1)(delayed(experiment)(
                    n_approximators, p, a, env, exponent,alg_version, double) for _ in range(n_experiment))

                scores_train= [x[0] for x in out]
                scores = [x[1] for x in out]
                train=np.mean(scores_train, axis=0)
                test=np.mean(scores, axis=0)
                if double:
                    d="Double_"
                else:
                    d=""
                np.save(env + '/'+d+alg_version+'_train_scores.npy', train)
                np.save(env + '/'+d+alg_version+'_eval_scores.npy', test)
    
