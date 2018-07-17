import sys

import numpy as np
from joblib import Parallel, delayed

from mushroom.core import Core
from mushroom.environments.generators.taxi import generate_taxi
from mushroom.utils.callbacks import CollectDataset
from mushroom.utils.dataset import parse_dataset
from mushroom.utils.parameters import ExponentialDecayParameter, Parameter

from pvf_learning import PVFMaxMeanLearning, PVFWeightedLearning
sys.path.append('..')
from policy import VPIPolicy, WeightedPolicy


def experiment(n_approximators, policy, agent_alg):
    np.random.seed()

    # MDP
    mdp = generate_taxi('../grid.txt')

    # Policy
    # epsilon = ExponentialDecayParameter(value=1., decay_exp=.5,
    #                                     size=mdp.info.observation_space.size)
    epsilon = Parameter(0.)
    pi = policy(n_approximators, epsilon=epsilon)

    # Agent
    learning_rate = ExponentialDecayParameter(value=1., decay_exp=.3,
                                              size=mdp.info.size)
    algorithm_params = dict(
                learning_rate=learning_rate, 
                VMax=500
                )
    agent = agent_alg(pi, mdp.info, **algorithm_params)

    # Algorithm
    collect_dataset = CollectDataset()
    callbacks = [collect_dataset]
    core = Core(agent, mdp, callbacks)

    # Train
    n_steps = 6e5
    core.learn(n_steps=n_steps, n_steps_per_fit=1, quiet=True)

    dataset = collect_dataset.get()
    _, _, reward, _, _, _ = parse_dataset(dataset)
    pi.set_eval(True)
    dataset = core.evaluate(n_steps=1000, quiet=True)
    reward_test = [r[2] for r in dataset]

    return reward, reward_test


if __name__ == '__main__':
    n_experiment = 10
    n_approximators = 32
    env="Taxi"
    policy_name = {VPIPolicy: 'VPI', WeightedPolicy: 'Weighted'}
    update_rule={PVFMaxMeanLearning:"MaxMean", PVFWeightedLearning:"Weighted"}
    for p in [VPIPolicy, WeightedPolicy]:
        for a in [PVFMaxMeanLearning, PVFWeightedLearning]:
            out = Parallel(n_jobs=-1)(delayed(experiment)(
                n_approximators, p, a) for _ in range(n_experiment))

            r = [x[0] for x in out]
            r_test = [x[1] for x in out]
            np.save('%s_r_%s_%s.npy' % (env, policy_name[p], update_rule[a]), r)
            np.save('%s_r_test_%s_%s.npy' % (env, policy_name[p], update_rule[a]), r_test)
