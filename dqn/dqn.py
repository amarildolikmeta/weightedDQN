from copy import deepcopy

import numpy as np

from mushroom.algorithms.agent import Agent
from mushroom.approximators.regressor import Ensemble, Regressor

from replay_memory import Buffer, ReplayMemory


class DQN(Agent):
    def __init__(self, approximator, policy, mdp_info, batch_size,
                 target_update_frequency, initial_replay_size, train_frequency,
                 max_replay_size, fit_params=None, approximator_params=None,
                 n_approximators=1, history_length=1, clip_reward=True,
                 max_no_op_actions=0, no_op_action_value=0, p_mask=2 / 3.,
                 dtype=np.float32, weighted_update=False):
        self._fit_params = dict() if fit_params is None else fit_params

        self._batch_size = batch_size
        self._n_approximators = n_approximators
        self._clip_reward = clip_reward
        self._target_update_frequency = target_update_frequency // train_frequency
        self._max_no_op_actions = max_no_op_actions
        self._no_op_action_value = no_op_action_value
        self._p_mask = p_mask
        self.weighted_update=weighted_update
        self._replay_memory = ReplayMemory(
            mdp_info, initial_replay_size, max_replay_size, history_length,
            n_approximators, dtype
        )
        self._buffer = Buffer(history_length, dtype)

        self._n_updates = 0
        self._episode_steps = 0
        self._no_op_actions = None

        apprx_params_train = deepcopy(approximator_params)
        apprx_params_train['name'] = 'train'
        apprx_params_target = deepcopy(approximator_params)
        apprx_params_target['name'] = 'target'
        self.approximator = Regressor(approximator, **apprx_params_train)
        self.target_approximator = Regressor(approximator,
                                             **apprx_params_target)
        policy.set_q(self.approximator)

        self.target_approximator.model.set_weights(
            self.approximator.model.get_weights())

        super(DQN, self).__init__(policy, mdp_info)

    def fit(self, dataset):
        
        mask = np.random.binomial(1, self._p_mask,
                                  size=(len(dataset),
                                            self._n_approximators))
        self._replay_memory.add(dataset, mask)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, _, mask =\
                self._replay_memory.get(self._batch_size)

            q = np.array(self.approximator.predict(state))[0]
            q = q.reshape((self._n_approximators * self._batch_size, -1))
            q = q[np.arange(self._n_approximators * self._batch_size),
                  np.tile(action.ravel(), self._n_approximators)]
            q = q.reshape((self._n_approximators, self._batch_size)).T
            
            idxs = q.argsort()

            if self._clip_reward:
                reward = np.clip(reward, -1, 1)

            q_next = self._next_q(next_state, absorbing)
            q_next_ordered = np.sort(q_next)
            #order target values to match the source values
            for i in range(idxs.shape[0]):
                q_next[i, idxs[i]] = q_next_ordered[i]

            q = reward.reshape(self._batch_size,
                               1) + self.mdp_info.gamma * q_next
            self.approximator.fit(state, action, q, mask=mask,
                                  **self._fit_params)

            self._n_updates += 1

            if self._n_updates % self._target_update_frequency == 0:
                self._update_target()

    def _update_target(self):
        """
        Update the target network.

        """
        self.target_approximator.model.set_weights(
            self.approximator.model.get_weights())

    def _next_q(self, next_state, absorbing):
        """
        Args:
            next_state (np.ndarray): the states where next action has to be
                evaluated;
            absorbing (np.ndarray): the absorbing flag for the states in
                `next_state`.

        Returns:
            Maximum action-value for each state in `next_state`.

        """
        q = np.array(self.target_approximator.predict(next_state))[0]
        for i in range(q.shape[1]):
            if absorbing[i]:
                q[:, i, :] *= 1. - absorbing[i]
        
        if not self.weighted_update:
            #find best actions
            best_actions=np.argmax(np.mean(q,axis=0),axis=1)
            max_q = np.zeros((q.shape[1], q.shape[0]))
            for i in range(q.shape[1]):
                max_q[i, :]=q[:, i, best_actions[i]]
            return max_q
        else:
            N=q.shape[0]
            num_actions=q.shape[2]
            batch_size=q.shape[1]
            probs=np.zeros((batch_size, num_actions))
            weights = 1/N
            #calculate probability of being maximum
            for b in range(batch_size):
                for i in range(num_actions):
                    particles=q[:, b,i]
                    p=0
                    for k in range(N):
                        p2=1
                        p_k=particles[k]
                        for j in range(num_actions):
                            if(j!=i):
                                particles2=q[:,b,j]
                                p3=0
                                for l in range(N):
                                    if particles2[l]<=p_k:
                                        p3+=weights
                                p2*=p3
                        p+=weights*p2
                    probs[b, i]=p
            max_q = np.zeros((batch_size , N))
            for i in range(batch_size):
                particles=np.zeros(N)
                for j in range(num_actions):
                    particles+=q[:, i, j]*probs[i, j]
                max_q[i, :]=particles
            return max_q
            

    def draw_action(self, state):
        self._buffer.add(state)

        if self._episode_steps < self._no_op_actions:
            action = np.array([self._no_op_action_value])
            self.policy.update_epsilon(state)
        else:
            extended_state = self._buffer.get()

            action = super(DQN, self).draw_action(extended_state)

        self._episode_steps += 1
        return action

    def episode_start(self):
        if self._max_no_op_actions == 0:
            self._no_op_actions = 0
        else:
            self._no_op_actions = np.random.randint(
                self._buffer.size, self._max_no_op_actions + 1)
        self._episode_steps = 0
        self.policy.set_idx(np.random.randint(self._n_approximators))


class DoubleDQN(DQN):
    """
    Double DQN algorithm.
    "Deep Reinforcement Learning with Double Q-Learning".
    Hasselt H. V. et al.. 2016.

    """
    def _next_q(self, next_state, absorbing):
        q = np.array(self.approximator.predict(next_state))[0]
        tq = np.array(self.target_approximator.predict(next_state))[0]
        for i in range(q.shape[1]):
            if absorbing[i]:
                tq[:, i, :] *= 1. - absorbing[i]
        best_actions=np.argmax(np.mean(q,axis=0),axis=1)
        double_q = np.zeros((q.shape[1], q.shape[0]))
        for i in range(q.shape[1]):
            double_q[i, :]=tq[:, i, best_actions[i]]
        return double_q

