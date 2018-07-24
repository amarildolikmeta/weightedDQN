
import numpy as np

from mushroom.algorithms.value import TD
from mushroom.utils.table import EnsembleTable


class PVF(TD):
    def __init__(self, policy, mdp_info, learning_rate, n_approximators=32,
                 VMax=500):
        self._n_approximators = n_approximators
        self.Q = EnsembleTable(self._n_approximators, mdp_info.size)
        self._means=np.zeros(shape=mdp_info.size)
        quantiles=[k*VMax/(n_approximators-1) for k in range(n_approximators)]
        mean=np.mean(quantiles)
        for j in range(mdp_info.size[0]):
            for k in range(mdp_info.size[1]):
                #particles=np.sort(np.random.uniform(0,VMax, self._n_approximators))
                #self._means[j, k]=np.mean(particles)
                self._means[j, k]=mean
                for i in range(len(self.Q.model)):
                    self.Q.model[i].table[j, k] = quantiles[i]
        super(PVF, self).__init__(self.Q, policy, mdp_info,
                                           learning_rate)
        
    def episode_start(self):
        return super(PVF, self).episode_start()
    def getMax(self, V):
        #brake ties
        maximums=np.where(V==np.max(V))[0]
        return np.random.choice(maximums)
    def _update(self, state, action, reward, next_state, absorbing):
        raise NotImplementedError
    

class PVFMaxMeanLearning(PVF):
    
    def _update(self, state, action, reward, next_state, absorbing):
        q_current = np.array([x[state, action] for x in self.Q.model])
        q_current=np.sort(q_current)
        alpha=self.alpha(state, action)
        if not absorbing:
            means=self._means[next_state, :]
            best_action=self.getMax(means)
            q_next=np.sort(np.array([x[next_state,best_action] for x in self.Q.model]))
        else:
            q_next=np.zeros(self._n_approximators)
        target=reward+self.mdp_info.gamma*q_next
        q_current=(1-alpha)*q_current+(alpha)*target
        self._means[state, action]=np.mean(q_current)
        for i in range(len(self.Q.model)):
            self.Q.model[i][state, action] =q_current[i]



class PVFWeightedLearning(PVF):
    
    def _update(self, state, action, reward, next_state, absorbing):
        alpha=self.alpha(state, action)
        q_current = np.sort(np.array([x[state, action] for x in self.Q.model]))
        N=len(self.Q.model)
        num_actions=self.Q.model[0].shape[1]
        probs=np.zeros(num_actions)
        weights = 1/N
        #calculate probability of being maximum
        if not absorbing:
            for i in range(num_actions):
                    particles= np.array([x[next_state, i] for x in self.Q.model])
                    p=0
                    for k in range(N):
                        p2=1
                        p_k=particles[k]
                        for j in range(num_actions):
                            if(j!=i):
                                particles2= np.array([x[next_state, j] for x in self.Q.model])
                                count=len(particles2[particles2<=p_k])
                                p3=count*weights
                                p2*=p3
                        p+=weights*p2
                    probs[i]=p
            q_next= np.zeros(N)
            for j in range(num_actions):
                particles=np.array([x[next_state, j] for x in self.Q.model])
                q_next+=particles*probs[j]
            q_next=np.sort(q_next)
        else:
            q_next=np.zeros(N)
        target=reward+self.mdp_info.gamma*q_next
        q_current=(1-alpha)*q_current+(alpha)*target
        self._means[state, action]=np.mean(q_current)
        for i in range(len(self.Q.model)):
            self.Q.model[i][state, action] =q_current[i]

