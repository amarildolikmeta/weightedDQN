import numpy as np

from mushroom.policy.td_policy import TDPolicy
from mushroom.utils.parameters import Parameter


class BootPolicy(TDPolicy):
    def __init__(self, n_approximators, epsilon=None):
        if epsilon is None:
            epsilon = Parameter(0.)

        super(BootPolicy, self).__init__()

        self._n_approximators = n_approximators
        self._epsilon = epsilon
        self._evaluation = False
        self._idx = None

    def draw_action(self, state):
        if not np.random.uniform() < self._epsilon(state):
            if self._evaluation:
                if isinstance(self._approximator.model, list):
                    q_list = list()
                    for q in self._approximator.model:
                        q_list.append(q.predict(state))
                else:
                    q_list = self._approximator.predict(state)

                max_as, count = np.unique(np.argmax(q_list, axis=1),
                                          return_counts=True)
                max_a = np.array([max_as[np.random.choice(
                    np.argwhere(count == np.max(count)).ravel())]])

                return max_a
            else:
                q = self._approximator.predict(state, idx=self._idx)

                max_a = np.argwhere(q == np.max(q)).ravel()
                if len(max_a) > 1:
                    max_a = np.array([np.random.choice(max_a)])

                return max_a
        else:
            return np.array([np.random.choice(self._approximator.n_actions)])

    def set_epsilon(self, epsilon):
        self._epsilon = epsilon

    def set_eval(self, eval):
        self._evaluation = eval

    def set_idx(self, idx):
        self._idx = idx

    def update_epsilon(self, state):
        self._epsilon(state)


class WeightedPolicy(TDPolicy):
    def __init__(self, n_approximators, epsilon=None):
        if epsilon is None:
            epsilon = Parameter(0.)

        super(WeightedPolicy, self).__init__()

        self._n_approximators = n_approximators
        self._epsilon = epsilon
        self._evaluation = False

    def draw_action(self, state):
        if not np.random.uniform() < self._epsilon(state):
            if self._evaluation:
                if isinstance(self._approximator.model, list):
                    q_list = list()
                    for q in self._approximator.model:
                        q_list.append(q.predict(state))
                else:
                    q_list = self._approximator.predict(state).squeeze()
                q=np.array(q_list)
                num_actions=q.shape[1]
                probs=np.zeros(num_actions)
                N=q.shape[0]
                weights=1/N
                for i in range(num_actions):
                    particles=q[:, i]
                    p=0
                    for k in range(N):
                        p2=1
                        p_k=particles[k]
                        for j in range(num_actions):
                            if(j!=i):
                                particles2=q[:, j]
                                count=len(particles2[particles2<=p_k])
                                p3=count*weights
                                p2*=p3
                        p+=weights*p2
                    probs[i]=p
                return np.array([np.random.choice(np.where(probs==np.max(probs))[0])])
            else:
                if isinstance(self._approximator.model, list):
                    q_list = list()
                    for i in range(self._n_approximators):
                        q_list.append(self._approximator.predict(state, idx=i))
                else:
                    q_list = self._approximator.predict(state).squeeze()

                qs = np.array(q_list)

                samples = np.ones(self._approximator.n_actions)
                for a in range(self._approximator.n_actions):
                    idx = np.random.randint(self._n_approximators)
                    samples[a] = qs[idx, a]

                max_a = np.array([np.argmax(samples)])

                return max_a
        else:
            return np.array([np.random.choice(
                self._approximator.n_actions)])
    
            
    def set_epsilon(self, epsilon):
        self._epsilon = epsilon

    def set_eval(self, eval):
        self._evaluation = eval

    def set_idx(self, idx):
        pass

    def update_epsilon(self, state):
        self._epsilon(state)

class VPIPolicy(BootPolicy):
        def draw_action(self, state):
            if not np.random.uniform() < self._epsilon(state):
                if self._evaluation:
                    if isinstance(self._approximator.model, list):
                        q_list = list()
                        for q in self._approximator.model:
                            q_list.append(q.predict(state))
                    else:
                        q_list = self._approximator.predict(state).squeeze()
                    q=np.array(q_list)
                    num_actions=q.shape[1]
                    probs=np.zeros(num_actions)
                    N=q.shape[0]
                    weights=1/N
                    for i in range(num_actions):
                        particles=q[:, i]
                        p=0
                        for k in range(N):
                            p2=1
                            p_k=particles[k]
                            for j in range(num_actions):
                                if(j!=i):
                                    particles2=q[:, j]
                                    count=len(particles2[particles2<=p_k])
                                    p3=count*weights
                                    p2*=p3
                            p+=weights*p2
                        probs[i]=p
                    return np.array([np.random.choice(np.where(probs==np.max(probs))[0])])
                    
                else:
                    if isinstance(self._approximator.model, list):
                        q_list = list()
                        for i in range(self._n_approximators):
                            q_list.append(self._approximator.predict(state, idx=i))
                        q=np.array(q_list)
                    else:
                        q = self._approximator.predict(state).squeeze()
                    N=q.shape[0]
                    num_actions=q.shape[1]
                    means=np.mean(q, axis=0)
                    ranking=np.zeros(num_actions)
                    best_action, second_best=get_2_best_actions(means)
                    mean1=means[best_action]
                    mean2=means[second_best]
                    weights = 1/N
                    for i in range(num_actions):
                        mean=means[i]
                        particles=q[:,i]
                        vpi=0
                        if i==best_action:
                            for j in range(N):
                                if particles[j]<=mean2:
                                    vpi+=(mean2-particles[j])*weights
                            ranking[i]=vpi+mean
                        else :
                            for j in range(N):
                                if particles[j]>=mean1:
                                    vpi+=(particles[j]-mean1)*weights
                            ranking[i]=vpi+mean
                    return np.array([getMax(ranking)])
            else:
                    return np.array([np.random.choice(self._approximator.n_actions)])

        def set_epsilon(self, epsilon):
            self._epsilon=epsilon
            
        def set_eval(self, eval):
            self._evaluation = eval

        def set_idx(self, idx):
            pass

        def update_epsilon(self, state):
            pass


            
def get_2_best_actions( A):
    max1=np.argmax(A[0:2])
    max2=np.argmin(A[0:2])
    if max2==max1 :
        max2=(1-max1)%2
    for i in range(2, len(A)):
        if A[i]>=A[max1]:
            max2=max1
            max1=i
        elif A[i]>=A[max2]:
            max2=i
    return max1, max2
        
def getMax( V):
    #brake ties
    maximums=np.where(V==np.max(V))[0]
    return np.random.choice(maximums)
    
