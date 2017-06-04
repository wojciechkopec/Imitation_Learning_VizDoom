import numpy as np
from random import sample, randint, random, shuffle
import itertools


class ReplayMemory:
    def __init__(self, capacity, resolution):
        state_shape = (capacity, resolution[0], resolution[1], 1)
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.bool_)
        self.expectedQValue = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0
        self.locked_samples_size = 0

    def add_transition(self, s1, action, s2, isterminal, reward, q2):
        self.s1[self.pos, :, :, 0] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, :, :, 0] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward
        self.expectedQValue[self.pos] = q2

        self.pos = ((self.pos + 1) % (self.capacity - self.locked_samples_size)) + self.locked_samples_size
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i], self.expectedQValue[i]

    def lock_current_samples(self):
        self.locked_samples_size = self.pos

class PerActionReplayMemory(ReplayMemory):
    
    def __init__(self, single_capacity, resolution, number_of_actions, weights = None):
        self.submemories = []
        if weights is None:
            weights = np.ones(number_of_actions)
        self.weights = weights
        for i in range(number_of_actions):
            self.submemories.append(ReplayMemory(single_capacity, resolution))
        self.size = 0

        
    def add_transition(self, s1, action, s2, isterminal, reward, q2):
        submemory = self.submemories[action]
        size_before = submemory.size
        submemory.add_transition(s1, action, s2, isterminal, reward, q2)
        self.size+=submemory.size - size_before

    def get_sample(self, sample_size):
        subsample_size = sample_size/len(self.submemories)
        subsample_size = max(1, sample_size)
        subsamples = []
        cnt = 0
        while cnt < sample_size:
            for i, submemory in enumerate(self.submemories):
                size = np.min((submemory.size, int(subsample_size * self.weights[i])))
                if size > 0:
                    cnt += size
                    subsamples.append(submemory.get_sample(size))

        result = map(lambda x: np.concatenate(x), zip(*subsamples))

        result = list(result)
        indexes = range(len(result[0]))
        shuffle(indexes)
        for i in range(len(result)):
            result[i] = result[i][indexes]
        return map(lambda x: x[0:sample_size], result)

    def lock_current_samples(self):
        for submemory in self.submemories:
            submemory.lock_current_samples()

class TransitionStore:
    def __init__(self, discountFactor):
        self.trajectory = []
        self.updatedTransition = []
        self.discountFactor = discountFactor

    def store(self, s1, a, s2, s2_isterminal, r):
        self.trajectory.append((s1, a, s2, s2_isterminal, r))
        if s2_isterminal:
            self._traverse_trajectory()

    def get_trajectory(self):
        result = self.updatedTransition
        self.updatedTransition = []
        return result

    def _traverse_trajectory(self):
        prevQValue = 0
        for t in reversed(self.trajectory):
            expectedQValue = t[4] + self.discountFactor * prevQValue
            self.updatedTransition.append((t[0], t[1], t[2], t[3], t[4], expectedQValue))
            prevQValue = expectedQValue
        self.trajectory = []





