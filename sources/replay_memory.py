import numpy as np
from random import sample, randint, random


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

    def add_transition(self, s1, action, s2, isterminal, reward, q2):
        self.s1[self.pos, :, :, 0] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, :, :, 0] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward
        self.expectedQValue[self.pos] = q2

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i], self.expectedQValue[i]


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





