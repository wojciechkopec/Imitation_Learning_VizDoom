from random import randint, random
from q_estimator import _create_convolution_layers
import os.path
import numpy as np

from sources.q_estimator import QEstimator
from sources.replay_memory import TransitionStore


class MultiQEstimator:
    def __init__(self, available_actions_count, resolution, K, sampleProb, sharedNetwork=True, replay_memory_size=10000,
                 store_trajectory=True, dump_file_name='out/weights.dump'):
        self.qEstimators = []
        self.K = K
        self.sampleProb = sampleProb
        self.discount_factor = 0.99
        self.store_trajectory = store_trajectory
        self.transition_store = TransitionStore(self.discount_factor)
        self.learning = True
        self.available_actions_count = available_actions_count
        path, name = os.path.split(dump_file_name)
        if sharedNetwork:
            convLayers = _create_convolution_layers(available_actions_count, resolution)
            create_conv_layers = lambda: convLayers
        else:
            create_conv_layers = lambda: _create_convolution_layers(available_actions_count, resolution)

        for i in range(0, K):
            self.qEstimators.append(
                QEstimator(available_actions_count, resolution, create_conv_layers,
                           int(replay_memory_size / (K * sampleProb)), False,
                           os.path.join(path, '') + str(i) + '_' + name))
        self.currentEstimator = self.qEstimators[0]

    def learn_from_transition(self, s1, a, s2, s2_isterminal, r, q2=float('-inf')):
        if self.store_trajectory:
            self.transition_store.store(s1, a, s2, s2_isterminal, r)
            if s2_isterminal:
                for t_s1, t_a, t_s2, t_s2_isterminal, t_r, t_q2 in self.transition_store.get_trajectory():
                    self._store(t_s1, t_a, t_s2, t_s2_isterminal, t_r, t_q2)

        else:
            self._store(s1, a, s2, s2_isterminal, r, -1)

        if s2_isterminal:
            self.currentEstimator = self.qEstimators[randint(0, self.K - 1)]

    def get_best_action(self, state):
        if self.learning:
            return self.get_best_action_learning(state)
        else:
            return self.get_best_action_testing(state)

    def get_best_action_learning(self, state):
        (dist, currentEstAction) = self._get_best_actions_dist(state)
        return currentEstAction, dist[currentEstAction] / self.K

    def get_best_action_testing(self, state):
        (dist, _) = self._get_best_actions_dist(state)
        chosenAction = np.argmax(dist)
        return chosenAction, dist[chosenAction] / self.K

    def _get_best_actions_dist(self, state):
        dist = np.zeros(self.available_actions_count)

        for e in self.qEstimators:
            action = e.get_best_action(state)[0]
            dist[action] += 1
            if e == self.currentEstimator:
                currentEstAction = action
        return dist, currentEstAction

    def _store(self, s1, a, s2, s2_isterminal, r, q2):
        for e in self.qEstimators:
            if (e == self.currentEstimator) or (random() <= self.sampleProb):
                e.learn_from_transition(s1, a, s2, s2_isterminal, r, q2)

    def suggest_action(self, state):
        return 3

    def save(self):
        for e in self.qEstimators:
            e.save();

    def load(self):
        for e in self.qEstimators:
            e.load();

    def learning_mode(self):
        self.learning = True

    def testing_mode(self):
        self.learning = False