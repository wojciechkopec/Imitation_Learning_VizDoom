from random import randint, random
import os.path

from sources.q_estimator import QEstimator

class MultiQEstimator:

    def __init__(self,available_actions_count,resolution,K,sampleProb,dumpFileName ='out/weights.dump'):
        self.qEstimators = []
        self.K = K
        self.sampleProb = sampleProb
        path, name = os.path.split(dumpFileName)
        for i in range(0, K):
            self.qEstimators.append(QEstimator(available_actions_count,resolution, os.path.join(path, '')+str(i)+'_'+name))
        self.currentEstimator = self.qEstimators[0]


    def learn_from_transition(self,s1, a, s2, s2_isterminal, r):
        for e in self.qEstimators:
            if (e == self.currentEstimator) or (random() <= self.sampleProb):
                e.learn_from_transition(s1, a, s2, s2_isterminal, r)
        if s2_isterminal:
            self.currentEstimator = self.qEstimators[randint(0,self.K - 1)]

    def get_best_action(self, state):
        action = self.currentEstimator.get_best_action(state)[0]
        matches = 0.0
        for e in self.qEstimators:
            otherAction = e.get_best_action(state)[0]
            if(otherAction == action):
                matches = matches+1
        return action,matches/self.K


    def save(self):
        for e in self.qEstimators:
            e.save();

    def load(self):
        for e in self.qEstimators:
            e.load();
