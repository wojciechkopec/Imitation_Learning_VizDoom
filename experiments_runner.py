#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import itertools as it
from random import randint, random
from time import time, sleep

import numpy as np
import skimage.color
import skimage.transform
from tqdm import trange
from datetime import datetime
import os
import json
from sources.replay_memory import PerActionReplayMemory, ReplayMemory

from vizdoom import *


# Converts and downsamples the input image
def preprocessIMG(img, resolution):
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    return img

def play(agentName, config, directory, agents):
    runner = ExperimentsRunner(agentName, config, agents[agentName], directory)
    runner.game.close()
    runner.play(runner.q_estimator)



def run(agentName, config, iterations, agents):
    start = datetime.now()
    startTime = time()
    dir_name = "out/" + start.strftime("%Y%m%d_%H%M_") + agentName + "_" + config.get_scenario()
    base_dir_name = dir_name
    i = 2
    while os.path.exists(dir_name):
        dir_name = base_dir_name + "_" + str(i)
        i += 1

    os.makedirs(dir_name)

    with open(dir_name + "/config", "w") as configFile:
        configFile.writelines([agentName])
        configFile.writelines(json.dumps(config.jsonable(), indent=4))

    resultsFile = open(dir_name + "/results.csv", "w")
    resultsFile.writelines([" " + " ".join(map(str, range(1, config.epochs + 1))) + "\n"])
    resultsSumsFile = open(dir_name + "/sumResults.csv", "w")
    resultsSumsFile.writelines([" " + "total_score time\n"])
    sumFromAllRuns = 0.0
    for i in range(1, iterations + 1):
        print "Iteration %d of agent %s" % (i, agentName)
        start = time()
        results = ExperimentsRunner(agentName, config, agents[agentName], dir_name).run()
        end = time()
        totalSum = sum(results)

        resultsFile.writelines([str(i) + " " + " ".join(map(str, results)) + "\n"])
        resultsSumsFile.writelines(
            [str(i) + " " + (str(totalSum) + " " + str(end - startTime)) + "\n"])
        resultsFile.flush()
        resultsSumsFile.flush()

        sumFromAllRuns += totalSum
        print "Finished iteration %d in %.2fs with score %.3f" % (i, (end - start), totalSum)
        print "%.2fs elapsed, average score for agent %s so far: %.3f" % (
        end - startTime, agentName, sumFromAllRuns / i)
    resultsFile.close()
    resultsSumsFile.close()
    print "Finished %d runs for agent %s with score %.3f" % (iterations, agentName, sumFromAllRuns / iterations)
    return sumFromAllRuns / iterations


class ExperimentsRunner:
    def __init__(self, agent_name, config, agent, directory_path):
        self.agent_name = agent_name
        self.config = config
        # Create Doom instance
        self.game = self.initialize_vizdoom(config.config_file_path)

        # Action = which buttons are pressed
        self.n = self.game.get_available_buttons_size()
        self.actions = [list(a) for a in it.product([0, 1], repeat=self.n)]
        self.q_estimator = agent(self.actions, config, directory_path + "/weights.dump")
        expert_config = self.config.expert_config
        # self.feed_expert_trajectories2(expert_config)
        self.q_estimator.memory.lock_current_samples()
        self.episode_state = {'explore': False}

    def feed_expert_trajectories(self, expert_config):
        for transition in expert_config.feed_memory:
            (s1, action, s2, r, isterminal) = transition
            s1 = self.preprocess(s1)
            s2 = self.preprocess(s2) if not isterminal else None
            for i, a in enumerate(self.actions):
                reward = r if a == action else expert_config.reward_for_another_action + r
                self.q_estimator.learn_from_transition(s1, self.actions.index(a), s2, reward, isterminal)

    def feed_expert_trajectories2(self, expert_config):
        memory = PerActionReplayMemory(len(expert_config.feed_memory), self.config.resolution, len(self.actions),
                                       [0, 1, 1, 1, 1, 1, 1, 1, 1])

        batch_size = 64
        print "Decoding expert trajectories"
        for t in trange(len(expert_config.feed_memory)):
            transition = expert_config.feed_memory[t]
            (s1, action, s2, r, isterminal) = transition
            s1 = self.preprocess(s1)
            if self.actions.index(action) != 0:
                memory.add_transition(s1, self.actions.index(action), s1, isterminal, r, -1)
                self.q_estimator.memory.add_transition(s1, self.actions.index(action), s1, isterminal, r, -1)

        print "Learning expert trajectories"
        for it in trange(int(len(expert_config.feed_memory)/6)):
            s1, a, _, _, _, _ = memory.get_sample(batch_size)
            e, acc = self.q_estimator.learn_actions(s1, a)
            if it % 30 == 0:
                print e, acc

    def preprocess(self, img):
        return preprocessIMG(img, self.config.resolution)

    def perform_learning_step(self, epoch):
        """ Makes an action according to eps-greedy policy, observes the result
        (next state, reward) and learns from the transition"""

        def exploration_rate(epoch):
            """# Define exploration rate change over time"""
            start_eps = self.config.initial_eps
            end_eps = self.config.dest_eps
            const_eps_epochs = 0.1 * self.config.epochs  # 10% of learning time
            eps_decay_epochs = 0.6 * self.config.epochs  # 60% of learning time

            if epoch < const_eps_epochs:
                return start_eps
            elif epoch < eps_decay_epochs:
                # Linear decay
                return start_eps - (epoch - const_eps_epochs) / \
                                   (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
            else:
                return end_eps

        print("Game variables: ", self.game.get_state().game_variables)
        s1 = self.preprocess(self.game.get_state().screen_buffer)

        # With probability eps make a random action.
        if self.config.explore_whole_episode:
            explore = self.episode_state['explore']
        else:
            eps = exploration_rate(epoch)
            explore = random() <= eps
        if explore:
            a = self.q_estimator.get_exploratory_action(s1)
        else:
            # Choose the best action according to the network.
            (a, uncert) = self.q_estimator.get_best_action(s1)
        reward = self.game.make_action(self.actions[a], self.config.frame_repeat)

        isterminal = self.game.is_episode_finished()
        if isterminal:
            eps = exploration_rate(epoch)
            self.episode_state['explore'] = random() <= eps

        s2 = self.preprocess(self.game.get_state().screen_buffer) if not isterminal else None

        self.q_estimator.learn_from_transition(s1, a, s2, isterminal, reward)


    # Creates and initializes ViZDoom environment.
    def initialize_vizdoom(self, config_file_path):
        print "Initializing doom..."
        game = DoomGame()
        game.load_config(config_file_path)
        game.set_window_visible(True)
        game.set_mode(Mode.PLAYER)
    	game.set_screen_format(ScreenFormat.GRAY8)
    	game.set_screen_resolution(ScreenResolution.RES_640X480)
        game.init()
        print "Doom initialized."
        return game


    def run(self):
        print "Starting the training!"
        time_start = time()
        train_results = []
        test_results = []
        certainties = []
        print "Initial test"
        test_scores, certainty = self.testing_epoch()
        test_results.extend(test_scores)
        for epoch in range(self.config.epochs):
            train_results.append(self.training_epoch(epoch))
            test_scores, certainty = self.testing_epoch()
            test_results.extend(test_scores)
            certainties.append(certainty)
            print "Saving the network weigths..."
            self.q_estimator.save()

            print "Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0)


        self.game.close()
        if self.config.play_agent:
            print "======================================"
            print "Training finished. It's time to watch!"
            self.q_estimator.load()
            self.play(self.q_estimator)
        else:
            print "======================================"
            print "Skipping watching"

        print "Training scores: ", train_results
        print "Test scores: ", test_results
        print "Certainties: ", certainties
        self.q_estimator.cleanup()
        return test_results

    def testing_epoch(self):
        print "\nTesting..."
        self.q_estimator.testing_mode()
        test_episode = []
        test_scores = []
        certaintiesSum = 0
        certaintiesCount = 0
        for test_episode in trange(self.config.test_episodes_per_epoch):
            self.game.new_episode()

            while not self.game.is_episode_finished():
                state = self.preprocess(self.game.get_state().screen_buffer)
                (best_action_index, certainty) = self.q_estimator.get_best_action(state)
                certaintiesSum += certainty
                certaintiesCount += 1
                self.game.make_action(self.actions[best_action_index], self.config.frame_repeat)
            r = self.game.get_total_reward()
            test_scores.append(r)
        test_scores = np.array(test_scores)
        print "Results: mean: %.1f±%.1f," % (
            test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(), "max: %.1f" % test_scores.max()
        print "Certainty: %.2f" % (certaintiesSum / certaintiesCount)
        return test_scores, certaintiesSum / certaintiesCount

    def training_epoch(self, epoch):
        print "\nEpoch %d\n-------" % (epoch + 1)
        train_episodes_finished = 0
        train_scores = []
        print "Training..."
        self.q_estimator.learning_mode()
        self.game.new_episode()
        for learning_step in trange(self.config.learning_steps_per_epoch):
            self.perform_learning_step(epoch)
            if self.game.is_episode_finished():
                score = self.game.get_total_reward()
                train_scores.append(score)
                self.game.new_episode()
                train_episodes_finished += 1
        print "%d training episodes played." % train_episodes_finished
        train_scores = np.array(train_scores)
        print "Results: mean: %.1f±%.1f," % (train_scores.mean(), train_scores.std()), \
            "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max()
        return train_scores.mean()


    def play(self, q_estimator):
        # Load the network's parameters from a file
        # Reinitialize the game with window visible
        self.game.set_window_visible(True)
        self.game.set_mode(Mode.ASYNC_PLAYER)
        self.game.init()
        episodes_to_watch = 10
        rewards_sum = 0
        for i in range(episodes_to_watch):
            self.game.new_episode()
            while not self.game.is_episode_finished():
                state = self.preprocess(self.game.get_state().screen_buffer)
                best_action_index = q_estimator.get_best_action(state)[0]

                # Instead of make_action(a, frame_repeat) in order to make the animation smooth
                self.game.set_action(self.actions[best_action_index])
                for i in range(self.config.frame_repeat):
                    self.game.advance_action()

            # Sleep between episodes
            sleep(1.0)
            score = self.game.get_total_reward()
            rewards_sum+=score
            print "Total score: ", score
        self.game.close()
        print "Average score: ", rewards_sum/episodes_to_watch
        return rewards_sum/episodes_to_watch
