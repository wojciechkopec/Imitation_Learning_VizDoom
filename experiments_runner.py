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

from lib.vizdoom import *

class ExperimentsRunner:

    def __init__(self, config, agent):
        self.config = config
            # Create Doom instance
        self.game = self.initialize_vizdoom(config.config_file_path)

    # Action = which buttons are pressed
        self.n = self.game.get_available_buttons_size()
        self.actions = [list(a) for a in it.product([0, 1], repeat=self.n)]
        self.qEstimator = agent(self.actions, config)

    # Converts and downsamples the input image
    def preprocess(self, img):
        img = img[0]
        img = skimage.transform.resize(img, self.config.resolution)
        img = img.astype(np.float32)
        return img

    def perform_learning_step(self, epoch):
        """ Makes an action according to eps-greedy policy, observes the result
        (next state, reward) and learns from the transition"""

        def exploration_rate(epoch):
            """# Define exploration rate change over time"""
            start_eps = 1.0
            end_eps = 0.1
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

        s1 = self.preprocess(self.game.get_state().screen_buffer)

        # With probability eps make a random action.
        eps = exploration_rate(epoch)
        if random() <= eps:
            a = randint(0, len(self.actions) - 1)
        else:
            # Choose the best action according to the network.
            a = self.qEstimator.get_best_action(s1)[0]
        reward = self.game.make_action(self.actions[a], self.config.frame_repeat)

        isterminal = self.game.is_episode_finished()
        s2 = self.preprocess(self.game.get_state().screen_buffer) if not isterminal else None

        self.qEstimator.learn_from_transition(s1, a, s2, isterminal, reward)


    # Creates and initializes ViZDoom environment.
    def initialize_vizdoom(self, config_file_path):
        print "Initializing doom..."
        game = DoomGame()
        game.load_config(config_file_path)
        game.set_window_visible(False)
        game.set_mode(Mode.PLAYER)
        game.init()
        print "Doom initialized."
        return game


    def run(self):
        print "Starting the training!"

        time_start = time()
        train_results = []
        test_results = []
        certainties = []
        for epoch in range(self.config.epochs):
            print "\nEpoch %d\n-------" % (epoch + 1)
            train_episodes_finished = 0
            train_scores = []

            print "Training..."
            self.qEstimator.learning_mode()
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
            train_results.append(train_scores.mean())
            print "Results: mean: %.1f±%.1f," % (train_scores.mean(), train_scores.std()), \
                "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max()

            print "\nTesting..."
            self.qEstimator.testing_mode()
            test_episode = []
            test_scores = []
            certaintiesSum = 0
            certaintiesCount = 0
            for test_episode in trange(self.config.test_episodes_per_epoch):
                self.game.new_episode()

                while not self.game.is_episode_finished():
                    state = self.preprocess(self.game.get_state().screen_buffer)
                    (best_action_index,certainty) = self.qEstimator.get_best_action(state)
                    certaintiesSum += certainty
                    certaintiesCount += 1
                    self.game.make_action(self.actions[best_action_index], self.config.frame_repeat)
                r = self.game.get_total_reward()
                test_scores.append(r)

            test_scores = np.array(test_scores)
            test_results.append(test_scores.mean())
            certainties.append(certaintiesSum/certaintiesCount)
            print "Results: mean: %.1f±%.1f," % (
            test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(), "max: %.1f" % test_scores.max()
            print "Certainty: %.2f" % (certaintiesSum/certaintiesCount)
            print "Saving the network weigths..."
            self.qEstimator.save()

            print "Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0)

        self.game.close()
        if self.config.playAgent:
            print "======================================"
            print "Training finished. It's time to watch!"

            # Load the network's parameters from a file
            self.qEstimator.load()

            # Reinitialize the game with window visible
            self.game.set_window_visible(True)
            self.game.set_mode(Mode.ASYNC_PLAYER)
            self.game.init()

            episodes_to_watch = 10
            for i in range(episodes_to_watch):
                self.game.new_episode()
                while not self.game.is_episode_finished():
                    state = self.preprocess(self.game.get_state().screen_buffer)
                    best_action_index = self.qEstimator.get_best_action(state)[0]

                    # Instead of make_action(a, frame_repeat) in order to make the animation smooth
                    self.game.set_action(self.actions[best_action_index])
                    for i in range(self.config.frame_repeat):
                        self.game.advance_action()

                # Sleep between episodes
                sleep(1.0)
                score = self.game.get_total_reward()
                print "Total score: ", score
        else:
            print "======================================"
            print "Skipping watching"

        print "Training scores: ", train_results
        print "Test scores: ", test_results
        print "Certainties: ", certainties