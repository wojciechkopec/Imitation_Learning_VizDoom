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
from sources.multi_q_estimator import MultiQEstimator
from sources.q_estimator import QEstimator


epochs = 20
learning_steps_per_epoch = 2000

# Training regime
test_episodes_per_epoch = 100

# Other parameters
frame_repeat = 12
resolution = (30, 45)

# Configuration file path
config_file_path = "./config/basic.cfg"

def createEstimator():
    K = 5
    includeSampleProbability = 0.9
    return MultiQEstimator(len(actions),resolution,K,includeSampleProbability,False)
    # return QEstimator(len(actions),resolution)


# Converts and downsamples the input image
def preprocess(img):
    img = img[0]
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    return img

def perform_learning_step(epoch):
    """ Makes an action according to eps-greedy policy, observes the result
    (next state, reward) and learns from the transition"""

    def exploration_rate(epoch):
        """# Define exploration rate change over time"""
        start_eps = 1.0
        end_eps = 0.1
        const_eps_epochs = 0.1 * epochs  # 10% of learning time
        eps_decay_epochs = 0.6 * epochs  # 60% of learning time

        if epoch < const_eps_epochs:
            return start_eps
        elif epoch < eps_decay_epochs:
            # Linear decay
            return start_eps - (epoch - const_eps_epochs) / \
                               (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
        else:
            return end_eps

    s1 = preprocess(game.get_state().image_buffer)

    # With probability eps make a random action.
    eps = exploration_rate(epoch)
    if random() <= eps:
        a = randint(0, len(actions) - 1)
    else:
        # Choose the best action according to the network.
        a = qEstimator.get_best_action(s1)[0]
    reward = game.make_action(actions[a], frame_repeat)

    isterminal = game.is_episode_finished()
    s2 = preprocess(game.get_state().image_buffer) if not isterminal else None

    qEstimator.learn_from_transition(s1, a, s2, isterminal, reward)


# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print "Initializing doom..."
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.init()
    print "Doom initialized."
    return game


# Create Doom instance
game = initialize_vizdoom(config_file_path)

# Action = which buttons are pressed
n = game.get_available_buttons_size()
actions = [list(a) for a in it.product([0, 1], repeat=n)]

# Create replay memory which will store the transitions
qEstimator = createEstimator()
# qEstimator = QEstimator(len(actions),resolution)
#net, learn, get_q_values, get_best_action = create_network(len(actions))


print "Starting the training!"

time_start = time()
train_results = []
test_results = []
certainties = []
for epoch in range(epochs):
    print "\nEpoch %d\n-------" % (epoch + 1)
    train_episodes_finished = 0
    train_scores = []

    print "Training..."
    qEstimator.learning_mode()
    game.new_episode()
    for learning_step in trange(learning_steps_per_epoch):
        perform_learning_step(epoch)
        if game.is_episode_finished():
            score = game.get_total_reward()
            train_scores.append(score)
            game.new_episode()
            train_episodes_finished += 1

    print "%d training episodes played." % train_episodes_finished

    train_scores = np.array(train_scores)
    train_results.append(train_scores.mean())
    print "Results: mean: %.1f±%.1f," % (train_scores.mean(), train_scores.std()), \
        "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max()

    print "\nTesting..."
    qEstimator.testing_mode()
    test_episode = []
    test_scores = []
    certaintiesSum = 0
    certaintiesCount = 0
    for test_episode in trange(test_episodes_per_epoch):
        game.new_episode()

        while not game.is_episode_finished():
            state = preprocess(game.get_state().image_buffer)
            (best_action_index,certainty) = qEstimator.get_best_action(state)
            certaintiesSum += certainty
            certaintiesCount += 1
            game.make_action(actions[best_action_index], frame_repeat)
        r = game.get_total_reward()
        test_scores.append(r)

    test_scores = np.array(test_scores)
    test_results.append(test_scores.mean())
    certainties.append(certaintiesSum/certaintiesCount)
    print "Results: mean: %.1f±%.1f," % (
    test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(), "max: %.1f" % test_scores.max()
    print "Certainty: %.2f" % (certaintiesSum/certaintiesCount)
    print "Saving the network weigths..."
    qEstimator.save()

    print "Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0)

game.close()
print "======================================"
print "Training finished. It's time to watch!"

# Load the network's parameters from a file
qEstimator.load()

# Reinitialize the game with window visible
game.set_window_visible(True)
game.set_mode(Mode.ASYNC_PLAYER)
game.init()

episodes_to_watch = 10
for i in range(episodes_to_watch):
    game.new_episode()
    while not game.is_episode_finished():
        state = preprocess(game.get_state().image_buffer)
        best_action_index = qEstimator.get_best_action(state)[0]

        # Instead of make_action(a, frame_repeat) in order to make the animation smooth
        game.set_action(actions[best_action_index])
        for i in range(frame_repeat):
            game.advance_action()

    # Sleep between episodes
    sleep(1.0)
    score = game.get_total_reward()
    print "Total score: ", score

print "Training scores: ", train_results
print "Test scores: ", test_results
print "Certainties: ", certainties
