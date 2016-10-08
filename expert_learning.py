#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from vizdoom import *
import itertools as it
import pickle
from random import sample, randint, random
from time import time, sleep
import numpy as np
import skimage.color, skimage.transform
from lasagne.init import HeUniform, Constant
from lasagne.layers import Conv2DLayer, InputLayer, DenseLayer, get_output, \
    get_all_params, get_all_param_values, set_all_param_values
from lasagne.nonlinearities import rectify
from lasagne.objectives import squared_error
from lasagne.updates import rmsprop
import theano
from theano import tensor
from tqdm import trange
from ReplayMemory import ReplayMemory

# Q-learning settings
learning_rate = 0.00025
# learning_rate = 0.0001
discount_factor = 0.99
epochs = 6
learning_steps_per_epoch = 500
replay_memory_size = 10000

# NN learning settings
batch_size = 64

# Training regime
test_episodes_per_epoch = 100

# Other parameters
frame_repeat = 12
resolution = (30, 45)

# Configuration file path
# config_file_path = "../../examples/config/my_basic.cfg"
#config_file_path = "../../examples/config/rocket_basic.cfg"
config_file_path = "../../examples/config/basic.cfg"

# Converts and downsamples the input image
def preprocess(img):
    img = img[0]
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    return img

def create_network(available_actions_count):
    # Create the input variables
    s1 = tensor.tensor4("States")
    a = tensor.vector("Actions", dtype="int32")
    q2 = tensor.vector("Next State's best Q-Value")
    r = tensor.vector("Rewards")
    isterminal = tensor.vector("IsTerminal", dtype="int8")

    # Create the input layer of the network.
    dqn = InputLayer(shape=[None, 1, resolution[0], resolution[1]], input_var=s1)

    # Add 2 convolutional layers with ReLu activation
    dqn = Conv2DLayer(dqn, num_filters=8, filter_size=[6, 6],
                      nonlinearity=rectify, W=HeUniform("relu"),
                      b=Constant(.1), stride=3)
    dqn = Conv2DLayer(dqn, num_filters=8, filter_size=[3, 3],
                      nonlinearity=rectify, W=HeUniform("relu"),
                      b=Constant(.1), stride=2)

    # Add a single fully-connected layer.
    dqn = DenseLayer(dqn, num_units=128, nonlinearity=rectify, W=HeUniform("relu"),
                     b=Constant(.1))

    # Add the output layer (also fully-connected).
    # (no nonlinearity as it is for approximating an arbitrary real function)
    dqn = DenseLayer(dqn, num_units=available_actions_count, nonlinearity=None)

    # Define the loss function
    q = get_output(dqn)
    # target differs from q only for the selected action. The following means:
    # target_Q(s,a) = r + gamma * max Q(s2,_) if isterminal else r
    target_q = tensor.set_subtensor(q[tensor.arange(q.shape[0]), a], r + discount_factor * (1 - isterminal) * q2)
    loss = squared_error(q, target_q).mean()

    # Update the parameters according to the computed gradient using RMSProp.
    params = get_all_params(dqn, trainable=True)
    updates = rmsprop(loss, params, learning_rate)

    # Compile the theano functions
    print "Compiling the network ..."
    function_learn = theano.function([s1, q2, a, r, isterminal], loss, updates=updates, name="learn_fn")
    function_get_q_values = theano.function([s1], q, name="eval_fn")
    function_get_best_action = theano.function([s1], tensor.argmax(q), name="test_fn")
    print "Network compiled."

    def simple_get_best_action(state):
        return function_get_best_action(state.reshape([1, 1, resolution[0], resolution[1]]))

    # Returns Theano objects for the net and functions.
    return dqn, function_learn, function_get_q_values, simple_get_best_action


def learn_from_transition(s1, a, s2, s2_isterminal, r):
    """ Learns from a single transition (making use of replay memory).
    s2 is ignored if s2_isterminal """

    # Remember the transition that was just experienced.
    memory.add_transition(s1, a, s2, s2_isterminal, r)

    # Get a random minibatch from the replay memory and learns from it.
    if memory.size > batch_size:
        s1, a, s2, isterminal, r = memory.get_sample(batch_size)
        q2 = np.max(get_q_values(s2), axis=1)
        # the value of q2 is ignored in learn if s2 is terminal
        learn(s1, q2, a, r, isterminal)


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
        reward = game.make_action(actions[a], frame_repeat)
    else:
        #Choose the best action according to the network.
        a = get_best_action(s1)
    playerAction = game.get_last_action()
    reward = game.make_action(playerAction, frame_repeat)
    # game.advance_action()
    # a = game.get_last_action()
    # reward = game.get_last_reward()

    isterminal = game.is_episode_finished()
    s2 = preprocess(game.get_state().image_buffer) if not isterminal else None

    learn_from_transition(s1, a, s2, isterminal, reward)


# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print "Initializing doom..."
    game = DoomGame()
    game.load_config(config_file_path)

    game.set_window_visible(False)
    # game.set_mode(Mode.SPECTATOR)
    game.set_mode(Mode.PLAYER)
    game.add_available_button(Button.MOVE_LEFT)
    game.add_available_button(Button.MOVE_RIGHT)
    # game.add_game_args("+freelook 1")
    game.init()
    print "Doom initialized."
    return game


# Create Doom instance
game = initialize_vizdoom(config_file_path)

# Action = which buttons are pressed
n = game.get_available_buttons_size()
actions = [list(a) for a in it.product([0, 1], repeat=n)]

# Create replay memory which will store the transitions
memory = ReplayMemory(capacity=replay_memory_size, resolution=resolution)

net, learn, get_q_values, get_best_action = create_network(len(actions))


print "Starting the training!"

time_start = time()
for epoch in range(epochs):
    print "\nEpoch %d\n-------" % (epoch + 1)
    train_episodes_finished = 0
    train_scores = []

    print "Training..."
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

    print "Results: mean: %.1f±%.1f," % (train_scores.mean(), train_scores.std()), \
        "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max()

    print "\nTesting..."
    test_episode = []
    test_scores = []
    for test_episode in trange(test_episodes_per_epoch):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess(game.get_state().image_buffer)
            best_action_index = get_best_action(state)

            game.make_action(actions[best_action_index], frame_repeat)
        r = game.get_total_reward()
        test_scores.append(r)

    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    test_scores = np.array(test_scores)
    print "Results: mean: %.1f±%.1f," % (
    test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(), "max: %.1f" % test_scores.max()

    print "Saving the network weigths..."
    pickle.dump(get_all_param_values(net), open('weights.dump', "w"))

    print "Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0)

game.close()
print "======================================"
print "Training finished. It's time to watch!"

# Load the network's parameters from a file
params = pickle.load(open('weights.dump', "r"))
set_all_param_values(net, params)

# Reinitialize the game with window visible
game.set_window_visible(True)
game.set_mode(Mode.ASYNC_PLAYER)
game.init()

episodes_to_watch = 10
for i in range(episodes_to_watch):
    game.new_episode()
    while not game.is_episode_finished():
        state = preprocess(game.get_state().image_buffer)
        best_action_index = get_best_action(state)

        # Instead of make_action(a, frame_repeat) in order to make the animation smooth
        game.set_action(actions[best_action_index])
        for i in range(frame_repeat):
            game.advance_action()

    # Sleep between episodes
    sleep(1.0)
    score = game.get_total_reward()
    print "Total score: ", score
