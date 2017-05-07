import pickle
from random import randint, random

from theano import tensor
from lasagne.init import HeUniform, Constant
from lasagne.layers import Conv2DLayer, InputLayer, DenseLayer, get_output, \
    get_all_params, get_all_param_values, set_all_param_values
from lasagne.nonlinearities import rectify
from lasagne.objectives import squared_error
from lasagne.updates import rmsprop
import theano
import numpy as np
import tensorflow as tf

from sources.replay_memory import ReplayMemory
from sources.replay_memory import TransitionStore


class QEstimator:
    def __init__(self, available_actions_count, resolution, subnets=5, replay_memory_size=10000,
                 store_trajectory=True, dump_file_name='out/weights.dump'):
        # Q-learning settings
        self.learning_rate = 0.0001
        self.available_actions_count = available_actions_count
        # learning_rate = 0.0001
        self.discount_factor = 0.99
        self.store_trajectory = store_trajectory
        self.transition_store = TransitionStore(self.discount_factor)
        self.replay_memory_size = replay_memory_size
        # NN learning settings
        self.batch_size = 256
        self.subnets = subnets
        self.session = tf.Session()
        self.learn, self.get_q_values, self.function_get_best_action = self._create_network(available_actions_count,
                                                                                             resolution)
        init = tf.initialize_all_variables()
        self.session.run(init)
        self.memory = ReplayMemory(capacity=self.replay_memory_size, resolution=resolution)
        self.dump_file_name = dump_file_name
        self.active_net = 0

    def _create_network(self, available_actions_count, resolution):
        # Create the input variables
        s1_ = tf.placeholder(tf.float32, [None] + list(resolution) + [1], name="State")
        a_ = tf.placeholder(tf.int32, [None], name="Action")
        target_q_ = tf.placeholder(tf.float32, [None, self.subnets, available_actions_count], name="TargetQ")

        # Add 2 convolutional layers with ReLu activation
        conv1 = tf.contrib.layers.convolution2d(s1_, num_outputs=8, kernel_size=[6, 6], stride=[3, 3],
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                biases_initializer=tf.constant_initializer(0.1))
        conv2 = tf.contrib.layers.convolution2d(conv1, num_outputs=8, kernel_size=[3, 3], stride=[2, 2],
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                biases_initializer=tf.constant_initializer(0.1))

        conv2_flat = tf.contrib.layers.flatten(conv2)

        first_dense_nodes = 128
        qs = []
        for net in range(self.subnets):
            fc1 = tf.contrib.layers.fully_connected(conv2_flat, num_outputs=first_dense_nodes, activation_fn=tf.nn.relu,
                                                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                    biases_initializer=tf.constant_initializer(0.1))

            q = tf.contrib.layers.fully_connected(fc1, num_outputs=available_actions_count, activation_fn=None,
                                                  weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                  biases_initializer=tf.constant_initializer(0.1))
            qs.append(q)

        qs = tf.stack(qs, axis =1)

        best_a = tf.argmax(qs, 2)

        loss = tf.contrib.losses.mean_squared_error(qs, target_q_)

        optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        # Update the parameters according to the computed gradient using RMSProp.
        train_step = optimizer.minimize(loss)

        def function_learn(s1, target_q):
            feed_dict = {s1_: s1, target_q_: target_q}
            l, _ = self.session.run([loss, train_step], feed_dict=feed_dict)
            return l

        def function_get_q_values(state):
            return self.session.run(qs, feed_dict={s1_: state})

        def function_get_best_action(state):
            return self.session.run(best_a, feed_dict={s1_: state})

        def function_simple_get_best_action(state):
            return function_get_best_action(state.reshape([1, resolution[0], resolution[1], 1]))[0], 1

        return function_learn, function_get_q_values, function_simple_get_best_action

    def learn_from_transition(self, s1, a, s2, s2_isterminal, r, q2=float("-inf")):
        """ Learns from a single transition (making use of replay memory).
        s2 is ignored if s2_isterminal """
        if self.store_trajectory:
            self.transition_store.store(s1, a, s2, s2_isterminal, r)
            if s2_isterminal:
                for t_s1, t_a, t_s2, t_s2_isterminal, t_r, t_q2 in self.transition_store.get_trajectory():
                    self.memory.add_transition(t_s1, t_a, t_s2, t_s2_isterminal, t_r, t_q2)

        # Remember the transition
        else:
            self.memory.add_transition(s1, a, s2, s2_isterminal, r, q2)

        # Get a random minibatch from the replay memory and learns from it.
        if self.memory.size > self.batch_size:
            s1, a, s2, isterminal, r, q2, mask = self.memory.get_sample(self.batch_size)
            # q2 = np.fmax(q2, np.max(self.get_q_values(s2), axis=1))
            # the value of q2 is ignored in learn if s2 is terminal
            target_q = self.get_q_values(s1)
            q2 = np.fmax(np.stack([q2 for _ in range(self.subnets)], axis=1), np.max(self.get_q_values(s2), axis=2))

            # target differs from q only for the selected action. The following means:
            # target_Q(s,a) = r + gamma * max Q(s2,_) if isterminal else r

            for ind in range(self.batch_size):
                target_q[ind, mask[ind], a[ind]] = r[ind] + self.discount_factor * (1 - isterminal[ind]) * q2[ind,mask[ind]]

            #indices = np.arange(target_q.shape[0])
            #for net in mask[0]:
            #target_q[indices, mask, a] = r + self.discount_factor * (1 - isterminal) * q2[:, mask]
            self.learn(s1, target_q)
            if s2_isterminal:
                self.active_net = randint(0, self.subnets - 1)

    def get_best_action(self, state):
        if self.learning:
            return self.get_best_action_learning(state)
        else:
            return self.get_best_action_testing(state)

    def get_best_action_learning(self, state):
        (dist, currentEstAction) = self._get_best_actions_dist(state)
        return currentEstAction, dist[currentEstAction] / self.subnets

    def get_best_action_testing(self, state):
        (dist, _) = self._get_best_actions_dist(state)
        chosenAction = np.argmax(dist)
        return chosenAction, dist[chosenAction] / self.subnets

    def _get_best_actions_dist(self, state):
        dist = np.zeros(self.available_actions_count)
        actions = self.function_get_best_action(state)[0]

        for net in range(self.subnets):
            action = actions[net]
            dist[action] += 1
            if net == self.active_net:
                currentEstAction = action
        return dist, currentEstAction

    def cleanup(self):
        self.session.close()

    def save(self):
        pass

    def load(self):
        pass

    def learning_mode(self):
        self.learning = True

    def testing_mode(self):
        self.learning = False