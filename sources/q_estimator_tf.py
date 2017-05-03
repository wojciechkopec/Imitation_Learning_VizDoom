import pickle

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

from numpy.random import randint
from sources.replay_memory import ReplayMemory
from sources.replay_memory import TransitionStore


def _create_convolution_layers(available_actions_count, resolution):
    tf.reset_default_graph()
    session = tf.Session()
    # Create the input variables
    s1 = tf.placeholder(tf.float32, [None] + list(resolution) + [1], name="State")
    a = tf.placeholder(tf.int32, [None], name="Action")
    q2 = tf.placeholder(tf.float32, [None, available_actions_count], name="TargetQ")

    # Add 2 convolutional layers with ReLu activation
    conv1 = tf.contrib.layers.convolution2d(s1, num_outputs=8, kernel_size=[6, 6], stride=[3, 3],
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            biases_initializer=tf.constant_initializer(0.1))
    conv2 = tf.contrib.layers.convolution2d(conv1, num_outputs=8, kernel_size=[3, 3], stride=[2, 2],
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            biases_initializer=tf.constant_initializer(0.1))
    conv2_flat = tf.contrib.layers.flatten(conv2)

    return (session, s1, a, q2, conv2_flat)


class QEstimator:
    def __init__(self, available_actions_count, resolution, create_convolution_layers=None, replay_memory_size=10000,
                 store_trajectory=True, dump_file_name='out/weights.dump'):
        # Q-learning settings
        self.learning_rate = 0.00025
        # learning_rate = 0.0001
        self.discount_factor = 0.99
        self.store_trajectory = store_trajectory
        self.transition_store = TransitionStore(self.discount_factor)
        self.replay_memory_size = replay_memory_size
        # NN learning settings
        self.batch_size = 64
        if create_convolution_layers == None:
            create_convolution_layers = lambda: _create_convolution_layers(available_actions_count, resolution)
        self.session, self.learn, self.get_q_values, self.get_best_action = self._create_network(available_actions_count,
                                                                                             resolution,
                                                                                             create_convolution_layers)
        self.memory = ReplayMemory(capacity=self.replay_memory_size, resolution=resolution)
        self.dump_file_name = dump_file_name

    def _create_network(self, available_actions_count, resolution, create_convolution_layers):
        session, s1, a, q2, dqn = create_convolution_layers()
        fc1 = tf.contrib.layers.fully_connected(dqn, num_outputs=128, activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                biases_initializer=tf.constant_initializer(0.1))

        q = tf.contrib.layers.fully_connected(fc1, num_outputs=available_actions_count, activation_fn=None,
                                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                                              biases_initializer=tf.constant_initializer(0.1))
        best_a = tf.argmax(q, 1)

        loss = tf.contrib.losses.mean_squared_error(q, q2)

        optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        # Update the parameters according to the computed gradient using RMSProp.
        train_step = optimizer.minimize(loss)

        def function_learn(s, target):
            feed_dict = {s1: s, q2: target}
            l, _ = session.run([loss, train_step], feed_dict=feed_dict)
            return l

        def function_get_q_values(state):
            return session.run(q, feed_dict={s1: state})

        def function_get_best_action(state):
            return (session.run(best_a, feed_dict={s1: state})[0], 1)

        def function_simple_get_best_action(state):
            return function_get_best_action(state.reshape([1, resolution[0], resolution[1], 1]))

        init = tf.global_variables_initializer()
        session.run(init)
        return session, function_learn, function_get_q_values, function_simple_get_best_action

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
            s1, a, s2, isterminal, r, q2 = self.memory.get_sample(self.batch_size)
            q2 = np.fmax(q2, np.max(self.get_q_values(s2), axis=1))
            # the value of q2 is ignored in learn if s2 is terminal
            target_q = self.get_q_values(s1)
            # q2 = np.max(self.get_q_values(s2), axis=1)

            # target differs from q only for the selected action. The following means:
            # target_Q(s,a) = r + gamma * max Q(s2,_) if isterminal else r
            target_q[np.arange(target_q.shape[0]), a] = r + self.discount_factor * (1 - isterminal) * q2
            self.learn(s1, target_q)

    def get_exploratory_action(self, state):
        return randint(0, self.available_actions_count - 1)


    def save(self):
        pass
    def load(self):
        pass


    def learning_mode(self):
        pass

    def testing_mode(self):
        pass

    def cleanup(self):
        pass