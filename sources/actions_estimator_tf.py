import numpy as np
import tensorflow as tf
from numpy.random import randint
import sys, tty, termios
from tqdm import trange

from sources.replay_memory import ReplayMemory, PerActionReplayMemory
from sources.replay_memory import TransitionStore
from experiments_runner import preprocessIMG


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


class ActionsEstimator:
    def __init__(self, actions, resolution, expert_config, create_convolution_layers=None, replay_memory_size=100000,
                 store_trajectory=True, dump_file_name='out/weights.dump'):
        # Q-learning settings
        self.learning_rate = 0.00025
        # learning_rate = 0.0001
        self.discount_factor = 0.99
        self.store_trajectory = store_trajectory
        self.transition_store = TransitionStore(self.discount_factor)
        self.replay_memory_size = replay_memory_size
        self.available_actions_count = len(actions)
        self.actions = actions
        self.resolution = resolution
        # NN learning settings

        self.batch_size = 64
        if create_convolution_layers is None:
            create_convolution_layers = lambda: _create_convolution_layers(len(actions), resolution)
        self.session, self.learn, self.get_q_values, self.get_best_action, self.learn_actions = self._create_network(
            len(actions),
            resolution,
            create_convolution_layers)
        self.memory = PerActionReplayMemory(len(expert_config.feed_memory)* 2, resolution,
                                            len(actions),
                                            [0, 1, 1, 1, 1, 1, 1, 1, 1])
        # self.memory = ReplayMemory(len(expert_config.feed_memory)* 2, resolution)
        self.dump_file_name = dump_file_name
        self.uncertain_actions_count = 0
        self.certain_actions_count = 0
        self.expert_mode = False
        self.testing= True
        self.store_expert_trajectories(expert_config, self.memory)
        self.learn_all()

    def _create_network(self, available_actions_count, resolution, create_convolution_layers):
        session, s1, a, q2, dqn = create_convolution_layers()
        fc1 = tf.contrib.layers.fully_connected(dqn, num_outputs=256, activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                biases_initializer=tf.constant_initializer(0.1))

        fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=128, activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                biases_initializer=tf.constant_initializer(0.1))

        q = tf.contrib.layers.fully_connected(fc2, num_outputs=available_actions_count, activation_fn=None,
                                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                                              biases_initializer=tf.constant_initializer(0.1))
        best_a = tf.argmax(q, 1)
        sm = tf.nn.softmax(q)
        chosen_action =tf.one_hot(a, available_actions_count)
        softmax = tf.nn.softmax_cross_entropy_with_logits(labels=chosen_action, logits=q)
        cross_entropy = tf.reduce_mean(softmax)
        sm = tf.nn.softmax(q)

        actions_optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        # Update the parameters according to the computed gradient using RMSProp.
        actions_train_step = actions_optimizer.minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(q, 1), tf.argmax(chosen_action, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

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
            a, sftmx = session.run([best_a, sm], feed_dict={s1: state})
            a = a[0]
            uncertainty = 1 - sftmx[0][a]

            if not self.testing:
                if uncertainty > 0.4:
                    self.uncertain_actions_count += 1
                else:
                    self.uncertain_actions_count = 0

                if self.uncertain_actions_count > 1 and not self.expert_mode:
                    print ("Quering expert!")
                    self.expert_mode = True
                    self.uncertain_actions_count = 0

                if self.expert_mode:
                    expert_a = a #self.get_action()

                    if expert_a == a:
                        self.certain_actions_count += 1
                        print (str(self.certain_actions_count) + " actions matching")
                    else:
                        self.certain_actions_count = 0

                    a = expert_a
                    if self.certain_actions_count > 20:
                        self.expert_mode = False
                        self.uncertain_actions_count = 0
                        self.certain_actions_count = 0

            return (a, uncertainty)

        def function_simple_get_best_action(state):
            return function_get_best_action(state.reshape([1, resolution[0], resolution[1], 1]))

        def function_learn_actions(states, actions):
            feed_dict = {s1: states, a: actions}
            e, acc, _ = session.run([cross_entropy, accuracy, actions_train_step], feed_dict=feed_dict)
            return e, acc

        init = tf.global_variables_initializer()
        session.run(init)
        return session, function_learn, function_get_q_values, function_simple_get_best_action, function_learn_actions

    def learn_from_transition(self, s1, a, s2, s2_isterminal, r, q2=float("-inf")):
        if self.expert_mode:
            self.memory.add_transition(s1, a, s2, s2_isterminal, r, q2)
        if s2_isterminal:
            self.expert_mode = False
        # Get a random minibatch from the replay memory and learns from it.
        # if self.memory.size > self.batch_size:
        #     s1, a, s2, isterminal, r, q2 = self.memory.get_sample(self.batch_size)
        #     self.learn_actions(s1, a)

    def get_exploratory_action(self, state):
        return randint(0, self.available_actions_count - 1)

    def save(self):
        pass

    def load(self):
        pass

    def learning_mode(self):
        self.testing = False

    def testing_mode(self):
        self.testing = True
        self.learn_all()

    def cleanup(self):
        pass

    def get_action(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            move = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        if move == 'j':
            return 4
        if move == 'l':
            return 2
        if move == 'a':
            return 1
        return 0

    def store_expert_trajectories(self, expert_config, memory):
        print "Decoding expert trajectories"
        for t in trange(len(expert_config.feed_memory)):
            transition = expert_config.feed_memory[t]
            (s1, action, s2, r, isterminal) = transition
            s1 = preprocessIMG(s1, self.resolution)
            if self.actions.index(action) != 0:
                memory.add_transition(s1, self.actions.index(action), s1, isterminal, r, -1)

    def learn_all(self):
        print "Learning expert trajectories (" + str(self.memory.size) + " frames)"
        batch_size = 64
        for it in trange(int(self.memory.size )):
            s1, a, _, _, _, _ = self.memory.get_sample(batch_size)
            e, acc = self.learn_actions(s1, a)
            if it % 50 == 0:
                print e, acc
            if acc > 0.95:
                break
