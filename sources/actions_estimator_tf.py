import pickle
import random
import sys
import termios
import threading
import time
import tty

import tensorflow as tf
from numpy.random import randint
from tqdm import trange

from experiments_runner import preprocessIMG
from sources.key_monitor import KeyMonitor
from sources.replay_memory import PerActionReplayMemory
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


class ActionsEstimator:
    def __init__(self, actions, resolution, expert_config, create_convolution_layers=None, replay_memory_size=100000,
                 store_trajectory=True, dump_file_name='out/weights.dump'):
        self.expert_config = expert_config
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
        self.dropout = 0.5
        # NN learning settings

        self.batch_size = 64
        if create_convolution_layers is None:
            create_convolution_layers = lambda: _create_convolution_layers(len(actions), resolution)
        self.session, self.learn, self.get_q_values, self.get_best_action, self.learn_actions = self._create_network(
            len(actions),
            resolution,
            create_convolution_layers)
        self.memory = PerActionReplayMemory(10000, resolution,
                                            len(actions),
                                            [0, 1, 1, 1, 1, 1, 1, 1, 1])
        # self.memory = ReplayMemory(len(expert_config.feed_memory)* 2, resolution)
        self.dump_file_name = dump_file_name
        self.uncertain_actions_count = 0
        self.certain_actions_count = 0
        self.expert_mode = False
        self.testing= True
        self.last_frames_count = 0
        self.store_expert_trajectories(expert_config, self.memory)
        self.learn_all()

        self.keys_thread = threading.Thread(target=KeyMonitor(['p', '.', ','], lambda key,press: self.__toggle_user_input(key) if press else False).run)
        self.keys_thread.daemon = True
        self.keys_thread.start()
        self.framerate = 20

    def _create_network(self, available_actions_count, resolution, create_convolution_layers):
        session, s1, a, q2, dqn = create_convolution_layers()
        keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)
        dqn = tf.nn.dropout(dqn, keep_prob)
        fc1 = tf.contrib.layers.fully_connected(dqn, num_outputs=256, activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                biases_initializer=tf.constant_initializer(0.1))

        fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=128, activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                biases_initializer=tf.constant_initializer(0.1))
        fc2 = tf.nn.dropout(fc2, keep_prob)
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
            feed_dict = {s1: s, q2: target, keep_prob: self.dropout}
            l, _ = session.run([loss, train_step], feed_dict=feed_dict)
            return l

        def function_get_q_values(state):
            return session.run(q, feed_dict={s1: state, keep_prob: 1})

        def function_get_best_action(state):
            a, sftmx = session.run([best_a, sm], feed_dict={s1: state, keep_prob: 1})
            a = a[0]
            uncertainty = 1 - sftmx[0][a]

            if not self.testing:
                if uncertainty > 0.4:
                    self.uncertain_actions_count += 1
                else:
                    self.uncertain_actions_count = 0

                switch_condition = False
                if self.expert_config.switch_expert_mode == 'random':
                    switch_condition = randint(100) >= 95 and not self.expert_mode
                elif self.expert_config.switch_expert_mode == 'uncertainty':
                    switch_condition = self.uncertain_actions_count > 1 and not self.expert_mode

                if switch_condition:
                    print ("Quering expert!")
                    self.expert_mode = True
                    self.uncertain_actions_count = 0

                if self.expert_mode:
                    print "Model would do " + str(self.actions[a])
                    expert_a = self.get_expert_action()
                    print "Expert would do " + str(self.actions[expert_a])

                    if expert_a == a:
                        self.certain_actions_count += 1
                        print (str(self.certain_actions_count) + " actions matching")
                    else:
                        self.certain_actions_count = 0

                    a = expert_a

                    exit_condition = False
                    if self.expert_config.switch_expert_mode == 'random':
                        exit_condition = randint(100) >= 95
                    elif self.expert_config.switch_expert_mode == 'uncertainty':
                        exit_condition = self.certain_actions_count > 20

                    if exit_condition > 20:
                        self.expert_mode = False
                        self.uncertain_actions_count = 0
                        self.certain_actions_count = 0
                else:
                    time.sleep(1.0/self.framerate)

            return (a, uncertainty)

        def function_simple_get_best_action(state):
            return function_get_best_action(state.reshape([1, resolution[0], resolution[1], 1]))

        def function_learn_actions(states, actions):
            feed_dict = {s1: states, a: actions, keep_prob: self.dropout}
            e, acc, _ = session.run([cross_entropy, accuracy, actions_train_step], feed_dict=feed_dict)
            return e, acc

        init = tf.global_variables_initializer()
        session.run(init)
        return session, function_learn, function_get_q_values, function_simple_get_best_action, function_learn_actions

    def learn_from_transition(self, s1, a, s2, s2_isterminal, r, q2=float("-inf")):
        if self.expert_mode:
            for i in range(5):
                self.memory.add_transition(s1, a, s2, s2_isterminal, r, q2)
        if s2_isterminal:
            self.expert_mode = False

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
        self.expert_mode = False
        self.learn_all()


    def cleanup(self):
        self.session.close()

    def get_expert_action(self):
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

        if move == 'i':
            return 1
        if move == 'u':
            return 5
        if move == 'o':
            return 3
        return 0

    def store_expert_trajectories(self, expert_config, memory):
        print "Decoding expert trajectories"
        random.shuffle(expert_config.feed_memory)
        frames_count = 0
        for filepath in expert_config.feed_memory:
            print "loading " + filepath
            with open(filepath, 'rb') as f:
                trajectories =  pickle.load(f)

                for t in trange(len(trajectories)):
                    transition = trajectories[t]
                    (s1, action, s2, r, isterminal) = transition
                    s1 = preprocessIMG(s1, self.resolution)
                    if self.actions.index(action) != 0:
                        memory.add_transition(s1, self.actions.index(action), s1, isterminal, r, -1)
                        frames_count += 1
                    if frames_count >= expert_config.frames_limit:
                        break
                del trajectories
                if frames_count >= expert_config.frames_limit:
                    return


    def learn_all(self, iterations = -1):
        if self.last_frames_count == self.memory.size:
            print "Memory unchanged, skipping learning"
            return
        print "Learning expert trajectories (" + str(self.memory.size) + " frames)"
        batch_size = 64
        acc_above_threshold = 0
        init = tf.global_variables_initializer()
        self.session.run(init)
        if iterations == -1:
            iterations = int(self.memory.size)
        for it in trange(iterations):
            s1, a, _, _, _, _ = self.memory.get_sample(batch_size)
            e, acc = self.learn_actions(s1, a)
            if it % 50 == 0:
                print e, acc
            if acc > 0.95:
                acc_above_threshold+=1
            else:
                acc_above_threshold = 0
            if acc_above_threshold >10:
                break
        self.last_frames_count = self.memory.size

    def __toggle_user_input(self, character):
        if character == 'p':
            if self.expert_config.switch_expert_mode != 'expert_call':
                return True
            self.expert_mode = not self.expert_mode
            print ("Expert toggled: " + str(self.expert_mode))
        elif character == '.':
            self.framerate+=5
            print ("Framerate: " + str(self.framerate))
        elif character == ',':
            self.framerate -= 5
            print ("Framerate: " + str(self.framerate))
        return True
