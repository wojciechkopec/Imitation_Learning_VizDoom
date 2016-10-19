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

from sources.replay_memory import ReplayMemory


def _create_convolution_layers(available_actions_count,resolution):
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
    return s1,a,q2,r,isterminal,dqn

class QEstimator:

    def __init__(self,available_actions_count,resolution,create_convolution_layers = None, dumpFileName ='out/weights.dump'):
        # Q-learning settings
        self.learning_rate = 0.00025
        # learning_rate = 0.0001
        self.discount_factor = 0.99
        self.replay_memory_size = 10000
        # NN learning settings
        self.batch_size = 64
        if create_convolution_layers == None:
            create_convolution_layers = lambda : _create_convolution_layers(available_actions_count, resolution)
        self.net, self.learn, self.get_q_values, self.get_best_action = self._create_network(available_actions_count,resolution,create_convolution_layers)
        self.memory = ReplayMemory(capacity=self.replay_memory_size,resolution=resolution)
        self.dumpFileName = dumpFileName

    def _create_network(self, available_actions_count,resolution,create_convolution_layers):
        s1,a,q2,r,isterminal,dqn = create_convolution_layers()

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
        target_q = tensor.set_subtensor(q[tensor.arange(q.shape[0]), a], r + self.discount_factor * (1 - isterminal) * q2)
        loss = squared_error(q, target_q).mean()

        # Update the parameters according to the computed gradient using RMSProp.
        params = get_all_params(dqn, trainable=True)
        updates = rmsprop(loss, params, self.learning_rate)

        # Compile the theano functions
        print "Compiling the network ..."
        function_learn = theano.function([s1, q2, a, r, isterminal], loss, updates=updates, name="learn_fn")
        function_get_q_values = theano.function([s1], q, name="eval_fn")
        function_get_best_action = theano.function([s1], tensor.argmax(q), name="test_fn")
        print "Network compiled."

        def simple_get_best_action(state):
            return function_get_best_action(state.reshape([1, 1, resolution[0], resolution[1]])), 1.0

        # Returns Theano objects for the net and functions.
        return dqn, function_learn, function_get_q_values, simple_get_best_action


    def learn_from_transition(self,s1, a, s2, s2_isterminal, r):
        """ Learns from a single transition (making use of replay memory).
        s2 is ignored if s2_isterminal """

        # Remember the transition that was just experienced.
        self.memory.add_transition(s1, a, s2, s2_isterminal, r)

        # Get a random minibatch from the replay memory and learns from it.
        if self.memory.size > self.batch_size:
            s1, a, s2, isterminal, r = self.memory.get_sample(self.batch_size)
            q2 = np.max(self.get_q_values(s2), axis=1)
            # the value of q2 is ignored in learn if s2 is terminal
            self.learn(s1, q2, a, r, isterminal)

    def save(self):
        pickle.dump(get_all_param_values(self.net), open(self.dumpFileName, "w"))

    def load(self):
        params = pickle.load(open(self.dumpFileName, "r"))
        set_all_param_values(self.net, params)


    def learning_mode(self):
        pass

    def testing_mode(self):
        pass