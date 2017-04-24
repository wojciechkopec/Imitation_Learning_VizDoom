'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import numpy as np
import time
import sys

from random import random
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/")

def run(dropout, test_dropout, calls, groups=False, show_charts=True):
    tf.reset_default_graph()

    # Parameters
    learning_rate = 0.0005
    training_iters = 1000000
    batch_size = 128
    display_step = 100

    # Network Parameters
    n_input = 784  # MNIST data input (img shape: 28*28)


    # tf Graph input
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, 1])
    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)


    # Create some wrappers for simplicity
    def conv2d(x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)


    def maxpool2d(x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')


    # Create model
    def conv_net(x, dropout):
        # Reshape input picture
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        conv1 = tf.contrib.layers.convolution2d(x, num_outputs=8, kernel_size=[2, 2], stride=[2, 2],
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                biases_initializer=tf.constant_initializer(0.1))
        conv2 = tf.contrib.layers.convolution2d(conv1, num_outputs=8, kernel_size=[2, 2], stride=[2, 2],
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                biases_initializer=tf.constant_initializer(0.1))
        conv2_flat = tf.contrib.layers.flatten(conv2)
        conv2_flat = tf.nn.dropout(conv2_flat, dropout)
        fc1 = tf.contrib.layers.fully_connected(conv2_flat, num_outputs=512, activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                biases_initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.dropout(fc1, dropout)

        out = tf.contrib.layers.fully_connected(fc1, num_outputs=1, activation_fn=None,
                                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                biases_initializer=tf.constant_initializer(0.1))
        return out


    # Construct model
    pred = conv_net(x, keep_prob)

    # Define loss and optimizer
    cost = tf.contrib.losses.mean_squared_error(pred, y)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.6).minimize(cost)

    # Evaluate model
    rounded = tf.round(pred * 10)
    correct_pred = tf.equal(rounded, y * 10)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.initialize_all_variables()


    def test(X, Y):

        result = np.zeros((calls, len(Y)), dtype=float)
        labels = Y  # np.argmax(Y, axis=1)
        Y = np.reshape(Y, (len(Y), 1)) / 10 - 0.5
        for i in range(calls):
            # Calculate accuracy for 256 mnist test images
            res = sess.run(pred, feed_dict={x: X,
                                            y: Y,
                                            keep_prob: test_dropout})
            result[i] = res[:, 0]
            # print(np.sum(np.argmax(res, axis=1) == labels) * 1.0 / len(res))
        #variance = np.std(result, axis=0)#
        variance = np.percentile(result, 75, axis=0) - np.percentile(result, 25, axis=0)
        # variance = np.max(result, axis=0) - np.min(result, axis=0)
        result = np.mean(result, axis=0)
        variance = np.mean(variance)
        preds = np.round(result * 10 + 5)
        accuracy = np.sum(preds == labels) * 1.0 / len(preds)
        return accuracy, variance


    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        iters = []
        accuracies = []

        step = 1
        trainSet = {0, 2, 4, 6, 8}
        start = time.time()
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            if groups:
                filter = np.array(map(lambda x: x in trainSet, batch_y))
            else:
                filter = np.array(map(lambda x: x / 10.0 > random(), batch_y))
            batch_x, batch_y = batch_x[filter], batch_y[filter]
            batch_y = np.reshape(batch_y, (len(batch_y), 1)) / 10.0 - 0.5
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                           keep_prob: dropout})
            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                iters.append(step * batch_size)
                accuracies.append(acc)
                print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
            step += 1
        elapsed = (time.time() - start)
        print("Optimization Finished in %f"%elapsed)
        if show_charts:
            plt.plot(iters, accuracies)
            plt.show()

        all_images = np.array(mnist.test.images)
        all_labels = np.array(mnist.test.labels)
        results = []
        accs = []
        certs = []
        start_test = time.time()
        var_known = 0
        var_unknown = 0
        for label in range(0, 10):
            filter = np.array(map(lambda x: x == label, all_labels))
            X, Y = all_images[filter], all_labels[filter]
            acc, certainty = test(X, Y)
            accs.append(acc)
            certs.append(certainty)
            results.append((label, certainty))
            if label in trainSet:
                var_known +=certainty
            else:
                var_unknown+=certainty

            print("label: " + str(label) + ", accuracy: " + "{:.6f}".format(acc) + ", certainty: ""{:.6f}".format(
                certainty))
        elapsed_test = (time.time() - start_test)
        print("Testing Finished in %f" % elapsed_test)

        res = sorted(results, key=lambda tup: tup[1])
        for r in res:
            print("label " + str(r[0]) + ": " + str(r[1]))

        ratio = 1.0 * var_unknown / var_known
        print("ratio: " + str(ratio))
        reg0 = np.corrcoef(accs, certs)[0, 1]
        print("regression :" + str(reg0))
        regNo0 = np.corrcoef(accs[1:], certs[1:])[0, 1]
        print("regression without 0:" + str(regNo0))
        if show_charts:
            plt.scatter(accs, certs)
            plt.show()


        acc = sess.run(accuracy, feed_dict={x: mnist.test.images,
                              y: np.reshape(mnist.test.labels, (len(mnist.test.labels), 1)) / 10.0 - 0.5,
                              keep_prob: 1.0})
        print("Testing Accuracy:", "{:.6f}".format(acc))

        return elapsed, elapsed_test, acc, reg0, regNo0, ratio


if __name__ == "__main__":
    tests = True
    if not tests:
        dropout = 0.75
        test_dropout = 0.5
        calls = 30
        groups = True
        show_charts = False
        run_tests = False
        if len(sys.argv) > 1:
            dropout = float(sys.argv[1])
        if len(sys.argv) > 2:
            test_dropout = float(sys.argv[2])
        if len(sys.argv) > 3:
            calls = float(sys.argv[3])
        if len(sys.argv) > 4:
            groups = str2bool(sys.argv[4])
        if len(sys.argv) > 5:
            show_charts = str2bool(sys.argv[5])
        if len(sys.argv) > 5:
            show_charts = str2bool(sys.argv[5])
        elapsed, elapsed_test, acc, reg0, regNo0, ratio = run(dropout, test_dropout, calls, groups, show_charts)

        print("[RESULTS] Training: " + str(elapsed) + " test: " + str(elapsed_test) + " acc: " + str(acc) + " reg0: " + str(
            reg0) + " regNo0: " + str(regNo0) + " ratio: " + str(ratio))

    dropouts = [0.25, 0.5, 0.75, 1]
    test_dropouts = [0.25, 0.5, 0.75]
    calls = [1, 10, 30, 50, 100]

    runs = 1
    group_tests = []
    with open('/home/wojtek/data/code/Imitation_Learning_VizDoom/spikes/results/mnist_basic_groups.csv', 'wb') as csvfile:
        for i in range(10):
            for call in calls:
                for dropout in dropouts:
                    for test_dropout in test_dropouts:
                        print("Running: " + str(call) + " calls with drops " + str(dropout) + "/" + str(test_dropouts) +"[" + str(runs) + "/" + str(
                            len(calls) * len(dropouts) * len(test_dropouts) * 10) + "]")
                        elapsed, elapsed_test, acc, reg0, regNo0, ratio = run(dropout, test_dropout, call, True, False)
                        group_tests.append((dropout, test_dropout, call, elapsed, elapsed_test, acc, ratio))
                        line = ",".join(map(lambda x: str(x), (
                            call, dropout, test_dropout, "{:.6f}".format(elapsed), "{:.6f}".format(elapsed_test), "{:.6f}".format(acc),
                            "{:.6f}".format(ratio)))) + '\n'

                        csvfile.writelines([line])
                        csvfile.flush()
                        runs += 1
