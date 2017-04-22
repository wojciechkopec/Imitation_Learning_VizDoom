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

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/")

# Parameters
learning_rate = 0.0005
training_iters = 2000000
batch_size = 128
display_step = 100

# Network Parameters
n_input = 784  # MNIST data input (img shape: 28*28)
dropout = 0.75  # Dropout, probability to keep units
test_dropout = 0.5

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
    calls = 50
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
    # variance = np.percentile(result, 75, axis=0) - np.percentile(result, 25, axis=0)
    variance = np.max(result, axis=0) - np.min(result, axis=0)
    result = np.mean(result, axis=0)
    variance = np.mean(variance)
    preds = np.round(result * 10 + 5)
    accuracy = np.sum(preds == labels) * 1.0 / len(preds)
    return accuracy, variance


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    trainSet = {0, 2, 4, 6, 8}
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        filter = np.array(map(lambda x: x in trainSet, batch_y))
        batch_x, batch_y = batch_x[filter], batch_y[filter]
        batch_y = np.reshape(batch_y, (len(batch_y), 1)) / 10.0 - 0.5
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc, p, r, c = sess.run([cost, accuracy, pred, rounded, correct_pred], feed_dict={x: batch_x,
                                                                                                    y: batch_y,
                                                                                                    keep_prob: 1.0})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    all_images = np.array(mnist.test.images)
    all_labels = np.array(mnist.test.labels)
    results = []
    var_known = 0
    var_unknown = 0
    for label in range(0, 10):
        filter = np.array(map(lambda x: x == label, all_labels))
        X, Y = all_images[filter], all_labels[filter]
        accuracy, certainty = test(X, Y)
        results.append((label, certainty))
        if label in trainSet:
            var_known +=certainty
        else:
            var_unknown+=certainty

        print("label: " + str(label) + ", accuracy: " + "{:.6f}".format(accuracy) + ", certainty: ""{:.6f}".format(
            certainty))
    res = sorted(results, key=lambda tup: tup[1])
    for r in res:
        print("label " + str(r[0]) + ": " + str(r[1]))

    print("ratio: " +str(1.0*var_unknown/var_known))


            # acc = sess.run(accuracy, feed_dict={x: mnist.test.images,
    #                               y: np.reshape(mnist.test.labels, (len(mnist.test.labels), 1)) / 10.0 - 0.5,
    #                               keep_prob: 1.0})
    # print("Testing Accuracy:", "{:.6f}".format(acc))
