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
from tensorflow.contrib.distributions.python.ops.sample_stats import percentile
from tensorflow.python.client import timeline

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/")

# Parameters

training_iters = 1000000
batch_size = 128
display_step = 100

# Network Parameters
n_input = 784  # MNIST data input (img shape: 28*28)
subnets = 5
inclusion_prob = 0.75

learning_rate = 0.005 / (subnets * inclusion_prob)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, 1])
mask = tf.placeholder(tf.bool, [subnets])


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
def conv_net(x, mask):
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

    outs = []
    for i in range(subnets):
        if mask[i] is not False:
            fc = tf.contrib.layers.fully_connected(conv2_flat, num_outputs=512, activation_fn=tf.nn.relu,
                                                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                    biases_initializer=tf.constant_initializer(0.1))
            out = tf.contrib.layers.fully_connected(fc, num_outputs=1, activation_fn=None,
                                                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                    biases_initializer=tf.constant_initializer(0.1))
        outs.append(out)
    return outs


# Construct model

outs = conv_net(x, mask)
# optimizers =[]
for i in range(subnets):
    net = outs[i]
    sub_cost = tf.losses.mean_squared_error(net, y)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.6, name="Optimizer_" + str(i)).minimize(
        sub_cost)
    # optimizers.append(optimizer)
# outs = tf.Print(outs, [outs], "Optimizations: " +str(len(outs)) + str(mask))
# optimizer = optimizers
outs = tf.stack(outs)
pred = tf.reduce_mean(outs, 0)
difference = percentile(outs, 75, 0) - percentile(outs, 25, 0)
difference = tf.reduce_mean(difference)
# Define loss and optimizer
cost = tf.losses.mean_squared_error(pred, y)

# optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.6).minimize(cost)

# Evaluate model
rounded = tf.round(pred * 10)
correct_pred = tf.equal(rounded, y * 10)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# Initializing the variables
init = tf.initialize_all_variables()


def test(X, Y):
    calls = 50
    labels = Y  # np.argmax(Y, axis=1)
    Y = np.reshape(Y, (len(Y), 1)) / 10 - 0.5
    result, variance, o = sess.run([pred, difference, outs], feed_dict={x: X,
                                        y: Y, mask: np.random.binomial(1, 1, subnets)})

    preds = np.round(result * 10 + 5)
    accuracy = np.sum(preds == labels) * 1.0 / len(preds)
    return accuracy, variance


# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    run_options = tf.RunOptions(trace_level=tf.RunOptions.SOFTWARE_TRACE) # FULL_TRACE
    run_metadata = tf.RunMetadata()

    step = 1
    trainSet = {0, 2, 4, 6, 8}
    # Keep training until reach max iterations
    start = time.time()
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        filter = np.array(map(lambda x: x in trainSet, batch_y))
        batch_x, batch_y = batch_x[filter], batch_y[filter]
        batch_y = np.reshape(batch_y, (len(batch_y), 1)) / 10.0 - 0.5
        m = np.random.binomial(1, inclusion_prob, subnets)

        optimizers = []
        for i in range(subnets):
            if m[i]:
                optimizers.append("Optimizer_" + str(i))

        sess.run(optimizers, feed_dict={x: batch_x, y: batch_y, mask: m})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            m = np.random.binomial(1, 1, subnets)
            loss, acc, p, r, c = sess.run([cost, accuracy, pred, rounded, correct_pred], feed_dict={x: batch_x,
                                                                                                    y: batch_y, mask: m, }, run_metadata=run_metadata, options=run_options)
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    elapsed = (time.time() - start)
    print("Optimization Finished in %f"%elapsed)
    prof_timeline = timeline.Timeline(run_metadata.step_stats)
    prof_ctf = prof_timeline.generate_chrome_trace_format()
    with open('/tmp/prof_ctf.json', 'w') as fp:
            print ('Dumped to prof_ctf.json')
            fp.write(prof_ctf)

    all_images = np.array(mnist.test.images)
    all_labels = np.array(mnist.test.labels)
    results = []
    var_known = 0
    var_unknown = 0
    for label in range(0, 10):
        filter = np.array(map(lambda x: x == label, all_labels))
        X, Y = all_images[filter], all_labels[filter]
        acc, certainty = test(X, Y)
        results.append((label, certainty))
        if label in trainSet:
            var_known +=certainty
        else:
            var_unknown+=certainty

        print("label: " + str(label) + ", accuracy: " + "{:.6f}".format(acc) + ", certainty: ""{:.6f}".format(
            certainty))
    res = sorted(results, key=lambda tup: tup[1])
    for r in res:
        print("label " + str(r[0]) + ": " + str(r[1]))

    print("ratio: " +str(1.0*var_unknown/var_known))

    validX = mnist.test.images
    validY = np.reshape(mnist.test.labels, (len(mnist.test.labels), 1)) / 10.0 - 0.5
    acc = sess.run(accuracy, feed_dict={x: validX,
                                          y: validY, mask: np.random.binomial(1, 1, subnets)})
    print("Testing Accuracy:", "{:.6f}".format(acc))