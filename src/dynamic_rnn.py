""" Dynamic Recurrent Neural Network.

TensorFlow implementation of a Recurrent Neural Network (LSTM) that performs
dynamic computation over sequences with variable length. This example is using
a toy dataset to classify linear sequences. The generated sequences have
variable length.

Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)

Original Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/

Adapted by Jeff, Peter, and Zac
Done with Reddit and BTC price data.
"""

from __future__ import print_function

import tensorflow as tf
import random
import sys
from pprint import pprint
import pandas as pd
import json

# ====================
#  DATA GENERATOR
# ====================
class SequenceGenerator(object):
    """ Generate sequence of data with dynamic length.
    This class generate samples for training:
    - Class 0: Price goes down
    - Class 1: Price goes up

    Some general notes about data:
        - Dictionary is built with 10,000 most common words. (Not all words)
        - For training data, we take first n_samples posts.
        - For testing, we take last n_samples data.
        - We take the maximum post length among the n_samples for max_seq_len
            - BUT, if more time, we should've done statistical analysis
              on mean and median reddit post lengths. (We have suspicions that
              the reddit post lengths are right skewed.)

    NOTICE:
    We have to pad each sequence to reach 'max_seq_len' for TensorFlow
    consistency (we cannot feed a numpy array with inconsistent
    dimensions). The dynamic calculation will then be perform thanks to
    'seqlen' attribute that records every actual sequence length.
    """

    # This function returns a cleaned list of words from a string.
    def build_wordlist(self, post):
        post = post.strip()
        post = post.split(' ')
        post = [x.strip() for x in post if x]
        return post

    def __init__(self, n_samples=1000, testing=False):
        train = pd.read_csv("raw_training_data.csv")
        with open('dictionary.json', 'r') as f:
            dictionary = json.load(f)
        if testing:
            training_data = train[-n_samples:]
        else:
            training_data = train[:n_samples]
        self.data = []
        self.labels = []
        self.seqlen = []
        self.max_seq_len = 0
        for index, row in training_data.iterrows():
            post = self.build_wordlist(str(row["post"]))
            # Building the vector using the dictionary
            vector = []
            for word in post:
                if word in dictionary:
                    vector.append([dictionary[word]])
            # A list to keep track of all the lenghts of the posts
            self.seqlen.append(len(vector))
            # Finding the max_seq_len
            if len(vector) > self.max_seq_len:
                self.max_seq_len = len(vector)
            # Add the vector
            self.data.append(vector)
            # Adding the label
            if int(row['label']) == 0:
                self.labels.append([0., 1.])
            else:
                self.labels.append([1., 0.])
        self.batch_id = 0

    # PADDING FUNCTION!!!
    def pad(self, max_seq_len_):
        for index, datum in enumerate(self.data):
            while len(datum) < max_seq_len_:
                self.data[index].append([0])

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen


# ==========
#   MODEL
# ==========

# Parameters
learning_rate = 0.01
training_steps = 10000
batch_size = 128
display_step = 200
seq_max_len = 0 # Global max langth

# Network Parameters
n_hidden = 64 # hidden layer num of features
n_classes = 2 # linear sequence or not

# Data here!
trainset = SequenceGenerator(n_samples=5000)
testset = SequenceGenerator(n_samples=5000, testing=True)

# Padding the data
if trainset.max_seq_len > testset.max_seq_len:
    seq_max_len = trainset.max_seq_len
else:
    seq_max_len = testset.max_seq_len
trainset.pad(max_seq_len_=seq_max_len)
testset.pad(max_seq_len_=seq_max_len)

print("Maximum sequence length: " + str(seq_max_len))

# tf Graph input
x = tf.placeholder("float", [None, seq_max_len, 1])
y = tf.placeholder("float", [None, n_classes])
# A placeholder for indicating each sequence length
seqlen = tf.placeholder(tf.int32, [None])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def dynamicRNN(x, seqlen, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, seq_max_len, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)

    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32,
                                sequence_length=seqlen)

    # When performing dynamic calculation, we must retrieve the last
    # dynamically computed output, i.e., if a sequence length is 10, we need
    # to retrieve the 10th output.
    # However TensorFlow doesn't support advanced indexing yet, so we build
    # a custom op that for each sample in batch size, get its length and
    # get the corresponding relevant output.

    # 'outputs' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_step, n_input]
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])

    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

    # Linear activation, using outputs computed above
    return tf.matmul(outputs, weights['out']) + biases['out']

pred = dynamicRNN(x, seqlen, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps + 1):
        batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       seqlen: batch_seqlen})
        if step % display_step == 0 or step == 1:
            # Calculate batch accuracy & loss
            acc, loss = sess.run([accuracy, cost], feed_dict={x: batch_x, y: batch_y,
                                                seqlen: batch_seqlen})
            print("Step " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
            # Calculate accuracy
            test_data = testset.data
            test_label = testset.labels
            test_seqlen = testset.seqlen
            print("Testing Accuracy:", \
                sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                              seqlen: test_seqlen}))

    print("Optimization Finished!")
