from pprint import pprint
import random
import math
import json
import sys
import os
import tensorflow as tf

# Set Network Training Parameters
HIDDEN_LAYER_SIZES = [500, 250, 100]
ACTIVATION_FUNCTION = tf.sigmoid
OPTIMIZER = tf.train.GradientDescentOptimizer(0.1)
NUM_EPOCHS = 10000
CHUNK_SIZE = 500

# Choose Dataset
PERIOD = 86400
OFFSET = -86400
MINIMUM_SCORE = 100

# Set Random Seeds
random.seed(0)
tf.set_random_seed(0)

# Load Training Data
description = 'minscore=%s,period=%s,offset=%s' % (MINIMUM_SCORE, PERIOD, OFFSET)

if not os.path.exists('processed_data-%s' % description):
    print('Dataset does not exist: %s' % description)
    sys.exit(1)

vectorized_posts = json.load(open('processed_data-%s/vectorized_posts' % description))
labels = json.load(open('processed_data-%s/labels' % description))

# Shuffle Data
combined_data = list(zip(vectorized_posts, labels))
random.shuffle(combined_data)
shuffled_vectorized_posts = [datum[0] for datum in combined_data]
shuffled_labels = [[datum[1]] for datum in combined_data]

# Split data into training and testing groups
x = shuffled_vectorized_posts[:-1000]
test_x = shuffled_vectorized_posts[-1000:]
y = shuffled_labels[:-1000]
test_y = shuffled_labels[-1000:]

# Network Layers
num_features = len(x[0])
num_labels = len(y[0])

layer_sizes = [num_features, *HIDDEN_LAYER_SIZES, num_labels]

# Generate Netork Weights
weights = []
for i, size in enumerate(layer_sizes[1:]):
    with tf.name_scope('weights'):
        weights.append( 
            tf.Variable(
                tf.random_normal((layer_sizes[i], layer_sizes[i+1])), name='weights_%s' % i
            )
        )
        tf.summary.histogram('weights_%s' % i, weights[-1])

# Generate Network Biases
biases = []
for i, weight in enumerate(weights):
    with tf.name_scope('biases'):
        biases.append(
            tf.Variable(
                tf.zeros((weight.shape[1])), name='biases_%s' % i
            )
        )
        tf.summary.histogram('biases_%s' % i, biases[-1])

# Create Network Graph
layers = []
x_ = tf.placeholder(tf.float32, (None, num_features), name='x_placeholder')
y_ = tf.placeholder(tf.float32, (None, num_labels), name='y_placeholder')
layers.append(x_)

for i in range(len(weights)):
    with tf.name_scope('layer_%s' % i):
        layers.append(
            ACTIVATION_FUNCTION(
                tf.matmul(layers[i], weights[i]) + biases[i]
            )
        )


output = layers[-1]

pprint(layers)

# Create Cost Function
cost = tf.reduce_mean((output - y_)**2)
tf.summary.scalar('cost', cost)

# Define Train
train = OPTIMIZER.minimize(cost)

# Define Evaluation
prediction = tf.round(output)
correct_predictions = tf.equal(prediction, y_)
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

tf.summary.scalar('accuracy', accuracy)

# Train
sess = tf.Session()

sess.run(tf.global_variables_initializer())

# Split data into chunks for stochastic descent
x_chunks = []
y_chunks = []
num_chunks = math.ceil(len(x)/CHUNK_SIZE)

for i in range(num_chunks):
    x_chunks.append(x[i * CHUNK_SIZE:(i+1) * CHUNK_SIZE])
    y_chunks.append(y[i * CHUNK_SIZE:(i+1) * CHUNK_SIZE])

# Define summary variables
merged_summary = tf.summary.merge_all()
train_summary_writer = tf.summary.FileWriter('tensorboard/%s/train/' % description, sess.graph)
test_summary_writer = tf.summary.FileWriter('tensorboard/%s/test/' % description, sess.graph)

# Train Networks
for i in range(NUM_EPOCHS):
    # Evaluate Cost and Accuracy
    if i % int(NUM_EPOCHS/100) == 0:
        print('Cost: ', sess.run(cost, feed_dict={x_: x, y_: y}))
        print('Training Accuracy: ', sess.run(accuracy, feed_dict={x_: x, y_: y}))
        print('Testing Accuracy: ', sess.run(accuracy, feed_dict={x_: test_x, y_: test_y}))
    
    # Write Summaries
    if i % 10 == 0:
        train_summary = sess.run(merged_summary, feed_dict={x_: x, y_: y})
        train_summary_writer.add_summary(train_summary, i)

        test_summary = sess.run(merged_summary, feed_dict={x_: test_x, y_: test_y})
        test_summary_writer.add_summary(test_summary, i)

    # Perform stochastic gradient descent
    x_chunk = x_chunks[i % num_chunks]
    y_chunk = y_chunks[i % num_chunks]
    sess.run(train, feed_dict={x_: x_chunk, y_: y_chunk})

pprint(sess.run(tf.round((output-y_)*100)/100, feed_dict={x_: x, y_: y}))
