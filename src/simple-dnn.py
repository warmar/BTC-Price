import tensorflow as tf
import random
from pprint import pprint
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import math

# Set Network Training Parameters
HIDDEN_LAYER_SIZES = [500, 250, 100]
ACTIVATION_FUNCTION = tf.sigmoid
OPTIMIZER = tf.train.GradientDescentOptimizer(0.1)
NUM_EPOCHS = 10000
CHUNK_SIZE = 500

# Set Random Seeds
random.seed(0)
tf.set_random_seed(0)

# Load Training Data
# post, label
training_data = pd.read_csv('raw_training_data.csv', header=0)

vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)

# Load posts and labels as lists
post_list = training_data['post'].tolist()
label_list = training_data['label'].tolist()

# Remove any broken data points
for i, post in reversed(list(enumerate(post_list))):
    if not type(post)==str:
        del post_list[i]
        del label_list[i]

# Vectorize posts
vectorized_training_data = vectorizer.fit_transform(post_list)
vocab = vectorizer.get_feature_names()

x = [datum.toarray()[0] for datum in vectorized_training_data[:-1000]]
test_x = [datum.toarray()[0] for datum in vectorized_training_data[-1000:]]
y = [[datum] for datum in label_list[:-1000]]
test_y = [[datum] for datum in label_list[-1000:]]

# Network Layers
num_features = len(x[0])
num_labels = len(y[0])

layer_sizes = [num_features, *HIDDEN_LAYER_SIZES, num_labels]

# Generate Netork Weights
weights = []
for i, size in enumerate(layer_sizes[1:]):
    weights.append(
        tf.Variable(
            tf.random_normal((layer_sizes[i], layer_sizes[i+1]))
        )
    )

# Generate Network Biases
biases = []
for weight in weights:
    biases.append(
        tf.Variable(
            tf.zeros((weight.shape[1]))
        )
    )

# Create Network Graph
layers = []
x_ = tf.placeholder(tf.float32, (None, num_features), name='x_placeholder')
y_ = tf.placeholder(tf.float32, (None, num_labels), name='y_placeholder')
layers.append(x_)

for i in range(len(weights)):
    layers.append(
        ACTIVATION_FUNCTION(
            tf.matmul(layers[i], weights[i]) + biases[i]
        )
    )


output = layers[-1]

pprint(layers)

# Create Cost Function
cost = tf.reduce_mean((output - y_)**2)

# Define Train
train = OPTIMIZER.minimize(cost)

# Define Evaluation
prediction = tf.round(output)
correct_predictions = tf.equal(prediction, y_)
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

# Train
sess = tf.Session()

sess.run(tf.global_variables_initializer())

x_chunks = []
y_chunks = []
num_chunks = math.ceil(len(x)/CHUNK_SIZE)

for i in range(num_chunks):
    x_chunks.append(x[i * CHUNK_SIZE:(i+1) * CHUNK_SIZE])
    y_chunks.append(y[i * CHUNK_SIZE:(i+1) * CHUNK_SIZE])

print([len(x) for x in x_chunks])
print('total: ', len(x))

for i in range(NUM_EPOCHS):
    if i % int(NUM_EPOCHS/100) == 0:
        print('Cost: ', sess.run(cost, feed_dict={x_: x, y_: y}))
        print('Training Accuracy: ', sess.run(accuracy, feed_dict={x_: x, y_: y}))
        print('Testing Accuracy: ', sess.run(accuracy, feed_dict={x_: test_x, y_: test_y}))
    
    x_chunk = x_chunks[i % num_chunks]
    y_chunk = y_chunks[i % num_chunks]
    sess.run(train, feed_dict={x_: x_chunk, y_: y_chunk})

pprint(sess.run(tf.round((output-y_)*100)/100, feed_dict={x_: x, y_: y}))
