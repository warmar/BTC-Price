import tensorflow as tf
import random
from pprint import pprint

# Set Network Training Parameters
HIDDEN_LAYER_SIZES = [100, 50, 10]
ACTIVATION_FUNCTION = tf.sigmoid
OPTIMIZER = tf.train.GradientDescentOptimizer(0.1)
NUM_EPOCHS = 10000

# Set Random Seeds
random.seed(0)
tf.set_random_seed(0)

# Training Data
x = []
y = []

# Generate Random Training Dataa
for i in range(10):
    vec = []
    for j in range(10):
        vec.append(random.choice((0.,1.)))
    x.append(vec)

for i in range(10):
    vec = []
    for j in range(1):
        vec.append(random.choice((0.,1.)))
    y.append(vec)

pprint(x)
pprint(y)

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

# Train
sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(NUM_EPOCHS):
    if i % int(NUM_EPOCHS/10) == 0:
        print(sess.run(cost, feed_dict={x_: x, y_: y}))
    sess.run(train, feed_dict={x_: x, y_: y})

pprint(sess.run(tf.round((output-y_)*100)/100, feed_dict={x_: x, y_: y}))
