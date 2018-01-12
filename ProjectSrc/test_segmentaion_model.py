from __future__ import print_function
import numpy as np
import tensorflow as tf
from Utilities.pipline import DataPipline

# upload data:
data = DataPipline(3, 1, 4, [1], {'zeroPadding': True, 'paddingSize': 240, 'normType': 'reg'})

image_size = 240
num_labels = 4
num_channels = 1 # grayscale

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

batch_size = 64
patch_size = 5
depth = 16
num_hidden = 64

X = tf.placeholder(dtype=tf.float32,shape=[batch_size, image_size, image_size, num_channels])
y = tf.placeholder(dtype=tf.float32, shape=[batch_size, num_labels])
keep_prob = tf.placeholder(dtype=tf.float32)

data.valSamples = tf.cast(data.valSamples, tf.float32)
data.testSamples = tf.cast(data.testSamples, tf.float32)
data.valLabels = tf.cast(data.valLabels, tf.float32)
data.testLabels = tf.cast(data.testLabels, tf.float32)
tf_valSamples = tf.Variable(data.valSamples)
tf_testSamples = tf.Variable(data.testSamples)

# Variables:
# Different dimensions for the max_pooling ....
#
# layer1_weights = tf.Variable(tf.truncated_normal(
#     [patch_size, patch_size, num_channels, depth], stddev=0.1))
# layer1_biases = tf.Variable(tf.zeros([depth]))
# layer2_weights = tf.Variable(tf.truncated_normal(
#     [patch_size, patch_size, depth, depth], stddev=0.1))
# layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
# layer3_weights = tf.Variable(tf.truncated_normal(
#     [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
# layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
# layer4_weights = tf.Variable(tf.truncated_normal(
#     [num_hidden, num_labels], stddev=0.1))
# layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)


def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


def conv_layer(x, Wshape, bshape, name, padding='SAME'):
	W = weight_variable(Wshape)
	b = bias_variable([bshape])
	return tf.nn.relu(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding) + b)


def deconv_layer(x, Wshape, bshape, name, padding='SAME'):
	W = weight_variable(Wshape)
	b = bias_variable([bshape])

	x_shape = tf.shape(x)
	out_shape = tf.stack([x_shape[0], x_shape[1], x_shape[2], Wshape[2]])

	return tf.nn.conv2d_transpose(x, W, out_shape, [1, 1, 1, 1], padding=padding) + b

def model(data):
	# conv - relu - fc -
	conv_1 = conv_layer(data, [3, 3, 3, 64], 64, 'conv_1_1')
	deconv_1 = deconv_layer(conv_1, [3, 3, 3, 64], 64, 'fc6_deconv')
	score_1 = deconv_layer(deconv_1, [1, 1, 21, 64], 21, 'score_1')
	return score_1

# Model.
# def model(data):
#     ''''
#         Model:
#         conv-relu-conv-relu-FC-drop_out-FC
#         Change variables !
#     '''
#     conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
#     hidden = tf.nn.relu(conv + layer1_biases)
#     conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
#     hidden = tf.nn.relu(conv + layer2_biases)
#     shape = hidden.get_shape().as_list()
#     reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
#     hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
#     dropout_layer = tf.nn.dropout(hidden, keep_prob)
#     return tf.matmul(dropout_layer, layer4_weights) + layer4_biases

# Train
logits = model(X)
loss_train = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))

optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss_train)

train_prediction = tf.nn.softmax(logits)
valid_prediction = tf.nn.softmax(model(tf_valSamples))
test_prediction = tf.nn.softmax(model(tf_testSamples))

num_steps = 3000

with tf.Session() as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
        batch_data, batch_labels = data.next_train_random_batch(batch_size)
        batch_data = tf.cast(batch_data, tf.float32)
        batch_labels = tf.cast(batch_labels, tf.float32)
        feed_dict = {X: batch_data, y: batch_labels, keep_prob: 0.6}
        _, l, predictions = session.run(
            [optimizer, loss_train, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(
                valid_prediction.eval(feed_dict={keep_prob: 1}), data.valLabels))
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(feed_dict={keep_prob: 1}), data.testLabels))