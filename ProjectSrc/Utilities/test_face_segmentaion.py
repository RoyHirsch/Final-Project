import tensorflow as tf
import numpy as np
import os
import scipy.io as spio
import scipy.ndimage as sci
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import sys

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def load_data(rootDir = '/Users/royhirsch/PycharmProjects/NN_self_learning/segmentaion_data/data', maxNum = 2000):
	Xtrain = []
	ind = 0
	for root, dirs, files in os.walk(rootDir):
		for fileName in files:
			# print(filename)
			match = re.search(r'.*.jpg', fileName)
			if match:
				img = plt.imread(os.path.join(root, fileName))
				if len(np.shape(img)) > 2:
					img = rgb2gray(np.array(img))
				img = img / 255 #normalize
				Xtrain.append(img)
				ind += 1
				if ind >= maxNum:
					return Xtrain
	return Xtrain

def load_labels(rootDir = '/Users/royhirsch/PycharmProjects/NN_self_learning/segmentaion_data/labels', maxNum = 2000):
	ytrain = []
	ind = 0
	for root, dirs, files in os.walk(rootDir):
		for fileName in files:
			# print(filename)
			match = re.search(r'.*.jpg', fileName)
			if match:
				img = plt.imread(os.path.join(root, fileName))
				img = img/255 # normalize
				ytrain.append(img)
				ind += 1
				if ind >= maxNum:
					return ytrain
	return ytrain

print('start load data ')
data = load_data()
print(np.shape(data))

label = load_labels()
print(np.shape(label))
print('end load data ')

train_num = 1000
val_num = 1500

train_dataset = data[:train_num]
train_labels = label[:train_num]
valid_dataset = data[train_num:val_num]
valid_labels = label[train_num:val_num]
test_dataset = data[val_num:]
test_labels = label[val_num:]

print('Training set', np.shape(train_dataset), np.shape(train_labels))
print('Validation set', np.shape(valid_dataset), np.shape(valid_labels))
print('Test set', np.shape(test_dataset), np.shape(test_labels))

# plt.imshow(data[0], cmap='gray')
# plt.show()
# plt.imshow(label[0], cmap='gray')
# plt.show()

image_size = 128
num_labels = 2
num_channels = 1 # grayscale

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


def get_bilinear_filter(filter_shape, upscale_factor):
	##filter_shape is [width, height, num_in_channels, num_out_channels]
	kernel_size = filter_shape[1]
	### Centre location of the filter for which value is calculated
	if kernel_size % 2 == 1:
		centre_location = upscale_factor - 1
	else:
		centre_location = upscale_factor - 0.5

	bilinear = np.zeros([filter_shape[0], filter_shape[1]])
	for x in range(filter_shape[0]):
		for y in range(filter_shape[1]):
			##Interpolation Calculation
			value = (1 - abs((x - centre_location) / upscale_factor)) * (
			1 - abs((y - centre_location) / upscale_factor))
			bilinear[x, y] = value
	weights = np.zeros(filter_shape)
	for i in range(filter_shape[2]):
		weights[:, :, i, i] = bilinear
	init = tf.constant_initializer(value=weights,
	                               dtype=tf.float32)

	bilinear_weights = tf.get_variable(name="decon_bilinear_filter", initializer=init,
	                                   shape=weights.shape)
	return bilinear_weights


batch_size = 64
patch_size = 5
depth = 16
num_hidden = 64


graph = tf.Graph()

with graph.as_default():
    X = tf.placeholder(dtype=tf.float32,shape=[batch_size, image_size, image_size, num_channels])
    y = tf.placeholder(dtype=tf.float32, shape=[batch_size, num_labels])
    keep_prob = tf.placeholder(dtype=tf.float32)

    tf_valid_dataset = tf.constant(np.array(valid_dataset))
    tf_test_dataset = tf.constant(np.array(test_dataset))

    # Variables:
    # Different dimensions for the max_pooling ....

    layer1_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, num_channels, depth], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))
    layer2_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
    layer3_weights = tf.Variable(tf.truncated_normal(
        [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    layer4_weights = tf.Variable(tf.truncated_normal(
        [num_hidden, num_labels], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))


    # Model.
    def model(data):
        ''''
            Model:
            conv-relu-max_pool(2x2)-conv-relu-max_pool(2x2)-FC-drop_out-FC
            Accuracy of 93%
        '''
        conv_1 = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
        hidden_1 = tf.nn.relu(conv_1 + layer1_biases)
        max_pool_1 = tf.nn.max_pool(hidden_1, [1,2,2,1], [1,2,2,1], padding='SAME')

        conv_2 = tf.nn.conv2d(max_pool_1, layer2_weights, [1, 1, 1, 1], padding='SAME')
        hidden_2 = tf.nn.relu(conv_2 + layer2_biases)
        max_pool_2 = tf.nn.max_pool(hidden_2, [1,2,2,1], [1,2,2,1], padding='SAME')

        shape = max_pool_2.get_shape().as_list()
        reshape = tf.reshape(max_pool_2, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)

        dropout_layer = tf.nn.dropout(hidden, keep_prob)
        return tf.matmul(dropout_layer, layer4_weights) + layer4_biases


    # Train
    logits = model(X)
    loss_train = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))

    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss_train)

    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))

    num_steps = 1000

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print('Initialized')
        for step in range(num_steps):
            ind = np.random.random_integers(0, train_num, batch_size)
            batch_data = train_dataset[ind, :, :]
            batch_labels = train_labels[ind, :]
            feed_dict = {X: batch_data, y: batch_labels, keep_prob: 0.6}
            _, l, predictions = session.run(
                [optimizer, loss_train, train_prediction], feed_dict=feed_dict)
            if (step % 50 == 0):
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                print('Validation accuracy: %.1f%%' % accuracy(
                    valid_prediction.eval(feed_dict={keep_prob: 1}), valid_labels))
        print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(feed_dict={keep_prob: 1}), test_labels))



