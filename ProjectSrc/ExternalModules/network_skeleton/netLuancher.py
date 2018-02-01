from ExternalModules.network_skeleton.loadData import *
from ExternalModules.network_skeleton.layers import *
import tensorflow as tf
import tensorboard as tb
import numpy as np
import os

# CONSTANTS:

IMAGE_SIZE = 128
NUM_CHANNELS = 1
NUM_LABELS = 1
BATCH_SIZE = 32
KERNEL_SIZE = 3
DEPTH = 32
POOL_SIZE = 2

# LOAD DATA|:

print('Start load data')
data = load_data(os.path.realpath(__file__ + "/../../" + 'toy_segmentaion_data/data'))
label = load_labels(os.path.realpath(__file__ + "/../../" + 'toy_segmentaion_data/labels'))
print('End load data\n')

train_num = 1000
val_num = 1500
test_num = 500

train_dataset = data[:train_num]
train_labels = label[:train_num]
valid_dataset = data[train_num:val_num]
valid_labels = label[train_num:val_num]
test_dataset = data[val_num:]
test_labels = label[val_num:]

print('Training set', np.shape(train_dataset), np.shape(train_labels))
print('Validation set', np.shape(valid_dataset), np.shape(valid_labels))
print('Test set', np.shape(test_dataset), np.shape(test_labels))

train_dataset = np.reshape(train_dataset, [-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS]).astype(np.float32)
train_labels = np.reshape(train_labels, [-1, IMAGE_SIZE, IMAGE_SIZE, NUM_LABELS]).astype(np.float32)
valid_labels = np.reshape(valid_labels, [-1, IMAGE_SIZE, IMAGE_SIZE, NUM_LABELS]).astype(np.float32)
test_labels = np.reshape(test_labels, [-1, IMAGE_SIZE, IMAGE_SIZE, NUM_LABELS]).astype(np.float32)

valid_dataset = np.reshape(valid_dataset, [-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS]).astype(np.float32)
test_dataset = np.reshape(test_dataset, [-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS]).astype(np.float32)



graph = tf.Graph()

with graph.as_default():

	valid_dataset = tf.constant(valid_dataset, dtype=tf.float32)
	test_dataset = tf.constant(test_dataset, dtype=tf.float32)

	### ---------- Model ----------- ####

	X = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
	Y = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, NUM_LABELS])

	def unet_model(X, unet_num_layers = 3):

		# downsample

		# unet_layer_1
		W1_1 = weight_variable([KERNEL_SIZE, KERNEL_SIZE, NUM_CHANNELS, DEPTH])
		b1_1 = bias_variable([DEPTH])
		W1_2 = weight_variable([KERNEL_SIZE, KERNEL_SIZE, DEPTH, DEPTH])
		b1_2 = bias_variable([DEPTH])

		conv1_1 = conv2d(X, W1_1, b1_1)
		conv1_2 = conv2d(conv1_1, W1_2, b1_2)
		max1 = max_pool(conv1_2, 2) # [N, IMAGE_SIZE // 2, IMAGE_SIZE // 2, DEPTH]

		# unet_layer_2
		W2_1 = weight_variable([KERNEL_SIZE, KERNEL_SIZE, DEPTH, DEPTH*2])
		b2_1 = bias_variable([DEPTH*2])
		W2_2 = weight_variable([KERNEL_SIZE, KERNEL_SIZE, DEPTH*2, DEPTH*2])
		b2_2 = bias_variable([DEPTH*2])

		conv2_1 = conv2d(max1, W2_1, b2_1)
		conv2_2 = conv2d(conv2_1, W2_2, b2_2)
		max2 = max_pool(conv2_2, 2)  # [N, IMAGE_SIZE // 4, IMAGE_SIZE // 4, DEPTH * 2]

		# unet_layer_3
		W3_1 = weight_variable([KERNEL_SIZE, KERNEL_SIZE, DEPTH * 2, DEPTH * 4])
		b3_1 = bias_variable([DEPTH * 4])
		W3_2 = weight_variable([KERNEL_SIZE, KERNEL_SIZE, DEPTH * 4, DEPTH * 4])
		b3_2 = bias_variable([DEPTH * 4])

		conv3_1 = conv2d(max2, W3_1, b3_1)
		conv3_2 = conv2d(conv3_1, W3_2, b3_2)
		max3 = max_pool(conv3_2, 2)  # [N, IMAGE_SIZE // 8, IMAGE_SIZE // 8, DEPTH * 4]

		# unet_layer_4
		W4_1 = weight_variable([KERNEL_SIZE, KERNEL_SIZE, DEPTH * 4, DEPTH * 8])
		b4_1 = bias_variable([DEPTH * 8])
		conv4_1 = conv2d(max3, W4_1, b4_1)
		W4_2 = weight_variable([KERNEL_SIZE, KERNEL_SIZE, DEPTH * 8, DEPTH * 8])
		b4_2 = bias_variable([DEPTH * 8])
		conv4_2 = conv2d(conv4_1, W4_2, b4_2)

		# upsample

		# unet_upsample_layer_1 (deconve - concat - conv - conv)
		WD1_1 = weight_variable_devonc([POOL_SIZE, POOL_SIZE, DEPTH*4, DEPTH*8])
		bd1_1 = bias_variable([DEPTH*4])
		dconv1_1 = deconv2d(conv4_2, WD1_1, bd1_1, POOL_SIZE)
		concat1 = concat(conv3_2, dconv1_1)

		WD1_2 = weight_variable([KERNEL_SIZE, KERNEL_SIZE, DEPTH * 8, DEPTH * 4])
		bd1_2 = bias_variable([DEPTH * 4])
		WD1_3 = weight_variable([KERNEL_SIZE, KERNEL_SIZE, DEPTH * 4, DEPTH * 4])
		bd1_3 = bias_variable([DEPTH * 4])
		dconv1_2 = conv2d(concat1, WD1_2, bd1_2)
		dconv1_3 = conv2d(dconv1_2, WD1_3, bd1_3)

		# unet_upsample_layer_2
		WD2_1 = weight_variable_devonc([POOL_SIZE, POOL_SIZE, DEPTH*2, DEPTH*4])
		bd2_1 = bias_variable([DEPTH*2])
		dconv2_1 = deconv2d(dconv1_3, WD2_1, bd2_1, POOL_SIZE)
		concat2 = concat(conv2_2, dconv2_1)

		WD2_2 = weight_variable([KERNEL_SIZE, KERNEL_SIZE, DEPTH*4, DEPTH*2])
		bd2_2 = bias_variable([DEPTH*2])
		WD2_3 = weight_variable([KERNEL_SIZE, KERNEL_SIZE, DEPTH*2, DEPTH*2])
		bd2_3 = bias_variable([DEPTH*2])
		dconv2_2 = conv2d(concat2, WD2_2, bd2_2)
		dconv2_3 = conv2d(dconv2_2, WD2_3, bd2_3)

		# unet_upsample_layer_3
		WD3_1 = weight_variable_devonc([POOL_SIZE, POOL_SIZE, DEPTH, DEPTH*2])
		bd3_1 = bias_variable([DEPTH])
		dconv3_1 = deconv2d(dconv2_3, WD3_1, bd3_1, POOL_SIZE)
		concat3 = concat(conv1_2, dconv3_1)

		WD3_2 = weight_variable([KERNEL_SIZE, KERNEL_SIZE, DEPTH*2, DEPTH])
		bd3_2 = bias_variable([DEPTH])
		WD3_3 = weight_variable([KERNEL_SIZE, KERNEL_SIZE, DEPTH, DEPTH])
		bd3_3 = bias_variable([DEPTH])
		dconv3_2 = conv2d(concat3, WD3_2, bd3_2)
		dconv3_3 = conv2d(dconv3_2, WD3_3, bd3_3)

		# FC (conv with kernel of 1x1)
		Wfc = tf.Variable(tf.truncated_normal([1, 1, DEPTH, NUM_LABELS], stddev=1),name='Wfc')
		# Wfc = weight_variable([1, 1, DEPTH, NUM_LABELS])
		bfc = bias_variable([NUM_LABELS])
		# return tf.nn.conv2d(dconv3_3, Wfc, strides=[1, 1, 1, 1], padding='SAME') + bfc
		return conv2d(dconv3_3, Wfc, bfc)
		# temp architecture TODO: FINISH ROY

	logits = unet_model(X)
	loss_train = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits))
	optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss_train)
	train_prediction = tf.nn.softmax(logits)
	valid_prediction = tf.nn.softmax(unet_model(valid_dataset))
	test_prediction = tf.nn.softmax(unet_model(test_dataset))

	def diceScore(prediction, groundTruth):
		tmp = np.zeros_like(prediction)
		tmp[prediction > np.mean(prediction)] = 1
		return 2*np.sum((np.multiply(tmp, groundTruth))) / np.sum(tmp + groundTruth)

	def accuracy(prediction, groundTruth):
		tmp = np.zeros_like(prediction)
		tmp[prediction > np.mean(prediction)] = 1
		eq = np.equal(tmp, groundTruth)
		return np.mean(eq)

num_steps = 300

with tf.Session(graph=graph) as session:
	tf.global_variables_initializer().run()
	print('Initialized')
	for step in range(num_steps):
		ind = np.random.randint(0, train_num - 1, BATCH_SIZE)
		batch_data = train_dataset[ind, :, :, :]
		batch_labels = train_labels[ind, :, :, :]
		feed_dict = {X: batch_data, Y: batch_labels}
		_, l, predictions = session.run(
			[optimizer, loss_train, train_prediction], feed_dict=feed_dict)
		if (step % 5 == 0):
			print('Minibatch loss at step %d: %f' % (step, l))
			print('Minibatch accuracy: %.3f%%' % accuracy(predictions, batch_labels))
			# print('Minibatch dice: %.3f%' % diceScore(predictions, batch_labels))
			prediction = valid_prediction.eval()
			print('Validation accuracy: %.3f%%' % accuracy(prediction, valid_labels))
	print('Test accuracy: %.3f%%' % accuracy(test_prediction.eval(), test_labels))








