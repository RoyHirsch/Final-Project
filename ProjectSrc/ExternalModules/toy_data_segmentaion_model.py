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

def load_data(rootDir, maxNum = 2000):
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

def load_labels(rootDir, maxNum = 2000):
	ytrain = []
	ind = 0
	for root, dirs, files in os.walk(rootDir):
		for fileName in files:
			# print(filename)
			match = re.search(r'.*.jpg', fileName)
			if match:
				img = plt.imread(os.path.join(root, fileName))
				img = img/255 # normalize
				class_1 = np.zeros(np.shape(img))
				class_2 = np.zeros(np.shape(img))
				class_1[img >= 0.5] = 1  # binary image
				class_1[img < 0.5] = 0
				class_2[img >= 0.5] = 1  # binary image
				class_2[img < 0.5] = 0
				ytrain.append(np.stack((class_1, class_2),axis=2))
				ind += 1
				if ind >= maxNum:
					return ytrain
	return ytrain

print('start load data ')
data = load_data(os.path.realpath(__file__ + "/../" + 'toy_segmentaion_data/data'))
print(np.shape(data))

label = load_labels(os.path.realpath(__file__ + "/../" + 'toy_segmentaion_data/labels'))
print(np.shape(label))
print('end load data ')

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

# plt.imshow(data[0], cmap='gray')
# plt.show()
# plt.imshow(label[0], cmap='gray')
# plt.show()

image_size = 128
num_labels = 2
num_channels = 1 # grayscale

def accuracy(predictions, labels):
	tmp = np.equal(np.argmax(predictions, 3), np.argmax(labels, 3))
	return np.mean(tmp)
	# return (100.0 * np.sum(np.argmax(predictions, 3) == np.argmax(labels, 3))
    #         / image_size*image_size)


batch_size = 64
kernel_size = 5
depth = 16
num_hidden = 64


graph = tf.Graph()

with graph.as_default():
    X = tf.placeholder(dtype=tf.float32,shape=[batch_size, image_size, image_size, num_channels])
    y = tf.placeholder(dtype=tf.float32, shape=[batch_size, image_size, image_size, num_labels])

    train_dataset = np.reshape(train_dataset, [-1, image_size, image_size, num_channels]).astype(np.float32)
    train_labels = np.reshape(train_labels, [-1, image_size, image_size, num_labels]).astype(np.float32)
    valid_labels = np.reshape(valid_labels, [-1, image_size, image_size, num_labels]).astype(np.float32)
    test_labels = np.reshape(test_labels, [-1, image_size, image_size, num_labels]).astype(np.float32)

    valid_dataset = np.reshape(valid_dataset, [-1, image_size, image_size, num_channels]).astype(np.float32)
    test_dataset = np.reshape(test_dataset, [-1, image_size, image_size, num_channels]).astype(np.float32)

    tf_valid_dataset = tf.constant(valid_dataset,dtype=tf.float32)
    tf_test_dataset = tf.constant(test_dataset,dtype=tf.float32)

    # Variables:
    # Different dimensions for the max_pooling ....

    W1 = tf.Variable(tf.truncated_normal(
        [kernel_size, kernel_size, num_channels, depth], stddev=0.1, name='W1'))
    b1 = tf.Variable(tf.zeros([depth]), name='b1')

    W2 = tf.Variable(tf.truncated_normal(
        [kernel_size, kernel_size, depth, depth*2], stddev=0.1), name='W2')
    b2 = tf.Variable(tf.constant(1.0, shape=[depth*2]), name='b2')

    W3 = tf.Variable(tf.truncated_normal(
	    [2, 2, depth, depth * 2], stddev=0.1),name='W3',dtype=tf.float32)
    b3 = tf.Variable(tf.constant(1.0, shape=[depth]),name='b3',dtype=tf.float32)

    W4 = tf.Variable(tf.truncated_normal(
	    [2, 2, num_labels, depth], stddev=0.1), name='W3',dtype=tf.float32)
    b4 = tf.Variable(tf.constant(1.0, shape=[num_labels]), name='b3',dtype=tf.float32)

   # Model.
    def model(data):
        ''''
            Model:
            conv-relu-max_pool(2x2)-conv-relu-max_pool(2x2)-relu-drop_out-FC
            Accuracy of 93%
        '''
        conv_1 = tf.nn.conv2d(data, W1, [1, 1, 1, 1], padding='SAME') # shape [batch_size, image_size, image_size, depth]
        relu_1 = tf.nn.relu(conv_1 + b1)
        max_pool_1 = tf.nn.max_pool(relu_1, [1,2,2,1], [1,2,2,1], padding='SAME') # shape [batch_size, image_size/2, image_size/2, depth]

        conv_2 = tf.nn.conv2d(max_pool_1, W2, [1, 1, 1, 1], padding='SAME')
        relu_2 = tf.nn.relu(conv_2 + b2)
        max_pool_2 = tf.nn.max_pool(relu_2, [1,2,2,1], [1,2,2,1], padding='SAME') # shape [batch_size, image_size/4, image_size/4, depth*2]
        deconv_1 = tf.nn.conv2d_transpose(value=max_pool_2,filter=W3, output_shape=[-1, image_size//2, image_size//2, depth],strides=[1,2,2,1])
        relu_3 = tf.nn.relu(deconv_1 + b3)

        deconv_2=tf.nn.conv2d_transpose(value=deconv_1,filter=W4, output_shape=[-1, image_size, image_size, num_labels],strides=[1,2,2,1])
        return tf.nn.relu(deconv_2 + b4)


# Train
    logits = model(X)
    loss_train = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))

    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss_train)

    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))

    num_steps = 100

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print('Initialized')
        for step in range(num_steps):
            ind = np.random.random_integers(0, train_num-1, batch_size)
            batch_data = train_dataset[ind, :, :, :]
            batch_labels = train_labels[ind, :]
            feed_dict = {X: batch_data, y: batch_labels}
            _, l, predictions = session.run(
                [optimizer, loss_train, train_prediction], feed_dict=feed_dict)
            if (step % 10 == 0):
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                print('Validation accuracy: %.1f%%' % accuracy(
                    valid_prediction.eval(),valid_labels))
        print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(),test_labels))



