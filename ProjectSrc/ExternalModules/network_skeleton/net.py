from ExternalModules.network_skeleton.loadData import *
from ExternalModules.network_skeleton.layers import *
from ExternalModules.network_skeleton.utils import *
import time

import tensorflow as tf
#import tensorboard as tb
import numpy as np
import os

# CONSTANTS:

IMAGE_SIZE = 128
NUM_CHANNELS = 1
NUM_LABELS = 1
BATCH_SIZE = 16
KERNEL_LIST = [3, 5, 7]
DEPTH_LIST = [16, 32, 64]
LEARNING_RATE = [0.03, 0.01, 0.001]
POOL_SIZE = 2
RESTORE = 0
DATE = None
# LOAD DATA|:


print('Start load data')
data = load_data(os.path.realpath(__file__ + "/../../" + 'toy_segmentaion_data/data'))
label = load_labels(os.path.realpath(__file__ + "/../../" + 'toy_segmentaion_data/labels'))


print('End load data\n')

train_num = 1000
val_num = 1200
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

for kernel_size in KERNEL_LIST:
    for depth in DEPTH_LIST:
        for lrate in LEARNING_RATE:
            graph = tf.Graph()
            with graph.as_default():
                valid_dataset = tf.constant(valid_dataset, dtype=tf.float32)
                test_dataset = tf.constant(test_dataset, dtype=tf.float32)
                ### ---------- Model ----------- ####

                X = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
                Y = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, NUM_LABELS])

                def unet_model(X, unet_num_layers=3):
                    W1_1 = weight_variable([kernel_size, kernel_size, NUM_CHANNELS, depth])
                    b1_1 = bias_variable([depth])
                    W1_2 = weight_variable([kernel_size, kernel_size, depth, depth])
                    b1_2 = bias_variable([depth])

                    conv1_1 = conv2d(X, W1_1, b1_1)
                    conv1_2 = conv2d(conv1_1, W1_2, b1_2)
                    max1 = max_pool(conv1_2, 2)  # [N, IMAGE_SIZE // 2, IMAGE_SIZE // 2, DEPTH]

                    # unet_layer_2
                    W2_1 = weight_variable([kernel_size, kernel_size, depth, depth * 2])
                    b2_1 = bias_variable([depth * 2])
                    W2_2 = weight_variable([kernel_size, kernel_size, depth * 2, depth * 2])
                    b2_2 = bias_variable([depth * 2])

                    conv2_1 = conv2d(max1, W2_1, b2_1)
                    conv2_2 = conv2d(conv2_1, W2_2, b2_2)
                    max2 = max_pool(conv2_2, 2)  # [N, IMAGE_SIZE // 4, IMAGE_SIZE // 4, DEPTH * 2]

                    # unet_layer_3
                    W3_1 = weight_variable([kernel_size, kernel_size, depth * 2, depth * 4])
                    b3_1 = bias_variable([depth * 4])
                    W3_2 = weight_variable([kernel_size, kernel_size, depth * 4, depth * 4])
                    b3_2 = bias_variable([depth * 4])

                    conv3_1 = conv2d(max2, W3_1, b3_1)
                    conv3_2 = conv2d(conv3_1, W3_2, b3_2)
                    max3 = max_pool(conv3_2, 2)  # [N, IMAGE_SIZE // 8, IMAGE_SIZE // 8, DEPTH * 4]

                    # unet_layer_4
                    W4_1 = weight_variable([kernel_size, kernel_size, depth * 4, depth * 8])
                    b4_1 = bias_variable([depth * 8])
                    conv4_1 = conv2d(max3, W4_1, b4_1)
                    W4_2 = weight_variable([kernel_size, kernel_size, depth * 8, depth * 8])
                    b4_2 = bias_variable([depth * 8])
                    conv4_2 = conv2d(conv4_1, W4_2, b4_2)

                    # upsample

                    # unet_upsample_layer_1 (deconve - concat - conv - conv)
                    WD1_1 = weight_variable_devonc([POOL_SIZE, POOL_SIZE, depth * 4, depth * 8])
                    bd1_1 = bias_variable([depth * 4])
                    dconv1_1 = deconv2d(conv4_2, WD1_1, bd1_1, POOL_SIZE)
                    concat1 = concat(conv3_2, dconv1_1)

                    WD1_2 = weight_variable([kernel_size, kernel_size, depth * 8, depth * 4])
                    bd1_2 = bias_variable([depth * 4])
                    WD1_3 = weight_variable([kernel_size, kernel_size, depth * 4, depth * 4])
                    bd1_3 = bias_variable([depth * 4])
                    dconv1_2 = conv2d(concat1, WD1_2, bd1_2)
                    dconv1_3 = conv2d(dconv1_2, WD1_3, bd1_3)

                    # unet_upsample_layer_2
                    WD2_1 = weight_variable_devonc([POOL_SIZE, POOL_SIZE, depth * 2, depth * 4])
                    bd2_1 = bias_variable([depth * 2])
                    dconv2_1 = deconv2d(dconv1_3, WD2_1, bd2_1, POOL_SIZE)
                    concat2 = concat(conv2_2, dconv2_1)

                    WD2_2 = weight_variable([kernel_size, kernel_size, depth * 4, depth * 2])
                    bd2_2 = bias_variable([depth * 2])
                    WD2_3 = weight_variable([kernel_size, kernel_size, depth * 2, depth * 2])
                    bd2_3 = bias_variable([depth * 2])
                    dconv2_2 = conv2d(concat2, WD2_2, bd2_2)
                    dconv2_3 = conv2d(dconv2_2, WD2_3, bd2_3)

                    # unet_upsample_layer_3
                    WD3_1 = weight_variable_devonc([POOL_SIZE, POOL_SIZE, depth, depth * 2])
                    bd3_1 = bias_variable([depth])
                    dconv3_1 = deconv2d(dconv2_3, WD3_1, bd3_1, POOL_SIZE)
                    concat3 = concat(conv1_2, dconv3_1)

                    WD3_2 = weight_variable([kernel_size, kernel_size, depth * 2, depth])
                    bd3_2 = bias_variable([depth])
                    WD3_3 = weight_variable([kernel_size, kernel_size, depth, depth])
                    bd3_3 = bias_variable([depth])
                    dconv3_2 = conv2d(concat3, WD3_2, bd3_2)
                    dconv3_3 = conv2d(dconv3_2, WD3_3, bd3_3)

                    Wfc = weight_variable([1, 1, depth, NUM_LABELS])
                    bfc = bias_variable([NUM_LABELS])
                    return tf.nn.conv2d(dconv3_3, Wfc, strides=[1, 1, 1, 1], padding='SAME') + bfc


                logits = unet_model(X)

                loss_train = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logits))
                # TODO: need to rethink about the loss function, it is not effective

                # gradient decent decay
                global_step = tf.Variable(0, trainable=False)
                starter_learning_rate = lrate
                learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 20, 0.96, staircase=True)
                optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_train)
                # optimizer = tf.train.AdamOptimizer(0.01).minimize(loss_train)
                valid_logits = unet_model(valid_dataset)
                test_logits = unet_model(test_dataset)


                def diceScore(logits, labels):
                    eps = 1e-5

                    prediction = tf.round(tf.nn.sigmoid(logits))
                    intersection = tf.reduce_sum(tf.multiply(prediction, labels))
                    union = eps + tf.reduce_sum(prediction) + tf.reduce_sum(labels)
                    res = 2 * intersection / (union + eps)
                    return res.eval()


                def accuracy(logits, labels):
                    predictions = tf.round(tf.nn.sigmoid(logits))
                    eq = tf.equal(predictions, labels)
                    res = tf.reduce_mean(tf.cast(eq, tf.float32))
                    return res.eval()

            num_steps = 700

            collector = MetaDataCollector()  # documentation object
            with tf.Session(graph=graph) as session:
                tf.global_variables_initializer().run()
                saver = tf.train.Saver()
                print('Initialized')
                # restoring data from model file
                if RESTORE == 1:
                    if DATE == None:
                        DATE = time.strftime('%d%m%y')
                    print('Loading data from {}'.format("/variables/{}_{}_{}_.ckpt".format('unet', 3, DATE)))
                    saver.restore(session, "/variables/{}_{}_{}_.ckpt".format('unet', 3, time.strftime('%d%m%y')))
                for step in range(num_steps):
                    ind = np.random.randint(0, train_num - 1, BATCH_SIZE)
                    batch_data = train_dataset[ind, :, :, :]
                    batch_labels = train_labels[ind, :, :, :]
                    feed_dict = {X: batch_data, Y: batch_labels}
                    _, l, logits_out = session.run(
                        [optimizer, loss_train, logits], feed_dict=feed_dict)
                    if (step % 20 == 0):
                        print('**** Minibatch step: %d ****' % step)
                        print('Loss: {}'.format(round(l, 4)))
                        trainAcc = accuracy(logits_out, batch_labels)
                        print('Accuracy: %.3f % %' % trainAcc)
                        dice = diceScore(logits_out, batch_labels)
                        print("lr={} ker={} depth{}".format(lrate, kernel_size, depth))
                        print('Dice: %.3f % %\n' % dice)
                        # valAcc = accuracy(valid_logits.eval(), valid_labels)
                        # print('Validation accuracy: %.3f%%\n' % valAcc)
                        collector.getStepValues(l, trainAcc, trainAcc)
                save_path = saver.save(session, "/variables/{}_{}_{}_k={}_lr={}_d={}_dice={}.ckpt".format('unet', 3,
                                                                                                          time.strftime(
                                                                                                              '%d%m%y'),
                                                                                                          kernel_size,
                                                                                                          lrate,
                                                                                                          depth,
                                                                                                          dice))
                print('Saving variables in : %s' % save_path)
                print('Test accuracy: %.3f%%' % accuracy(test_logits.eval(), test_labels))
                with open('model_file.txt', 'a') as file1:
                    file1.write(save_path)
