#from modelOptimizers import optimizers
from loadData import *
from utils import *
from metrices import *
from resultsDisplay import *
import time
from unetModel import unet_model
import tensorflow as tf
import numpy as np
import os


# CONSTANTS:

IMAGE_SIZE = 128
NUM_CHANNELS = 1
NUM_LABELS = 1
BATCH_SIZE = 16
KERNEL_LIST = [3]
DEPTH_LIST = [32]
LEARNING_RATE = [0.01]
POOL_SIZE = 2
RESTORE = 1
LAYERS=2
PATH='/variables/unet_3_200218_k=3_lr=0.01_d=32.ckpt'


# LOAD DATA


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

                X = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
                Y = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, NUM_LABELS])
                logits = unet_model(X, layers=2, num_channels=1, num_labels=1, kernel_size=3, depth=32,
                                    pool_size=2)
                softprediction=tf.sigmoid(logits)
                loss_train = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logits))
                # TODO: need to rethink about the loss function, it is not effective
                #Adding the train loss to the graph
                tf.summary.scalar('Train_loss', loss_train)
                merged = tf.summary.merge_all()
                # gradient decent decay
                global_step = tf.Variable(0, trainable=False)
                starter_learning_rate = lrate
                learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 20, 0.96, staircase=True)


                #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_train)
                optimizer = tf.train.AdamOptimizer(0.001).minimize(loss_train)
            num_steps = 600
            collector = MetaDataCollector()  # documentation object
            with tf.Session(graph=graph) as session:
                tf.global_variables_initializer().run()
                train_writer = tf.summary.FileWriter('C:\\Users\\ochayoot\Documents\\GitHub\\Final-Project\\ProjectSrc\\ExternalModules\\network_skeleton\\tensorboard\\graph',session.graph)
                saver = tf.train.Saver()
                print('Initialized')
                # restoring data from model file
                if RESTORE == 1:
                    print('Loading data from {}'.format((PATH)))
                    saver.restore(session, "{}".format((PATH)))
                for step in range(num_steps):
                    ind = np.random.randint(0, train_num - 1, BATCH_SIZE)
                    batch_data = train_dataset[ind, :, :, :]
                    batch_labels = train_labels[ind, :, :, :]
                    feed_dict = {X: batch_data, Y: batch_labels}
                    _, l, logits_out,summary = session.run(
                        [optimizer, loss_train, logits,merged], feed_dict=feed_dict)
                    if (step % 10 == 0):
                        train_writer.add_summary(summary, step)
                        results_display(logits=logits_out,label=train_labels,data=train_dataset,index=ind[0],rindex=0,thresh=0.55)
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
                save_path = saver.save(session, "/variables/{}_{}_{}_k={}_lr={}_d={}.ckpt".format('unet', LAYERS,
                                                                                                          time.strftime(
                                                                                                              '%d%m%y'),
                                                                                                          kernel_size,
                                                                                                          lrate,
                                                                                                          depth))
                print('Saving variables in : %s' % save_path)
                with open('model_file.txt', 'a') as file1:
                    file1.write(save_path)
                    file1.write(' dice={}\n'.format(dice))
