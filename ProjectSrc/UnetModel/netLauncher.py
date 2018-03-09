from Utilities.DataPipline import *
from UnetModel.utils import *
from UnetModel.metrices import *
from UnetModel.resultsDisplay import *
from UnetModel.unetModel import unet_model

import time
import tensorflow as tf
import numpy as np
import os


# CONSTANTS:

IMAGE_SIZE = 240
NUM_CHANNELS = 3
NUM_LABELS = 1
BATCH_SIZE = 2
KERNEL_SIZE = 3
DEPTH = 32
LEARNING_RATE = 0.01
POOL_SIZE = 2
RESTORE = 0
LAYERS = 2
PATH='/variables/unet_3_200218_k=3_lr=0.01_d=32.ckpt'
LOG_DIR = os.path.realpath(__file__ + "/../" + "/tensorboard")
NUM_STEPS = 600

# LOAD DATA
dataPipe = DataPipline(numTrain=5, numVal=1, numTest=4, modalityList=[1,2,3],
                     optionsDict={'zeroPadding': True, 'paddingSize': 240, 'normalize': True,
                                  'normType': 'reg', 'binaryLabel': True})

graph = tf.Graph()
with graph.as_default():

    X = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
    Y = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, NUM_LABELS])
    logits = unet_model(X, layers=3, num_channels=3, num_labels=1, kernel_size=3, depth=32,
                        pool_size=2)
    loss_train = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logits))

    #Adding the train loss to the graph
    tf.summary.scalar('Train_loss', loss_train)
    merged = tf.summary.merge_all()

    # gradient decent decay
    # global_step = tf.Variable(0, trainable=False)
    # starter_learning_rate = LEARNING_RATE
    # learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 20, 0.96, staircase=True)

    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss_train)

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    train_writer = tf.summary.FileWriter(LOG_DIR ,session.graph)
    saver = tf.train.Saver()
    print('Initialized')

    # restoring data from model file
    if RESTORE == 1:
        print('Loading data from {}'.format((PATH)))
        saver.restore(session, "{}".format((PATH)))

    for step in range(NUM_STEPS):
        batchData, batchLabels = dataPipe.next_train_random_batch(BATCH_SIZE)
        feed_dict = {X: batchData, Y: batchLabels}
        _, l, logits_out, summary = session.run(
            [optimizer, loss_train, logits, merged], feed_dict=feed_dict)
        if (step % 2 == 0):
            train_writer.add_summary(summary, step)
            # results_display(logits_out,train_labels,train_dataset,ind[0],0,0.55, IMAGE_SIZE)
            print('**** Minibatch step: %d ****' % step)
            print('Loss: {}'.format(round(l, 4)))
            trainAcc = accuracy(logits_out, batchLabels)
            print('Accuracy: %.3f % %' % trainAcc)
            dice = diceScore(logits_out, batchLabels)
            print('Dice: %.3f % %\n' % dice)
            # valAcc = accuracy(valid_logits.eval(), valid_labels)
            # print('Validation accuracy: %.3f%%\n' % valAcc)
    save_path = saver.save(session, "/variables/{}_{}_{}_k={}_lr={}_d={}.ckpt".format('unet', LAYERS,
                                                                                              time.strftime(
                                                                                                  '%d%m%y'),
                                                                                              KERNEL_SIZE,
                                                                                              LEARNING_RATE,
                                                                                              DEPTH))
    print('Saving variables in : %s' % save_path)
    with open('model_file.txt', 'a') as file1:
        file1.write(save_path)
        file1.write(' dice={}\n'.format(dice))

# Roy: call for tensorboard
# python3 -m tensorboard.main --logdir /Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/UnetModel/tensorboard