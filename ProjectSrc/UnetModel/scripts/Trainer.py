from UnetModel import *
import tensorflow as tf
import time as time


class Trainer(object):
    def __init__(self, net, argsDict):
        logging.info('#### -------- Trainer object was created -------- ####\n')

        self.net = net
        self.argsDict = argsDict

    def __del__(self):
        pass

    def to_string(self, batchSize, numSteps, printInterval):
        logging.info('Trainer object properties:')
        logging.info('batchSize : ' + str(batchSize))
        logging.info('numSteps : ' + str(numSteps))
        logging.info('printInterval : ' + str(printInterval))
        for key, value in self.argsDict.items():
            logging.info(str(key) + ' : ' + str(value))
        logging.info('\n')

    def train(self, dataPipe, batchSize, numSteps, printInterval, logPath, restore=False, restorePath=''):

        self.to_string(batchSize, numSteps, printInterval)

        with tf.Session(graph=self.net.graph) as session:
            tf.global_variables_initializer().run()
            train_writer = tf.summary.FileWriter(logPath, session.graph)
            saver = tf.train.Saver()
            logging.info('Session begun\n')

            # restoring data from model file
            if restore:
                logging.info('Loading data from {}'.format((restorePath)))
                saver.restore(session, "{}".format((restorePath)))

            self.numEpoches = len(dataPipe.trainSamples) // batchSize

            for step in range(numSteps):

                if step % (len(dataPipe.trainSamples) // batchSize) == 0:
                    logging.info("######## Epoch number {:} ########\n".format(int(step / (len(dataPipe.trainSamples) // batchSize))))
                    dataPipe.initBatchStackCopy()

                batchData, batchLabels = dataPipe.nextBatchFromPermutation(batchSize)
                feed_dict = {self.net.X: batchData, self.net.Y: batchLabels}

                if step == (numSteps - 1):
                    summary = session.run(self.net.merged, feed_dict=feed_dict)
                    train_writer.add_summary(summary, step)

                else:
                    _, loss, predictions, summary = session.run(
                        [self.net.optimizer, self.net.loss, self.net.predictions, self.net.merged_loss],
                        feed_dict=feed_dict)

                if step % printInterval == 0:
                    train_writer.add_summary(summary, step)
                    epochAccuracy = accuracy(predictions, batchLabels)
                    epochDice = diceScore(predictions, batchLabels)
                    logging.info("++++++ Iteration number {:} ++++++".format(step))
                    logging.info('Minibatch Loss : {:.4f}'.format(loss))
                    logging.info('Training Accuracy : {:.4f}'.format(epochAccuracy))
                    logging.info('Dice score: {:.4f}\n'.format(epochDice))
                    # resultDisplay(predictions=predictions, labels=batchLabels, images=batchData, sampleInd=1,imageSize=240, imageMod=1, thresh=0.5)

                # print validation data
                if 'printValidation' in self.argsDict.keys() and self.argsDict['printValidation']:
                    if step % self.argsDict['printValidation'] == 0:
                        feed_dict = {self.net.X: dataPipe.valSamples, self.net.Y: dataPipe.valLabels}
                        _, lossVal, predictionsVal= session.run(
                            [self.net.optimizer, self.net.loss, self.net.predictions],feed_dict=feed_dict)
                        epochAccuracyVal = accuracy(predictionsVal, dataPipe.valLabels)
                        epochDiceVal = diceScore(predictionsVal, dataPipe.valLabels)
                        logging.info("++++++ Validation for step num {:} ++++++".format(step))
                        logging.info('Minibatch Loss : {:.4f}'.format(lossVal))
                        logging.info('Training Accuracy : {:.4f}'.format(epochAccuracyVal))
                        logging.info('Dice score: {:.4f}\n'.format(epochDiceVal))

            save_path = saver.save(session, str(logPath)+"/{}_{}_{}_{}.ckpt".format('unet', self.net.layers, self.net.argsDict['weightVal'], time.strftime('%H_%M__%d_%m_%y')))
            logging.info('Saving variables in : %s' % save_path)
            with open('model_file.txt', 'a') as file1:
                file1.write(save_path)
                file1.write(' dice={}\n'.format(epochDice))


def diceScore(predictions, labels):
    eps = 1e-5
    predictions = tf.round(predictions)
    intersection = tf.reduce_sum(tf.multiply(predictions, labels))
    union = eps + tf.reduce_sum(predictions) + tf.reduce_sum(labels)
    res = 2 * intersection / (union + eps)
    return res.eval()

def accuracy(predictions, labels):
    predictions = tf.round(predictions)
    eq = tf.equal(predictions, labels)
    res = tf.reduce_mean(tf.cast(eq, tf.float32))
    return res.eval()




