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

    def train(self, dataPipe, batchSize, numSteps, printInterval, logPath, serialNum):

        self.to_string(batchSize, numSteps, printInterval)

        with tf.Session() as session:

            tf.global_variables_initializer().run()
            train_writer = tf.summary.FileWriter(logPath, session.graph)
            saver = tf.train.Saver()
            logging.info('Session begun\n')

            self.numEpoches = len(dataPipe.trainSamples) // batchSize
            startTime = time.time()
            diceValList = []
            dicetrainList = []

            for step in range(numSteps):

                if step % self.numEpoches == 0:
                    logging.info("######## Epoch number {:} ########\n".format(int(step / (len(dataPipe.trainSamples) // batchSize))))
                    dataPipe.initBatchStackCopy()

                batchData, batchLabels = dataPipe.nextBatchFromPermutation(batchSize)
                feed_dict = {self.net.X: batchData, self.net.Y: batchLabels, self.net.isTrain: True}

                _, loss, predictions, summary = session.run(
                    [self.net.optimizer, self.net.loss, self.net.predictions, self.net.merged_loss],feed_dict=feed_dict)

                if step % printInterval == 0:
                    train_writer.add_summary(summary, step)
                    epochAccuracy = accuracy(predictions, batchLabels)
                    epochDice = diceScore(predictions, batchLabels)
                    logging.info("++++++ Iteration number {:} ++++++".format(step))
                    logging.info('Minibatch Loss : {:.4f}'.format(loss))
                    logging.info('Training Accuracy : {:.4f}'.format(epochAccuracy))
                    logging.info('Dice score: {:.4f}\n'.format(epochDice))
                    dicetrainList.append(epochDice)

                # Early stop - train loss is not convarging

                # Break if train dice is 0
                if len(dicetrainList) >= 4 and (all(not(dicetrainList[-1:-4]))):
                    break

                # Break if train dice is 0
                # if (len(dicetrainList) >= 4) and (dicetrainList[-1] < dicetrainList[-2]) and (dicetrainList[-2] < dicetrainList[-3]):
                #     break

                # print validation data
                if 'printValidation' in self.argsDict.keys() and self.argsDict['printValidation']:
                    if step % self.argsDict['printValidation'] == 0 and step:

                        valPredictionList = []
                        valBatchSize = 128
                        sizeValArray = np.shape(dataPipe.valSamples)[0]

                        for valBatchInd in range(0, sizeValArray, valBatchSize):

                            feed_dict = {self.net.X: dataPipe.valSamples[valBatchInd:valBatchInd+valBatchSize],
                                         self.net.Y: dataPipe.valLabels[valBatchInd:valBatchInd+valBatchSize], self.net.isTrain: False}

                            valBatchPredictions = session.run(self.net.predictions, feed_dict=feed_dict)
                            valPredictionList.append(valBatchPredictions)

                        valPredictions = np.concatenate(valPredictionList, axis=0)
                        accuracyVal = accuracy(valPredictions, dataPipe.valLabels)
                        diceVal = diceScore(valPredictions, dataPipe.valLabels)
                        diceValList.append(diceVal)

                        logging.info("++++++ Validation for step num {:} ++++++".format(step))
                        # logging.info('Minibatch Loss : {:.4f}'.format(lossVal))
                        logging.info('Training Accuracy : {:.4f}'.format(accuracyVal))
                        logging.info('Dice score: {:.4f}\n'.format(diceVal))

                # Save checkPoint
                saver.save(session, str(logPath)+"/{}_{}_{}.ckpt".format('per_number_', serialNum, time.strftime('%H_%M__%d_%m_%y')),
                           global_step=self.argsDict['printValidation'], write_meta_graph=False)

                # Early stop - over-fitting
                if (len(diceValList) >= 3) and (diceValList[-1] < diceValList[-2]) and (diceValList[-2] < diceValList[-3]):
                    break

            # test statistics
            testBatchSize = 128
            sizeTestArray = np.shape(dataPipe.testSamples)[0]
            testPredictionList = []

            for testBatchInd in range(0, sizeTestArray, testBatchSize):

                feed_dict = {self.net.X: dataPipe.testSamples[testBatchInd:testBatchInd+testBatchSize],
                             self.net.Y: dataPipe.testLabels[testBatchInd:testBatchInd+testBatchSize], self.net.isTrain: False}

                testBatchPredictions = session.run(self.net.predictions, feed_dict=feed_dict)
                testPredictionList.append(testBatchPredictions)

            testPredictions = np.concatenate(testPredictionList, axis=0)
            accuracyTest = accuracy(testPredictions, dataPipe.testLabels)
            diceTest = diceScore(testPredictions, dataPipe.testLabels)

            logging.info("++++++ Test data +++++++++")
            # logging.info('Minibatch Loss : {:.4f}'.format(meanLossTest))
            logging.info('Training Accuracy : {:.4f}'.format(accuracyTest))
            logging.info('Dice score: {:.4f}\n'.format(diceTest))

            save_path = saver.save(session, str(logPath)+"/{}_{}_{}.ckpt".format('final_save_', serialNum, time.strftime('%H_%M__%d_%m_%y')))

            logging.info('Saving variables in : %s' % save_path)
            with open('model_file.txt', 'a') as file1:
                file1.write(save_path)
                file1.write(' dice={}\n'.format(diceTest))
            endTime = time.time()
            logging.info('Total run time of train is : {0:.4f} min.'.format(round((endTime-startTime)/60, 4)))

            return


def diceScore(predictions, labels):
    eps = 1e-10
    predictions = tf.round(predictions)
    intersection = tf.reduce_sum(tf.multiply(predictions, labels))
    union = eps + tf.reduce_sum(predictions) + tf.reduce_sum(labels)
    res = (2. * intersection) / (union + eps)
    return res.eval()

def accuracy(predictions, labels):
    predictions = tf.round(predictions)
    eq = tf.equal(predictions, labels)
    res = tf.reduce_mean(tf.cast(eq, tf.float32))
    return res.eval()

# class SaveStatistics(object):
#
#     def __init__(self):
#         self.trainDict = {}
#         self.trainDict['loss'] = []
#         self.trainDict['accuracy'] = []
#         self.trainDict['dice'] = []
#         self.valDict = {}
#         self.valDict['loss'] = []
#         self.valDict['accuracy'] = []
#         self.valDict['dice'] = []
#         self.testDict = {}
#
#     def pushTrainStatistics(self, batchLoss, batchAcc, batchDice):
#         self.trainDict['loss'].append(batchLoss)
#         self.trainDict['accuracy'].append(batchAcc)
#         self.trainDict['dice'].append(batchDice)
#
#     def pushValStatistics(self, valLoss, valAcc, valDice):
#         self.valDict['loss'].append(valLoss)
#         self.valDict['accuracy'].append(valAcc)
#         self.valDict['dice'].append(valDice)
#
#     def getNetStatiscits(self, testLoss, testAcc, testDice):
#         # sum the results from train and val data
#
#         self.trainDict['meanLoss'] = round(np.mean(self.trainDict['loss']),4)
#         self.trainDict['meanAccuracy'] = round(np.mean(self.trainDict['accuracy']),4)
#         self.trainDict['meanDice'] = round(np.mean(self.trainDict['dice']),4)
#
#         if len(self.valDict['loss']):
#             self.valDict['meanLoss'] = round(np.mean(self.valDict['loss']),4)
#             self.valDict['meanAccuracy'] = round(np.mean(self.valDict['accuracy']),4)
#             self.valDict['meanDice'] = round(np.mean(self.valDict['dice']),4)
#
#         # a fix for a case where no validation was tested
#         else:
#             self.valDict['meanLoss'] = 0
#             self.valDict['meanAccuracy'] = 0
#             self.valDict['meanDice'] = 0
#
#         self.testDict['loss'] = round(testLoss,4)
#         self.testDict['accuracy'] = round(testAcc,4)
#         self.testDict['dice'] = round(testDice,4)
#
#         return self.trainDict, self.valDict, self.testDict


