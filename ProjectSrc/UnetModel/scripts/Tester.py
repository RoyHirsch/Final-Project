from UnetModel import *

class Tester(object):
    def __init__(self, net,testList=[], argsDict={'mod':[1,3]}):
        logging.info('')
        logging.info('#### -------- Tester object was created -------- ####\n')
        self.net = net
        self.testList = testList
        self.argsDict = argsDict


    def __del__(self):
        # logging.info('#### -------- Tester object was deleted -------- ####\n')
        pass

    def test(self, dataPipe, logPath, restorePath=''):
        with tf.Session(graph=self.net.graph) as session:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()
            logging.info('Initialized')

            # restoring data from model file
            logging.info('Loading data from {}'.format((restorePath)))
            saver.restore(session, "{}".format((restorePath)))
            for item in self.testList:
                starttime = time.time()
                batchData, batchLabels = dataPipe.next_image(item)
                predictionlist=[]
                batchLabellist=[]
                Dicelist=[]
                for j in range(0,batchData.shape[0],16):
                    batchDatatemp=batchData[j:j+16,:,:,:]
                    batchLabelstemp=batchLabels[j:j+16,:,:]

                    feed_dict = {self.net.X: batchDatatemp, self.net.Y: batchLabelstemp}
                    predictions = session.run([self.net.predictions],feed_dict=feed_dict)
                    predictionstemp=predictions[0]
                    predictionlist.append(predictionstemp)
                    batchLabellist.append(batchLabelstemp)
                predictionscheck = np.array(predictionlist)
                batchLabelcheck=np.array(batchLabellist)
                predictionscheck= np.reshape(predictionscheck,(-1,240,240,1))
                batchLabelcheck= np.reshape(batchLabelcheck,(-1,240,240,1))
                endtime = time.time()
                logging.info('Total example time={}'.format(endtime - starttime))
                epochDice = diceScore(predictionscheck, batchLabelcheck)
                Dicelist.append(epochDice)
                maxindex=batchLabelcheck.shape[0]

                logging.info('Dice={}\n'.format(epochDice))
                while (True):
                    index = input('\nEnter slice number to view, for next example press Q: ')
                    if index == 'Q':
                        break
                    elif 0 < int(index) and int(index) < maxindex:
                        resultDisplay(predictions=predictionscheck, labels=batchLabelcheck, images=batchData, sampleInd=int(index),
                                      imageSize=240, imageMod=1, thresh=0.5)
                    else:
                        logging.info('Number not in range, please select number < {}'.format(maxindex))
            logging.info('Mean Dice={}'.format(np.mean(np.array(Dicelist))))


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
