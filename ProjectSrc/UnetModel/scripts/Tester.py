from UnetModel import *
from skimage.transform import resize
import tensorflow as tf
import time as time
class Tester(object):
    def __init__(self, net,testList=[], argsDict={'mod':[1,3]}):
        logging.info('#### -------- Tester object was created -------- ####\n')
        self.net = net
        self.testList = testList
        self.argsDict = argsDict

    def __del__(self):
        # logging.info('#### -------- Tester object was deleted -------- ####\n')
        pass

    def test(self, dataPipe, restorePath='/UnetModel/runData/RunFolder_23_13__21_03_18/unet_3_13_23_19__21_03_18.ckpt'):
        with tf.Session(graph=self.net.graph) as session:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()
            logging.info('Initialized')

            # restoring data from model file
            logging.info('Loading data from {}'.format((restorePath)))
            saver.restore(session, "{}".format((restorePath)))
            Dicelist = []
            for item in self.testList:
                starttime = time.time()
                batchData, batchLabels = dataPipe.next_image(item)
                predictionlist=[]
                batchLabellist=[]
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
                predictionscheck= np.reshape(predictionscheck,(-1,self.net.image_size,self.net.image_size,1))
                batchLabelcheck= np.reshape(batchLabelcheck,(-1, self.net.image_size,self.net.image_size ,1))
                endtime = time.time()
                logging.info('Total example time={}'.format(endtime - starttime))
                epochDice = diceScore(predictionscheck, batchLabelcheck)
                Dicelist.append(epochDice)

                logging.info('Dice={}\n'.format(epochDice))
                while (True):
                    index = input('\nFor 3d viewer press V\nFor next example press Q:\n')
                    if index == 'Q':
                        break
                    elif index == 'V':
                        modality = input('Please enter modality to view from the list {}\n'
                                         '0=T1 ,1=T2 ,2=T1g,3=Flair :'.format(dataPipe.modalityList))
                        modview = batchData[0:predictionscheck.shape[0], :, :, int(modality)]
                        slidesViewer(modview, predictionscheck[:, :, :, 0], batchLabelcheck[:, :, :, 0])
                        plt.show()
                    elif index == 'F':
                        ax, fig = make_ax()
                        img3d = predictionscheck[:, :, :, 0]
                        img3d = (img3d - np.min(img3d)) / (np.max(img3d) - np.min(img3d))
                        img3d = resize(img3d, (img3d.shape[0] // 2, img3d.shape[1] // 2, img3d.shape[2] // 2),
                                       mode='constant')
                        ax.voxels(img3d, facecolors='#1f77b430', edgecolors='gray')
                        plt.show()
                    else:
                        print('Wrong option, please try again:\n')
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
