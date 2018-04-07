from UnetModel import *
from UnetModel.scripts.layers import *

class UnetModelClass(object):

    def __init__(self, layers, num_channels, num_labels, image_size,
                 kernel_size, depth, pool_size, costStr, optStr, argsDict = {}):

        logging.info('#### -------- UnetModel object was created -------- ####\n')
        self.dispImage=False
        self.layersTodisplay=argsDict['layersTodisplay']
        self.layers = layers
        self.num_channels = num_channels
        self.num_labels = num_labels
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.depth = depth
        self.pool_size = pool_size
        self.costStr = costStr
        self.optStr = optStr
        self.argsDict = argsDict
            
        self.weights_dict = {}
        self.convd_dict = {}
        self.convu_dict = {}
        self.deconv_dict = {}
        self.concat_dict = {}
        self.max_dict = {}
        self.ndepth = 1
        self.to_string()
        self.logits = self._createNet()
        with self.graph.as_default():
            self.predictions = tf.nn.sigmoid(self.logits)
        self.loss = self._getCost()
        self.optimizer = self._getOptimizer()

    def to_string(self):
        logging.info('UnetModel object properties:')
        logging.info('layers : ' + str(self.layers))
        logging.info('num_channels : ' + str(self.num_channels))
        logging.info('num_labels : ' + str(self.num_labels))
        logging.info('image_size : ' + str(self.image_size))
        logging.info('depth : ' + str(self.depth))
        logging.info('pool_size : ' + str(self.pool_size))
        logging.info('costStr : ' + str(self.costStr))
        logging.info('optStr : ' + str(self.optStr))
        for key, value in self.argsDict.items():
            logging.info(str(key) + ' : ' + str(value))
        logging.info('\n')


    def __del__(self):
        # logging.info('#### -------- UnetModel object was deleted -------- ####\n')
        pass

    def _createNet(self):
        self.graph = tf.Graph()
        with self.graph.as_default():

            # placeholders for training
            self.X = tf.placeholder(dtype=tf.float32, shape=[None, self.image_size, self.image_size, self.num_channels])
            self.Y = tf.placeholder(dtype=tf.float32, shape=[None, self.image_size, self.image_size, self.num_labels])

            # creates weights,convolution self.layers and downs samples
            for l in range(1, self.layers + 2):
                if l == 1:
                    with tf.name_scope('convolution_Down_{}'.format(l)):
                        self.weights_dict['WD1_{}'.format(l)] = weight_variable([self.kernel_size, self.kernel_size,
                                                                            self.num_channels, self.depth])
                        self.weights_dict['WD2_{}'.format(l)] = weight_variable([self.kernel_size, self.kernel_size, self.depth, self.depth])
                        self.weights_dict['b1_{}'.format(l)] = bias_variable([self.depth])
                        self.weights_dict['b2_{}'.format(l)] = bias_variable([self.depth])
                        self.convd_dict['convd1_{}'.format(l)] = conv2d(self.X, self.weights_dict['WD1_{}'.format(l)],
                                                                   self.weights_dict['b1_{}'.format(l)])
                        self.convd_dict['convd2_{}'.format(l)] = conv2d(self.convd_dict['convd1_{}'.format(l)],
                                                                   self.weights_dict['WD2_{}'.format(l)],
                                                                   self.weights_dict['b2_{}'.format(l)])
                    with tf.name_scope('Max_Pool{}'.format(l)):
                        self.max_dict['max_{}'.format(l)] = max_pool(self.convd_dict['convd2_{}'.format(l)], 2)
                else:
                    self.ndepth = self.ndepth * 2

                    with tf.name_scope('convolution_Down_{}'.format(l)):
                        self.weights_dict['WD1_{}'.format(l)] = weight_variable([self.kernel_size, self.kernel_size,
                                                                            int(self.depth * self.ndepth / 2), self.depth * self.ndepth])
                        self.weights_dict['WD2_{}'.format(l)] = weight_variable([self.kernel_size, self.kernel_size,
                                                                            self.depth * self.ndepth, self.depth * self.ndepth])
                        self.weights_dict['b1_{}'.format(l)] = bias_variable([self.depth * self.ndepth])
                        self.weights_dict['b2_{}'.format(l)] = bias_variable([self.depth * self.ndepth])
                        self.convd_dict['convd1_{}'.format(l)] = conv2d(self.max_dict['max_{}'.format(l - 1)],
                                                                   self.weights_dict['WD1_{}'.format(l)],
                                                                   self.weights_dict['b1_{}'.format(l)])
                        self.convd_dict['convd2_{}'.format(l)] = conv2d(self.convd_dict['convd1_{}'.format(l)],
                                                                   self.weights_dict['WD2_{}'.format(l)],
                                                                   self.weights_dict['b2_{}'.format(l)])
                    if l != (self.layers + 1):
                        with tf.name_scope('Max_Pool{}'.format(l)):
                            self.max_dict['max_{}'.format(l)] = max_pool(self.convd_dict['convd2_{}'.format(l)], 2)
                    else:
                        with tf.name_scope('Middle'):
                            self.convu_dict['convu2_{}'.format(l)] = self.convd_dict['convd2_{}'.format(l)]

            # upsampling and weights
            for l in range(self.layers, 0, -1):
                # deconvolution
                with tf.name_scope('deconvolution_{}'.format(l)):
                    self.weights_dict['W_{}'.format(l)] = weight_variable([2, 2,
                                                                      int(self.depth * self.ndepth / 2), self.depth * self.ndepth])
                    self.weights_dict['b_{}'.format(l)] = bias_variable([int(self.depth * self.ndepth / 2)])
                    self.deconv_dict['deconv_{}'.format(l)] = deconv2d(self.convu_dict['convu2_{}'.format(l + 1)],
                                                                  self.weights_dict['W_{}'.format(l)],
                                                                  self.weights_dict['b_{}'.format(l)], self.pool_size)
                    self.concat_dict['conc_{}'.format(l)] = concat(self.convd_dict['convd2_{}'.format(l)],
                                                              self.deconv_dict['deconv_{}'.format(l)])
                with tf.name_scope('convoultion_up_{}'.format(l)):
                    self.weights_dict['WU1_{}'.format(l)] = weight_variable([self.kernel_size, self.kernel_size,
                                                                        self.depth * self.ndepth, int(self.depth * self.ndepth / 2)])
                    self.weights_dict['WU2_{}'.format(l)] = weight_variable([self.kernel_size, self.kernel_size,
                                                                        int(self.depth * self.ndepth / 2), int(self.depth * self.ndepth / 2)])
                    self.weights_dict['b1u_{}'.format(l)] = bias_variable([int(self.depth * self.ndepth / 2)])
                    self.weights_dict['b2u_{}'.format(l)] = bias_variable([int(self.depth * self.ndepth / 2)])
                    self.convu_dict['convu1_{}'.format(l)] = conv2d(self.concat_dict['conc_{}'.format(l)],
                                                               self.weights_dict['WU1_{}'.format(l)],
                                                               self.weights_dict['b1u_{}'.format(l)])
                    self.convu_dict['convu2_{}'.format(l)] = conv2d(self.convu_dict['convu1_{}'.format(l)],
                                                               self.weights_dict['WU2_{}'.format(l)],
                                                               self.weights_dict['b2u_{}'.format(l)])
                self.ndepth = int(self.ndepth / 2)
            #put images in tensorboard
            displaylist=[]
            mergedImages = tf.summary.image('input_image', self.X)
            displaylist.append( mergedImages)
            for l in self.layersTodisplay:
                for k in range(self.convd_dict['convd1_{}'.format(l)].shape[3]-1):
                    tempMerge1=tf.summary.image('convlayer1_{}_{}'.format(l,k), self.convd_dict['convd1_{}'.format(l)][1:2,:,:,k:k+1])
                    displaylist.append(tempMerge1)
                for k in range(self.convd_dict['convd1_{}'.format(l)].shape[3]):
                    tempMerge2=tf.summary.image('convlayer2_{}_{}'.format(l,k), self.convd_dict['convd2_{}'.format(l)][1:2,:,:,k:k+1])
                    displaylist.append(tempMerge2)
            self.merged = tf.summary.merge(displaylist)
            with tf.name_scope('Finel_Layer'):
                Wfc = weight_variable([1, 1, self.depth, self.num_labels])
                bfc = bias_variable([self.num_labels])
                return tf.nn.conv2d(self.convu_dict['convu2_{}'.format(l)], Wfc, strides=[1, 1, 1, 1], padding='SAME') + bfc

    def _getCost(self):
        flat_logits = tf.reshape(self.logits, [-1, self.num_labels])
        flat_labels = tf.reshape(self.Y, [-1, self.num_labels])

        with self.graph.as_default():
            if self.costStr == "softmax":
                    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels))

            elif self.costStr == "sigmoid":
                    if 'weightedSum' in self.argsDict.keys() and self.argsDict['weightedSum']:
                        loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=self.Y, logits=self.logits ,pos_weight=self.argsDict['weightVal']))
                    else:
                        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y, logits=self.logits))

            else:
                logging.info ("Error : Not defined cost function.")

            loss_tensorboard=tf.summary.scalar('Train_loss', loss)
            self.merged_loss=tf.summary.merge([loss_tensorboard])
            return loss

    def _getOptimizer(self):
        learningRate = self.argsDict.pop('learningRate', 0.01)
        with self.graph.as_default():
            if self.optStr == 'adam':
                    optimizer = tf.train.AdamOptimizer(learningRate).minimize(self.loss)

            elif self.optStr == 'momentum':
                    momentum = self.argsDict.pop("momentum", 0.2)
                    optimizer = tf.train.MomentumOptimizer(learning_rate=learningRate, momentum=momentum).minimize(self.loss)
            else:
                logging.info ("Error : Not defined optimizer.")
            return optimizer

    def getLogits(self):
        return self.logits

