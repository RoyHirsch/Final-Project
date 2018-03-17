from Utilities.loadData import *
from UnetModel import *
import skimage.transform as ski
import os
import pprint

MOD_LIST = ['T1', 'T2', 'T1g', 'FLAIR']
MAX_SAMPLES = 30
MAX_SIZE = 240
ROOT_DIR = os.path.realpath(__file__ + "/../../")


class DataPipline(object):

    batch_offset = 0
    optionsDict = {}

    def __init__(self, numTrain, numVal, numTest, modalityList, permotate, optionsDict):

        '''

        class object for holding and managing all the data for the net train and testing.

        PARAMS:
            numTrain: number of samples to use in train dataset
            numVal: number of samples to use in val dataset
            numTest: number of samples to use in test dataset
            modalityList: list of numbers to represent the number of channels/modalities to use
            MOD_LIST = ['T1', 'T2', 'T1g', 'FLAIR'] represented as: [0,1,2,3]

            optionsDict: additional options dictionary:
               'zeroPadding': bool
               'paddingSize': int
               'normalize': bool
               'normType': ['reg', 'clip']
               'binaryLabels': bool - for flattening the labels into binary classification problem
               'resize': bool
               'newSize': int - new image size for resize
               'filterSlices': bool
               'minParentageLabeledVoxals': int - for filtering slices, parentage in [0,1]
        '''
        logging.info('')
        logging.info('#### -------- DataPipline object was created -------- ####\n')

        self.trainNumberList = []
        self.valNumberList = []
        self.testNumberList = []
        self.batchesDict = {}
        self.modalityList = modalityList
        self.optionsDict = optionsDict
        if permotate:
            self._permotate_samples(numTrain, numVal, numTest)
        else:
            self._manual_samples(numTrain, numVal, numTest)
        self.get_samples_list()

    def __del__(self):
        # logging.info('#### -------- DataPipline object was deleted -------- ####\n')
        pass

    def _permotate_samples(self, numTrain, numVal, numTest):
        '''
            randomly selects the data samples to each list.
        '''

        list = np.random.permutation(MAX_SAMPLES).tolist()
        for _ in range(numTrain):
            self.trainNumberList.append(list.pop())
        for _ in range(numVal):
            self.valNumberList.append(list.pop())
        for _ in range(numTest):
            self.testNumberList.append(list.pop())

    def _manual_samples(self, numTrain, numVal, numTest):
        '''
            randomly selects the data samples to each list.
        '''
        self.trainNumberList=list(range(0, numTrain))
        self.valNumberList=[numVal]
        self.testNumberList=list(range(numTrain+numVal, numTrain+numVal+numTest))


    def _normalize_image_modality(self, imgMod):
        # mean 0 and std 1 per image
        if self.optionsDict['normType'] == 'clip':
            b, t = np.percentile(imgMod, (0.5, 99.5))
            imgMod = np.clip(imgMod, b, t)
            mean = np.mean(imgMod)
            var = np.std(imgMod)
            normImg = (imgMod - mean) / var

        elif self.optionsDict['normType'] == 'reg':
            mean = np.mean(imgMod)
            var = np.std(imgMod)
            normImg = (imgMod - mean) / var
        # between 0-1
        elif self.optionsDict['normType'] == 'zeroToOne':
            normImg = imgMod / np.max(imgMod)

        return normImg

    def _normalize_image(self, img):
        normImg = np.zeros(np.shape(img))
        H, W, D, C = np.shape(img)
        for i in range(0, C):
            normImg[:, :, :, i] = self._normalize_image_modality(img[:, :, :, i])
        return normImg

    def _zero_padding_img(self, maxSize, img):
        [H, W, D, C] = np.shape(img)
        if (H == maxSize) and (W == maxSize):
            return img
        else:
            hOffset = int((maxSize - H) / 2)
            wOffset = int((maxSize - W) / 2)
            paddedImg = np.zeros([maxSize, maxSize, D, C])
            paddedImg[hOffset:H + hOffset, wOffset:W + wOffset, :, :] = img
            return paddedImg

    def _zero_padding_label(self, label):
        [H, W, D] = np.shape(label)
        maxSize = self.optionsDict['paddingSize']

        if (H == maxSize) and (W == maxSize):
            return label
        else:
            hOffset = int((maxSize - H) / 2)
            wOffset = int((maxSize - W) / 2)
            paddedLabel = np.zeros([maxSize, maxSize, D])
            paddedLabel[hOffset:H + hOffset, wOffset:W + wOffset, :] = label
            return paddedLabel

    def _crop_iamge(self, img, maxH, maxW):
        H, W, D, C = np.shape(img)
        diffH = round((H - maxH) / 2)
        diffW = round((W - maxW) / 2)
        cropedImg = img[diffH:H - diffH, diffW:W - diffW, :, :]
        return cropedImg

    def _resize_image(self, img):
        newSize = self.optionsDict['newSize']
        H, W, D, C = np.shape(img)
        resizeImg = np.zeros([newSize, newSize, D, C])
        for i in range(D):
            resizeImg[:,:,i,:] = ski.resize(img[:,:,i,:], [newSize,newSize,C], mode='constant')
        return resizeImg

    def _resize_label(self, label):
        newSize = self.optionsDict['newSize']
        H, W, D = np.shape(label)
        resizeLabel = np.zeros([newSize, newSize, D])
        for i in range(D):
            resizeLabel[:, :, i] = ski.resize(label[:, :, i], [newSize, newSize], mode='constant')
        return resizeLabel

        # ---- Prepare Lists ---- #

    def pre_process_list(self, listName='train',num=-1):
        '''
            Processing a list of samples (may be train, val or test list)
            This funcrion gets the optionsDist and preforms all the pre-processing on the data.
            THe output is [outSampleArray, outLabelArray] , 4D and 3D arrays containing the pre-processed data.
        '''
        if num!=-1:
            numbersList=[num]
            self.optionsDict['filterSlices']=False
        elif listName == 'train':
            numbersList = self.trainNumberList
        elif listName in ['val','validation']:
            numbersList = self.valNumberList
        elif listName == 'test':
            numbersList = self.testNumberList
        else:
            logging.info('Error while calling pre_process_list')

        outSampleArray = []
        outLabelArray = []
        for i in numbersList:
            img = self.data[i][:, :, :, self.modalityList]
            label = self.labels[i]

            if 'binaryLabels' in self.optionsDict.keys() and self.optionsDict['binaryLabels']:
                self.numOfLabels = 1
                tmpLabel = np.zeros_like(label)
                tmpLabel[label != 0] = 1
                label = tmpLabel

            else:
                self.numOfLabels = 4

            if 'zeroPadding' in self.optionsDict.keys() and self.optionsDict['zeroPadding']:
                img = self._zero_padding_img(self.optionsDict['paddingSize'], img)
                label = self._zero_padding_label(label)

            if 'resize' in self.optionsDict.keys() and self.optionsDict['resize']:
                img = self._resize_image(img)
                label = self._resize_label(label)

            if 'normalize' in self.optionsDict.keys() and self.optionsDict['normalize']:
                img = self._normalize_image(img)

            H, W, D, C = np.shape(img)

            for j in range(0, D):
                if 'filterSlices' in self.optionsDict.keys() and self.optionsDict['filterSlices']:
                    labeledVoxals = np.sum(np.not_equal(label[:, :, j], 0))
                    parcentageLabeledVoxals = labeledVoxals / H*W
                    if self.optionsDict['minParentageLabeledVoxals'] < parcentageLabeledVoxals:
                        outSampleArray.append(img[:, :, j, :])
                        outLabelArray.append(label[:, :, j])
                else:
                    outSampleArray.append(img[:, :, j, :])
                    outLabelArray.append(label[:, :, j])

        # a fix to locate the D dimension in it's place
        # outSampleArray = np.swapaxes(outSampleArray, 0, 2)
        # outLabelArray = np.swapaxes(outLabelArray, 0, 2)

        imageSize = np.shape(outSampleArray)[1]
        outSampleArray = np.array(outSampleArray).astype(np.float32)

        # reshape to fit tensorflow constrains
        outLabelArray = np.reshape(outLabelArray, [-1, imageSize, imageSize, 1]).astype(np.float32)

        return outSampleArray, outLabelArray

    def get_samples_list(self):
        '''
            Main function for data loading.
            Loads all the data from the directory.
            Creates the train, val and test samples lists
        '''

        self.trainSamples = []
        self.trainLabels = []
        self.valSamples = []
        self.valLabels = []
        self.testSamples = []
        self.testLabels = []

        self.data, self.labels = get_data_and_labels_from_folder()
        logging.info('Data and labels were uploaded successfully.')
        self.trainSamples, self.trainLabels = self.pre_process_list(listName='train')
        logging.info('Train samples list processed successfully.')
        self.valSamples, self.valLabels = self.pre_process_list(listName='val')
        logging.info('Validation samples list processed successfully.')
        self.testSamples, self.testLabels = self.pre_process_list(listName='test')
        logging.info('Train, val and test database created successfully.')

        # logging.infoings for debug:
        logging.info('Train dataset, samples number: ' + str(self.trainNumberList))
        logging.info('Shape of train dataset: ' + str(np.shape(self.trainSamples)))
        logging.info('Val dataset, samples number: ' + str(self.valNumberList))
        logging.info('Shape of val dataset: ' + str(np.shape(self.valSamples)))
        logging.info('Test dataset, samples number: ' + str(self.testNumberList))
        logging.info('Shape of test dataset: ' + str(np.shape(self.testSamples)))

    # ---- Getters ---- #

    def to_string_pipline(self):
        logging.info('\n\nPipline object properties:\n')
        logging.info('Train dataset, samples number: ' + str(self.trainNumberList) + '\n' +
              'Shape of train dataset: ' + str(np.shape(self.trainSamples)) + '\n' +
              'Shape of train labe ls: ' + str(np.shape(self.trainLabels)))
        logging.info('Validation dataset, samples number: ' + str(self.valNumberList) + '\n' +
              'Shape of val dataset: ' + str(np.shape(self.valSamples)) + '\n' +
              'Shape of val labels: ' + str(np.shape(self.valLabels)))
        logging.info('Test dataset, samples number: ' + str(self.testNumberList) + '\n' +
              'Shape of test dataset: ' + str(np.shape(self.testSamples)) + '\n' +
              'Shape of test labels: ' + str(np.shape(self.testLabels)))
        logging.info('\nPipline object parameters:\n"')
        plogging.info.plogging.info(self.optionsDict)

    def get_train_dataset_and_labels(self):
        return self.trainSamples, self.trainLabels

    def get_val_dataset_and_labels(self):
        return self.valSamples, self.valLabels

    def get_test_dataset_and_labels(self):
        return self.testSamples, self.testLabels

    def reset_train_batch_offset(self, offset = 0):
        self.batch_train_offset = offset

    def next_train_batch(self, batch_size):
        start = self.batch_offset
        end = start + batch_size
        self.batch_offset = end

        return self.trainSamples[:, :, range(start, end), :], self.trainLabels[:, :, range(start, end)]

    def init_batch_number(self):
        self.batchNumer = 0

    def next_train_random_batch(self, batch_size):
        ind = np.random.random_integers(0, np.shape(self.trainSamples)[0]-1, batch_size)
        return self.trainSamples[ind, :, :, :], self.trainLabels[ind, :, :]

# ---- Help Functions ---- #

    @staticmethod
    def print_img_statistics(img):
        modalities = ['T1', 'T2', 'T1g', 'FLAIR']
        for i in range(0, 4):
            logging.info('Image modality: ' + modalities[i] + ': Mean: ' +
                  str(np.mean(img[:, :, :, i])) + ' Variance: ' + str(np.std(img[:, :, :, i])))
            logging.info('Image max: ' + str(np.max(img)) + ' Image min: ' + str(np.min(img)))

    @staticmethod
    def print_histogram(img):

        counts, bins = np.histogram(img.ravel(), bins=255)
        # plt.bar(bins[1:-1],counts[1:])
        plt.bar(bins[:-1], counts)
        plt.show()

    @staticmethod
    def print_multimodal_image(img, slice):
        fig = plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(img[:, :, slice, 0], cmap='gray')
        plt.subplot(2, 2, 2)
        plt.imshow(img[:, :, slice, 1], cmap='gray')
        plt.subplot(2, 2, 3)
        plt.imshow(img[:, :, slice, 2], cmap='gray')
        plt.subplot(2, 2, 4)
        plt.imshow(img[:, :, slice, 3], cmap='gray')
        plt.show()

    def next_image(self,sliceNumber):
        img,labels=self.pre_process_list(listName='train' ,num=sliceNumber)
        return img,labels



# ---- Test Code ---- #

# # declare some data object
# pipObj = DataPipline(numTrain=5, numVal=1, numTest=4, modalityList=[1,2,3],
#                      optionsDict={'zeroPadding': True, 'paddingSize': 240, 'normalize': True, 'normType': 'reg'})
#
# logging.info(np.shape(pipObj.trainLabels))
# logging.info(np.shape(pipObj.trainSamples))
# pipObj.to_string_pipline()
# # branch few train batches
# batch_size = 64
# pipObj.init_batch_number()
# for i in range(4):
#      train_batch_data, train_batch_labels = pipObj.next_train_random_batch(batch_size)
#      logging.info(np.shape(train_batch_data))
#      logging.info(np.shape(train_batch_labels))
