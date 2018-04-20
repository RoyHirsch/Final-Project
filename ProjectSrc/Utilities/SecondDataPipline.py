from Utilities.loadData import *
from UnetModel import *
import skimage.transform as ski
import os


MOD_LIST = ['T1', 'T2', 'T1g', 'FLAIR']
MAX_SAMPLES = 30
MAX_SIZE = 240
ROOT_DIR = os.path.realpath(__file__ + "/../../")


class SecondDataPipline(object):

    batch_offset = 0
    optionsDict = {}

    def __init__(self, numTrain, numVal, numTest, modalityList, permotate, optionsDict):

        '''

        temp class object to feed data into the second classificatino network

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
               'minPerentageLabeledVoxals': float - for filtering slices, parentage in [0,1]
               'percentageOfLabeledData': float - the total percentage of labeled data in the pipline, in range [0,1]

               'catPatch' - bool, True to cut the slices into patches
               'patchSize' - int, must be a dividor of image hieght and width

        '''
        logging.info('')
        logging.info('#### -------- DataPipline object was created -------- ####\n')
        self.trainNumberList = []
        self.valNumberList = []
        self.testNumberList = []
        self.batchesDict = {}
        self.modalityList = modalityList
        self.optionsDict = optionsDict
        self.to_string()

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
            manualy selects the data samples to each list.
        '''
        self.trainNumberList=list(range(0, numTrain))
        self.valNumberList=[numVal]
        self.testNumberList=list(range(numTrain+numVal, numTrain+numVal+numTest))


    def _normalize_image_modality(self, imgMod):
        if self.optionsDict['normType'] == 'clip':
            b, t = np.percentile(imgMod, (0.5, 99.5))
            imgMod = np.clip(imgMod, b, t)
            mean = np.mean(imgMod)
            var = np.std(imgMod)
            normImg = (imgMod - mean) / var

        # standartization
        elif self.optionsDict['normType'] == 'reg':
            mean = np.mean(imgMod)
            var = np.std(imgMod)
            normImg = (imgMod - mean) / var

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

        samplesList = []
        labelList = []
        maskList = []

        for i in numbersList:
            img = self.data[i][:, :, :, self.modalityList]
            label = self.labels[i]

            mask = np.zeros_like(label)
            mask[label != 0] = 1

            self.numOfLabels = 4

            if 'normalize' in self.optionsDict.keys() and self.optionsDict['normalize']:
                img = self._normalize_image(img)

            # filter only the train pipeline
            if listName == 'train' and 'filterSlices' in self.optionsDict.keys() and self.optionsDict['filterSlices']:
                img, label = self.filterTrainDataPipe(img, label)

            else:
                # swap axes of the data array to match TF format
                img = np.swapaxes(img, 0, 2)
                label = np.swapaxes(label, 0, 2)

            self.extractPacthes(img, label, mask)



        return outSampleArray, outLabelArray, outMaskArray

    def get_samples_list(self):
        '''
            Main function for data loading.
            Loads all the data from the directory.
            Creates the train, val and test samples lists
        '''

        self.trainSamples = []
        self.trainLabels = []
        self.trainMasks = []
        self.valSamples = []
        self.valLabels = []
        self.valMasks = []
        self.testSamples = []
        self.testLabels = []
        self.testMasks = []


        self.data, self.labels = get_data_and_labels_from_folder()
        logging.info('Data and labels were uploaded successfully.')
        self.trainSamples, self.trainLabels, self.trainMasks = self.pre_process_list(listName='train')
        logging.info('Train samples list processed successfully.')
        self.valSamples, self.valLabels, self.valMasks = self.pre_process_list(listName='val')
        logging.info('Validation samples list processed successfully.')
        self.testSamples, self.testLabels, self.valMasks = self.pre_process_list(listName='test')
        logging.info('Train, val and test database created successfully.')

        # logging for debug:
        logging.info('Train dataset, samples number: ' + str(self.trainNumberList))
        logging.info('Shape of train dataset: ' + str(np.shape(self.trainSamples)))
        logging.info('Val dataset, samples number: ' + str(self.valNumberList))
        logging.info('Shape of val dataset: ' + str(np.shape(self.valSamples)))
        logging.info('Test dataset, samples number: ' + str(self.testNumberList))
        logging.info('Shape of test dataset: ' + str(np.shape(self.testSamples)) + '\n')

    def zeroPadSlice(self, slice, H, W, patchSize):
        paddedSlice = np.zeros([H + 2*patchSize, W + 2*patchSize])
        paddedSlice[patchSize:H + patchSize, patchSize:H + patchSize] = slice
        return paddedSlice

    def getPatch(self, slice, x, y, patchSize, H, W):
        # extracts a patch from the image, if not enough values - fill with zeros
        pass

    def extractPacthes(self, samplesArray, labelsArray, masksArray):
        N ,H, W, D = np.shape(samplesArray)
        patchSize = self.optionsDict['patchSize']
        halfPatch = patchSize // 2
        patchList = []
        labelList = []
        for sliceNum in range(N):
            numOfPatches = np.sum(np.nonzero(masksArray[sliceNum]))
            if numOfPatches > 0:
                patchesFromSlice = np.zeros([numOfPatches, patchSize, patchSize, D])
                X, Y = np.nonzero(self.zeroPadSlice(masksArray[sliceNum], H, W, patchSize))
                Xtrue, Ytrue = np.nonzero(labelsArray[sliceNum])
                paddedSlice = self.zeroPadSlice(samplesArray[sliceNum, :, :, :], H, W, patchSize)
                for p in numOfPatches:
                    # patch = self.getPatch(self, paddedSlice, X[point], Y[point], patchSize, W, H)
                    patch = paddedSlice[X[p]-halfPatch:X[p]+halfPatch, Y[p]-halfPatch:Y[p]+halfPatch]
                    patchesFromSlice[p, :, :, :] = patch

                    labelList.append(labelsArray[sliceNum, Xtrue[p], Ytrue[p]])
                patchList.append(patchesFromSlice)
        print(np.shape(patchList))






    def getPatchesFromSlices(self, image, label):
        H, W, D, C = np.shape(image)
        if 'patchSize' in self.optionsDict.keys() and self.optionsDict['patchSize']:
            patchSize = self.optionsDict['patchSize']
        else:
            logging.info('Error - no patchSize in optionsDict')
        # imageSize is 240 so patchSize should be a dividor of 240
        if not(H / patchSize):
            logging.info('Error - patchSize is not a dividor of imageSize')

        n = np.shape(image)[0] // patchSize
        ind = 0
        patchArrayImage = np.zeros([patchSize, patchSize, D*(n*n), C])
        patchArrayLabel = np.zeros([patchSize, patchSize, D*(n*n)])
        for i in range(n):
            for j in range(n):
                patchArrayImage[:, :, D*(ind):D*(ind+1), :] = image[patchSize*i:patchSize*(i+1), patchSize*j:patchSize*(j+1), :, :]
                patchArrayLabel[:, :, D*(ind):D*(ind+1)] = label[patchSize*i:patchSize*(i+1), patchSize*j:patchSize*(j+1), :]
                ind += 1
        return patchArrayImage, patchArrayLabel

    def filterTrainDataPipe(self, imagesArray, labelsArray):
        # the function filters the pipline to create a pipline with a specific percentage of labels patches\slices
        # three parameters:
        # filterSlices - determine if the pipline needs to be filtered
        # minPerentageLabeledVoxals - determine if a patch\slice will be 'labeled'
        # percentageOfLabeledData - the total percentage of labeled data in the pipline

        H, W, D, C = np.shape(imagesArray)

        indOfLabeledData = []
        indOfNotLabeledData = []
        for j in range(0, D):
            labeledVoxals = np.sum(np.not_equal(labelsArray[:, :, j], 0))
            parcentageLabeledVoxals = labeledVoxals / H * W
            if self.optionsDict['minPerentageLabeledVoxals'] < parcentageLabeledVoxals:
                indOfLabeledData.append(j) # keep the indexes of data that is 'labeled'
            else:
                indOfNotLabeledData.append(j)  # keep the indexes of data that is not 'labeled'

        # find the number of 'not labeled' data to exclude from the pipline in order to get the desire label percentage
        parentageOfLabeledData = self.optionsDict['percentageOfLabeledData']
        totalNumberOfNotLabeledData = (parentageOfLabeledData * len(indOfLabeledData)) / (1 - parentageOfLabeledData)
        indToInclude = indOfNotLabeledData[0:int(totalNumberOfNotLabeledData)]
        indToInclude = indToInclude + indOfLabeledData

        # filter out the right amount of not labeled data
        outSampleArray = []
        outLableArray = []
        # newInd = 0
        for j in range(0, D):
            if j in indToInclude:
                outSampleArray.append(imagesArray[:, :, j, :])
                outLableArray.append(labelsArray[:, :, j])

        return outSampleArray, outLableArray

    # ---- Getters ---- #

    def to_string(self):
        logging.info('Pipline object properties:')
        for key, value in self.optionsDict.items():
            logging.info(str(key) + ' : ' + str(value))
        logging.info('\n')

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

# new batching method:

    # create a random permutation vector in size of train data
    def initBatchStackCopy(self):
        self.batchStack = np.random.permutation(np.shape(self.trainSamples)[0])
        self.batchNumer = 0
        return

    # extracts a randomly selected batch for train
    def nextBatchFromPermutation(self, batch_size):
        batchImage = self.trainSamples[self.batchStack[self.batchNumer: self.batchNumer + batch_size]]
        batchLabel = self.trainLabels[self.batchStack[self.batchNumer: self.batchNumer + batch_size]]
        self.batchNumer += batch_size
        return batchImage, batchLabel

        # ---- Help Functions ---- #
    @staticmethod
    def getSlicesFromPatches(patchArrayImage, patchArrayLabel, imageSize):
        # converts an array of patches into slices
        # input needs to be numpy arrays and not tensorflow abject

        patchArrayImage = np.swapaxes(patchArrayImage, 0, 2)
        patchArrayLabel = np.swapaxes(patchArrayLabel, 0, 2)
        H, W, D, C = np.shape(patchArrayImage)
        n = imageSize // H

        ind = 0
        imageArray = np.zeros([imageSize, imageSize, D // (n * n), C])
        labelArray = np.zeros([imageSize, imageSize, D // (n * n)])
        for i in range(n):
            for j in range(n):
                imageArray[H * i:H * (i + 1), H * j:H * (j + 1), :, :] = patchArrayImage[:, :,
                                                                         D // (n * n) * (ind):D // (n * n) * (ind + 1),
                                                                         :]
                labelArray[H * i:H * (i + 1), H * j:H * (j + 1), :] = patchArrayLabel[:, :,
                                                                      D // (n * n) * (ind):D // (n * n) * (ind + 1)]
                ind += 1
        return imageArray, labelArray

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

    def next_image(self,sliceNumber):
        img,labels=self.pre_process_list(listName='train' ,num=sliceNumber)
        return img, labels

# ---- Test Code ---- #
# Example to run different configurations of Datapipline

pipObj = SecondDataPipline(numTrain=1, numVal=1, numTest=1, modalityList=[1,2,3], permotate=True,
                     optionsDict={'normalize': True,
                                  'patchSize': 40,
                                  'normType': 'reg',
                                  'filterSlices': True, 'minPerentageLabeledVoxals': 0.05, 'percentageOfLabeledData': 0.5})

# pipObj = DataPipline(numTrain=2, numVal=2, numTest=2, modalityList=[1,2,3], permotate=True,
#                      optionsDict={'zeroPadding': True, 'paddingSize': 240, 'normalize': True,
#                                   'normType': 'reg', 'filterSlices': True, 'minPerentageLabeledVoxals': 0.05, 'percentageOfLabeledData': 0.5})

# imageArray, labelArray = DataPipline.getSlicesFromPatches(pipObj.trainSamples, pipObj.trainLabels, 240)
# print(np.shape(imageArray))

# pipObj.to_string_pipline()
# # branch few train batches
# batch_size = 64
# pipObj.init_batch_number()
# for i in range(4):
#      train_batch_data, train_batch_labels = pipObj.next_train_random_batch(batch_size)
#      logging.info(np.shape(train_batch_data))
#      logging.info(np.shape(train_batch_labels))
