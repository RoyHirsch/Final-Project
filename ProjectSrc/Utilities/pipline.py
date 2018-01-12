from Utilities.loadData import *
import os
import tensorflow as tf

MOD_LIST = ['T1', 'T2', 'T1g', 'FLAIR']
MAX_SAMPLES = 30
MAX_SIZE = 240

# ROOT_DIR = os.path.realpath(__file__ + "/../../")

class DataPipline:

    batch_offset = 0
    optionsDict = {}

    def __init__(self, numTrain, numVal, numTest, modalityList, optionsDict):

        '''
        :param numTrain: number of samples to use in train dataset
        :param numVal: number of samples to use in val dataset
        :param numTest: number of samples to use in test dataset
        :param modalityList: list of numbers to represent the number of channels/modalities to use
               MOD_LIST = ['T1', 'T2', 'T1g', 'FLAIR']
        :param optionsDict: additional options dictionary:
               {'zeroPadding': bool ,'paddingSize': int ,'normType': ['reg', 'clip'] }
        '''

        print('\n#### -------- DataPipline object was created -------- ####\n')
        self.trainNumberList = []
        self.valNumberList = []
        self.testNumberList = []

        self.trainSamples = []
        self.trainLabels = []
        self.valSamples = []
        self.valLabels = []
        self.testSamples = []
        self.testLabels = []

        self.modalityList = modalityList
        self.optionsDict = optionsDict
        # __init__ method
        self._permotate_samples(numTrain, numVal, numTest)
        self.get_samples_list()

    def __del__(self):
        print('\n#### -------- DataPipline object was deleted -------- ####\n')

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

    def normalize_image_modality(self, imgMod):
        if self.optionsDict['normType'] == 'clip':
            b, t = np.percentile(imgMod, (0.5, 99.5))
            imgMod = np.clip(imgMod, b, t)
        elif self.optionsDict['normType'] == 'reg':
            pass
        minImg = np.min(imgMod)
        maxImg = np.max(imgMod)
        diff = maxImg - minImg
        return (imgMod - minImg) / diff

    def normalize_image(self, img):
        normImg = np.zeros(np.shape(img))
        for i in range(0, 4):
            normImg[:, :, :, i] = self.normalize_image_modality(img[:, :, :, i])
        return normImg

    def zero_padding_img(self, maxSize, img):
        [H, W, D, C] = np.shape(img)
        if (H == maxSize) and (W == maxSize):
            return img
        else:
            hOffset = int((maxSize - H) / 2)
            wOffset = int((maxSize - W) / 2)
            paddedImg = np.zeros([maxSize, maxSize, D, C])
            paddedImg[hOffset:H + hOffset, wOffset:W + wOffset, :, :] = img
            return paddedImg

    def zero_padding_label(self, label):
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

    def crop_iamge(self, img, maxH, maxW):
        H, W, D, C = np.shape(img)
        diffH = round((H - maxH) / 2)
        diffW = round((W - maxW) / 2)
        cropedImg = img[diffH:H - diffH, diffW:W - diffW, :, :]
        return cropedImg

    def pre_process_list(self,listName, data, labels):
        '''
        load data, extract only the relevant channels zero-pad, normalize, output as a list of np.array
        '''
        if listName == 'train':
            numbersList = self.trainNumberList
        elif listName in ['val','validation']:
            numbersList = self.valNumberList
        elif listName == 'test':
            numbersList = self.testNumberList
        else:
            print('Error while calling pre_process_list')

        outSampleArray = []
        outLabelArray = []
        for i in numbersList:
            img = data[i]
            label = labels[i]

            if self.optionsDict['zeroPadding']:
                img = self.zero_padding_img(self.optionsDict['paddingSize'], img)
                label = self.zero_padding_label(label)

            img = self.normalize_image(img)
            H, W, D, C = np.shape(img)
            tmp = img[:, :, :, self.modalityList]

            for j in range(0,D):
                outSampleArray.append(tmp[:, :, j, :])
                outLabelArray.append(label[:, :, j])

        # a fix to locate the D dimension in it's place
        # outSampleArray = np.swapaxes(outSampleArray, 0, 2)
        # outLabelArray = np.swapaxes(outLabelArray, 0, 2)
        outSampleArray = np.array(outSampleArray)
        outLabelArray = np.array(outLabelArray)
        return outSampleArray, outLabelArray

    def get_samples_list(self):
        '''
            samples the data , pre-process and divide the data into the specified lists
        '''
        data, labels = get_data_and_labels_from_folder()
        print('Data and labels where uploaded successfully')
        self.trainSamples, self.trainLabels = self.pre_process_list('train', data, labels)
        self.valSamples, self.valLabels = self.pre_process_list('val', data, labels)
        self.testSamples, self.testLabels = self.pre_process_list('test', data, labels)
        print('Train, val and test database created successfully.')

        # Printings for debug:
        print('Train dataset, samples: ' + str(self.trainNumberList) + '\n' +
              ' Shape of train dataset: ' + str(np.shape(self.trainSamples)))
        print('Val dataset, samples: ' + str(self.valNumberList) + '\n' +
              ' Shape of val dataset: ' + str(np.shape(self.valSamples)))
        print('Test dataset, samples: ' + str(self.testNumberList) + '\n' +
              ' Shape of test dataset: ' + str(np.shape(self.testSamples)))

    # ---- Getters ---- #

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

    def next_train_random_batch(self, batch_size):
        ind = np.random.random_integers(0, np.shape(self.trainSamples)[2], batch_size)
        print(ind)
        return self.trainSamples[ind, :, :, :], self.trainLabels[ind, :, :]

# ---- Help Functions ---- #

def print_img_statistics(img):
    modalities = ['T1', 'T2', 'T1g', 'FLAIR']
    for i in range(0, 4):
        print('Image modality: ' + modalities[i] + ': Mean: ' +
              str(np.mean(img[:, :, :, i])) + ' Variance: ' + str(np.std(img[:, :, :, i])))
        print('Image max: ' + str(np.max(img)) + ' Image min: ' + str(np.min(img)))

def print_histogram(img):

    counts, bins = np.histogram(img.ravel(), bins=255)
    # plt.bar(bins[1:-1],counts[1:])
    plt.bar(bins[:-1], counts)
    plt.show()

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

# ---- Test Code ---- #

# declare some data object
# pipObj = DataPipline(5, 1, 4, [1,2,3], {'zeroPadding': True, 'paddingSize': 240, 'normType': 'reg'})
# print(np.shape(pipObj.trainLabels))
# print(np.shape(pipObj.trainSamples))
# branch few train batches
# pipObj.reset_train_batch_offset()
# batch_size = 64
# for i in range(4):
#     train_batch_data, train_batch_labels = pipObj.next_train_random_batch(batch_size)
#     print(np.shape(train_batch_data))
#     print(np.shape(train_batch_labels))
#     print(pipObj.batch_offset)
