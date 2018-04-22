from UnetModel import *

'''
   main script for parameter search to run from script line. 
'''

class PermutationDict(object):
    '''
    helper class to save and generate all the permutation to be tested
    '''

    def __init__(self):
        self.patch_size = [48, 80]
        self.filterSlices = [True, False]
        self.min_perentage_labeled_voxals = [0.01, 0.05]
        self.percentage_of_labeled_data = [0.7, 0.5]

        self.num_layers = [3, 4]
        self.depth = [32, 64]
        self.weight_val = [10, 15]
        self.weighted_sum = True
        self.isBatchNorm = [True, False]

        self.learning_rate = [0.01, 0.005, 0.001]
        self.batchSize = [32, 64]

    def get_per_dict_instance(self):

        '''
        getter for a permutaion of the hyper-params for a single random net run
        '''

        paramsDict = {}

        paramsDict['patch_size'] = self.patch_size[self.random_bool()]
        paramsDict['filterSlices'] = self.filterSlices[self.random_bool()]
        num = self.random_bool()
        paramsDict['min_perentage_labeled_voxals'] = self.min_perentage_labeled_voxals[num]
        paramsDict['percentage_of_labeled_data'] = self.percentage_of_labeled_data[num]

        paramsDict['num_layers'] = self.num_layers[self.random_bool()]
        paramsDict['depth'] = self.depth[self.random_bool()]
        paramsDict['weight_val'] = self.weight_val[self.random_bool()]
        paramsDict['weighted_sum'] = False if not(paramsDict['weight_val']) else True
        paramsDict['isBatchNorm'] = self.isBatchNorm[self.random_bool()]

        paramsDict['learning_rate'] = self.learning_rate[self.random_int(1, 3)]
        paramsDict['batchSize'] = self.batchSize[self.random_bool()]

        return paramsDict

    @staticmethod
    def random_bool():
        randNum = np.random.rand(1)
        return int(randNum < 0.5)

    @staticmethod
    def random_int(lowVal, highVal):
        return int(np.random.randint(lowVal, highVal, 1))


def main_func(numOfPerms = 2):

    permDict = PermutationDict()

    lossList = []
    accuracyList = []
    diceList = []

    for iter in range(numOfPerms):

        # init the permutation and create a folder to save logs and files
        createFolder(os.path.realpath(__file__ + "/../"), 'runData')
        runFolderStr = time.strftime('RunFolder_%d_%m_%y__%H_%M_iter_num_{}'.format(iter))
        createFolder(os.path.realpath(__file__ + "/../") + "/runData/", runFolderStr)
        runFolderDir = os.path.realpath(__file__ + "/../") + "/runData/" + runFolderStr
        logFolder = runFolderDir
        startLogging(logFolder, False)

        # get permutation of the parameters dict
        paramsDict = permDict.get_per_dict_instance()
        logging.info('###############################################\n')
        logging.info('Parameters search, iteration mum: {}\n'.format(iter))
        logging.info('Permutation dict values:')

        # print for permutation dict for debug
        for key, value in paramsDict.items():
            logging.info(str(key) + ' : ' + str(value))
        logging.info('###############################################\n')

        # LOAD DATA
        logging.info('Run mode: logging dir: {}'.format(logFolder))
        dataPipe = DataPipline(numTrain=120,
                               numVal=7,
                               numTest=3,
                               modalityList=[0, 1, 2, 3],
                               permotate=False,
                               optionsDict={'zeroPadding': True,
                                            'paddingSize': 240,
                                            'normalize': True,
                                            'normType': 'reg',
                                            'cutPatch': True,
                                            'patchSize': paramsDict['patch_size'],
                                            'binaryLabelsC':True,
                                            'filterSlices': paramsDict['filterSlices'],
                                            'minPerentageLabeledVoxals': paramsDict['min_perentage_labeled_voxals'],
                                            'percentageOfLabeledData': paramsDict['percentage_of_labeled_data']})
        # CREATE MODEL
        unetModel = UnetModelClass(layers=paramsDict['num_layers'],
                                   num_channels=len(dataPipe.modalityList),
                                   num_labels=1,
                                   image_size=paramsDict['patch_size'],
                                   kernel_size=3,
                                   depth=paramsDict['depth'],
                                   pool_size=2,
                                   costStr='sigmoid',
                                   optStr='adam',
                                   argsDict={'layersTodisplay':[1],'weightedSum': paramsDict['weighted_sum'],
                                             'weightVal': paramsDict['weight_val'],
                                             'isBatchNorm': paramsDict['isBatchNorm']})

        # TRAIN AND TEST MODEL
        trainModel = Trainer(net=unetModel, argsDict={'printValidation': 1000})

        trainDict, valDict, testDict = trainModel.train(dataPipe=dataPipe,
                                                        batchSize=paramsDict['batchSize'],
                                                        numSteps=10000,
                                                        printInterval=100,
                                                        logPath=logFolder,
                                                        restore=False,
                                                        restorePath='',
                                                        getStat=True)

        logging.info('Summery data for permutation number {}:'.format(iter))
        logging.info('Train session: mean loss: {}, mean accuracy: {}, mean dice: {}.'.format(round(trainDict['meanLoss'],4),
                                                                                              round(trainDict['meanAccuracy'],4),
                                                                                              round(trainDict['meanDice'],4)))

        logging.info('Validation session: mean loss: {}, mean accuracy: {}, mean dice: {}.'.format(round(valDict['meanLoss'],4),
                                                                                                   round(valDict['meanAccuracy'],4),
                                                                                                   round(valDict['meanDice'],4)))

        logging.info('Test session: mean loss: {}, mean accuracy: {}, mean dice: {}.'.format(round(testDict['loss'],4),
                                                                                             round(testDict['accuracy'],4),
                                                                                             round(testDict['dice'],4)))
        # summery data to compare
        lossList.append(testDict['loss'])
        accuracyList.append(testDict['accuracy'])
        diceList.append(testDict['dice'])

    # print the best performance
    logging.info('The permutation with the best dice was number: {}'.format(np.argmax(diceList)))
    logging.info('The permutation with the best accuracy was number: {}'.format(np.argmax(accuracyList)))
    logging.info('The permutation with the best loss was number: {}'.format(np.argmin(lossList)))

# run as main
if __name__ == "__main__":
    main_func()

# Roy: call for tensorboard
# python3 -m tensorboard.main --logdir /Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/UnetModel/tensorboard
