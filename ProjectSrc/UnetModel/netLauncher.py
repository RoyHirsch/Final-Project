from UnetModel import *

'''
    main script for running and testing Unet model
    contains three main components (classes):
    - data pipeline
    - net architecture
    - train

Created by Roy Hirsch and Ori Chayoot, 2018, BGU
'''

##############################
# CONSTANTS
##############################
flags = tf.app.flags

flags.DEFINE_string('runMode', 'Train',
                    'run mode for the whole sequence: Train, Test or Restore')
flags.DEFINE_bool('debug', False,
                  'logging level - if true debug mode')
tf.app.flags.DEFINE_string('logFolder', '',
                           'logging folder for the sequence, filled automatically')
tf.app.flags.DEFINE_string('restoreFile', '',
                           'path to a .ckpt file for Restore or Test run modes')
FLAGS = flags.FLAGS

# Make new logging folder only in Train mode
if FLAGS.runMode == 'Train':
    createFolder(os.path.realpath(__file__ + "/../"), 'runData')
    runFolderStr = time.strftime('RunFolder_%H_%M__%d_%m_%y')
    createFolder(os.path.realpath(__file__ + "/../") + "/runData/", runFolderStr)
    runFolderDir = os.path.realpath(__file__ + "/../") + "/runData/" + runFolderStr
    FLAGS.logFolder = runFolderDir

# Use perilously defined folder for Test or Restore run modes
if FLAGS.runMode in ['Test', 'Restore']:
    itemsList = FLAGS.restoreFile.split('/')
    FLAGS.logFolder = '/'.join(itemsList[:-1])


##############################
# LOAD DATA
##############################
startLogging(FLAGS.logFolder, FLAGS.debug)
logging.info('All load and set - let\'s go !')
logging.info('Run mode: {} :: logging dir: {}'.format(FLAGS.runMode, FLAGS.logFolder))
dataPipe = DataPipline(numTrain=6,
                       numVal=1,
                       numTest=1,
                       modalityList=[0, 1, 2, 3],
                       permotate=True,#################
                       optionsDict={'zeroPadding': True,
                                    'paddingSize': 240,
                                    'normalize': True,
                                    'normType': 'reg',
                                    'cutPatch': True, #####
                                    'patchSize': 64,
                                    'binaryLabelsC':True,
                                    'filterSlices': True, #####
                                    'minPerentageLabeledVoxals': 0.05,
                                    'percentageOfLabeledData': 0.5})
##############################
# CREATE MODEL
##############################
unetModel = UnetModelClass(layers=3,
                           num_channels=len(dataPipe.modalityList),
                           num_labels=1,
                           image_size=64,
                           kernel_size=3,
                           depth=32,
                           pool_size=2,
                           costStr='combined',
                           optStr='adam',
                           argsDict={'layersTodisplay':[1],'weightedSum': 'True', 'weightVal': 13, 'isBatchNorm': True})

##############################
# RUN MODEL
##############################
if FLAGS.runMode in ['Train', 'Restore']:
    trainModel = Trainer(net=unetModel, argsDict={'printValidation': 50})
    #
    trainModel.train(dataPipe=dataPipe,
                     batchSize=16,
                     numSteps=200,
                     printInterval=20,
                     logPath=FLAGS.logFolder,
                     serialNum=0)

elif FLAGS.runMode == 'Test':
    testModel = Tester(net=unetModel, testList=[1], argsDict={'isPatches': True})
    testModel.test(dataPipe=dataPipe, batchSize=64, restorePath=FLAGS.restoreFile)

else:
    logging.info('Error - unknown runMode.')

# COMMENTS (070418):

# if we use cutPatch option for generating a patches train pipeline, the programmer should make sure that patchSize is a
# dividor of the original image size

# filter train pipeline option gets three parameters:
# filterSlices - determine if the pipeline needs to be filtered (bool)
# minPerentageLabeledVoxals - determine if a patch\slice will be 'labeled', the mean percentage of labeled pixels in a labeled slice
# percentageOfLabeledData - the total percentage of labeled data in the pipline

# UnetModelClass should be suitable to the image_size:
# for example if we will fetch patches of size: 60X60 we will be able to produce a net with 2 max-pooling
# buy won't be able to produce a net with 3 max-pooling (60/8 = 7.5)

# Roy: call for tensorboard
# python3 -m tensorboard.main --logdir /Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/UnetModel/tensorboard

# goof run
# /Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/UnetModel/runData/RunFolder_22_26__14_04_18/unet_2_13_10_37__15_04_18.ckpt
