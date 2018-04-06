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
dataPipe = DataPipline(numTrain=5,
                       numVal=2,
                       numTest=2,
                       modalityList=[0, 1, 2],
                       permotate=False,
                       optionsDict={'zeroPadding': True,
                                    'paddingSize': 240,
                                    'normalize': True,
                                    'normType': 'reg',
                                    'cutPatch': True,
                                    'patchSize': 60,
                                    'binaryLabelsC':True,
                                    'filterSlices': True,
                                    'minPerentageLabeledVoxals': 0.05,
                                    'percentageOfLabeledData': 0.5})

##############################
# CREATE MODEL
##############################
unetModel = UnetModelClass(layers=2,
                           num_channels=len(dataPipe.modalityList),
                           num_labels=1,
                           image_size=60,
                           kernel_size=3,
                           depth=32,
                           pool_size=2,
                           costStr='sigmoid',
                           optStr='adam',
                           argsDict={'layersTodisplay':[1],'weightedSum': 'True', 'weightVal': 13})

##############################
# RUN MODEL
##############################
if FLAGS.runMode in ['Train', 'Restore']:
    trainModel = Trainer(net=unetModel, argsDict={})

    trainModel.train(dataPipe=dataPipe,
                     batchSize=16,
                     numSteps=100,
                     printInterval=50,
                     logPath=FLAGS.logFolder,
                     restore=FLAGS.runMode == 'Restore',
                     restorePath=FLAGS.restoreFile)

elif FLAGS.runMode == 'Test':
    testModel = Tester(net=unetModel, testList=[1,2,3,4], argsDict={})
    testModel.test(dataPipe=dataPipe, restorePath=FLAGS.restoreFile)

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