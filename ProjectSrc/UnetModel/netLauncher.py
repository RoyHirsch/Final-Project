from UnetModel import *

'''
    main script for running and testing Unet model
    contains three main components (classes):
    - data pipeline
    - net architecture
    - train

Created by Roy Hirsch and Ori Chayoot, 2018, BGU

HOW TO RUN THE SCRIPT ?
Train mode - no need for special flags, leave logFolder and restoreFile empty
Test or Restore mode - need to specify restoreFile, tho code will automatically load the .ckpt
file and will autocomplete the logFolder attribute.

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
                           'logging folder for the sequence, always filled automatically - DO NOT EDIT')
tf.app.flags.DEFINE_string('restoreFile', '',
                           'path to a .ckpt file for Restore or Test run modes')
FLAGS = flags.FLAGS
FLAGS = initLoggingFolder(FLAGS)

##############################
# LOAD DATA
##############################
startLogging(FLAGS.logFolder, FLAGS.debug)
logging.info('All load and set - let\'s go !')
logging.info('Run mode: {} :: logging dir: {}'.format(FLAGS.runMode, FLAGS.logFolder))
dataPipe = DataPipline(numTrain=2,
                       numVal=1,
                       numTest=1,
                       modalityList=[1,2,3],
                       permotate=False,
                       optionsDict={'zeroPadding': True,
                                    'paddingSize': 240,
                                    'normalize': True,
                                    'normType': 'reg',
                                    'binaryLabels': True,
                                    'filterSlices': True,
                                    'minParentageLabeledVoxals': 0.1})

##############################
# CREATE MODEL
##############################
unetModel = UnetModelClass(layers=3,
                           num_channels=len(dataPipe.modalityList),
                           num_labels=1,
                           image_size=240,
                           kernel_size=3,
                           depth=32,
                           pool_size=2,
                           costStr='sigmoid',
                           optStr='adam',
                           argsDict={'weightedSum': 'True', 'weightVal': 13})

##############################
# RUN MODEL
##############################
if FLAGS.runMode in ['Train', 'Restore']:
    trainModel = Trainer(net=unetModel, argsDict={})

    trainModel.train(dataPipe=dataPipe,
                     batchSize=16,
                     numSteps=500,
                     printInterval=20,
                     logPath=FLAGS.logFolder,
                     restore=FLAGS.runMode == 'Restore',
                     restorePath=FLAGS.restoreFile)

elif FLAGS.runMode == 'Test':
    testModel = Tester(net=unetModel, testList=[1,2,3,4], argsDict={})
    testModel.test(dataPipe=dataPipe, restorePath=FLAGS.restoreFile)

else:
    logging.info('Error - undefined runMode')




# Roy: call for tensorboard
# python3 -m tensorboard.main --logdir /Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/UnetModel/tensorboard