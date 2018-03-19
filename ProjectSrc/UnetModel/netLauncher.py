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

flags.DEFINE_string('runMode', 'Restore',
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
                     batchSize=4,
                     numSteps=100,
                     printInterval=20,
                     logPath=FLAGS.logFolder,
                     restore=FLAGS.runMode == 'Restore',
                     restorePath=FLAGS.restoreFile)

elif FLAGS.runMode == 'Test':
    testModel = Tester(net=unetModel, testList=[1,2,3,4], argsDict={})
    testModel.test(dataPipe=dataPipe, restorePath=FLAGS.restoreFile)

else:
    logging.info('Error - unknown runMode.')

# Roy: call for tensorboard
# python3 -m tensorboard.main --logdir /Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/UnetModel/tensorboard