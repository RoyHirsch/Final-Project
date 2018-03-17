from UnetModel import *

'''
    main script for running and testing Unet model
    contains three main components (classes):
    - data pipeline
    - net architecture
    - train

Created by Roy Hirsch and Ori Chayoot, 2018, BGU
'''

train=True

# Make run folder
runFolderStr = time.strftime('RunFolder_%H_%M__%d_%m_%y')
createFolder(os.path.realpath(__file__ + "/../"), 'runData')
createFolder(os.path.realpath(__file__ + "/../") + "/runData/", runFolderStr)
runFolderDir = os.path.realpath(__file__ + "/../") + "/runData/" + runFolderStr

# CONSTANTS
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('run_log_folder',runFolderDir,
                           'a folder for saving all the data from the model\'s run')
tf.app.flags.DEFINE_bool('DEGUB', False, 'deturmine the logging level')
tf.app.flags.DEFINE_bool('train', True, 'train vs. test run mode')

# PATH = '/variables/unet_3_200218_k=3_lr=0.01_d=32.ckpt'
LOG_DIR = os.path.realpath(__file__ + "/../" + "/runData/" + runFolderStr)

# load data
startLogging(FLAGS.run_log_folder)
logging.info('All load and set - let\'s go !')
dataPipe = DataPipline(numTrain=2,  numVal=1, numTest=1,
                       modalityList=[1,2,3], permotate=False,
                       optionsDict={'zeroPadding': True,
                                    'paddingSize': 240,
                                    'normalize': True,
                                    'normType': 'reg',
                                    'binaryLabels': True,
                                    'filterSlices': True,
                                    'minParentageLabeledVoxals': 0.1})

# create net model
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

# train
if train:
    trainModel = Trainer(net=unetModel,argsDict={})

    trainModel.train(dataPipe=dataPipe,
                     batchSize=4,
                     numSteps=100,
                     printInterval=20,
                     logPath=FLAGS.run_log_folder,
                     restore=False,
                     restorePath='')

else:
    testModel = Tester(net=unetModel, testList=[1,2,3,4], argsDict={'mod':[1,3]})
    testModel.test(dataPipe=dataPipe, logPath=LOG_DIR, restorePath='/variables/unet_3_15_140318.ckpt')



# Roy: call for tensorboard
# python3 -m tensorboard.main --logdir /Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/UnetModel/tensorboard