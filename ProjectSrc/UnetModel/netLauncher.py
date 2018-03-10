from Utilities.DataPipline import *
from UnetModel.UnetModelClass import *
from UnetModel.Trainer import *
import numpy as np
import os

'''
    main script for running and testing Unet model
    contains three main components (classes):
    - data pipeline
    - net architecture
    - train

Created by Roy Hirsch and Ori Chayoot, 2018, BGU
'''


# CONSTANTS:
PATH = '/variables/unet_3_200218_k=3_lr=0.01_d=32.ckpt'
LOG_DIR = os.path.realpath(__file__ + "/../" + "/tensorboard")

# load data
dataPipe = DataPipline(numTrain=5, numVal=1, numTest=1, modalityList=[1,2,3],
                     optionsDict={'zeroPadding': True, 'paddingSize': 240, 'normalize': True,
                                  'normType': 'reg', 'binaryLabels': True})

# create net model
unetModel = UnetModelClass(layers=3, num_channels=len(dataPipe.modalityList), num_labels=1, image_size=240, kernel_size=3, depth=32,
                        pool_size=2, costStr='sigmoid', optStr='adam', argsDict={})

# train
trainModel = Trainer(net=unetModel, batchSize=2, argsDict={})
trainModel.train(dataPipe=dataPipe, logPath=LOG_DIR, outPath='', numSteps=600, restore=False, restorePath='')

# Roy: call for tensorboard
# python3 -m tensorboard.main --logdir /Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/UnetModel/tensorboard