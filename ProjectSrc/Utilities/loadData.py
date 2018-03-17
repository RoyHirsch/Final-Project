from UnetModel import *
import scipy.io as spio
import numpy as np
import os
import re
import sys

ROOT_DIR = os.path.realpath(__file__ + "/../../")

def get_image_from_mat_file(dir =""):
    mat = spio.loadmat(dir, squeeze_me=True)
    im = mat['im']
    # optional values
    # volnames = mat['im_volnames']
    # filenames = mat['im_filenames']
    # resolution = mat['im_resolution']
    return im

def get_labels_from_mat_file (dir =""):
    mat = spio.loadmat(dir, squeeze_me=True)
    img = mat['gt4']
    return img

def get_data_and_labels_from_folder(dir = ROOT_DIR+'/Data'):
    """
        Extracts all the BARTS data and labels

        Creates a list with N entries (number of samples).
        In each cell is a 4D numpy.ndarray object with size: [H,W,D,C]
            H - image height
            W - image wights
            D - image depth
            C - channels / modalities
        dir - a root folder where all the data is in, the function scans the root folder.
    """
    Xtrain = []
    ytrain = []
    flag = 0
    for root, dirs, files in os.walk(dir):
        for fileName in files:
            # logging.info(filename)
            match = re.search(r'dataBN', fileName)
            if match:
                img = get_image_from_mat_file(os.path.join(root, fileName))
                Xtrain.append(img)
            match = re.search(r'gt4', fileName)
            if match:
                labelImg = get_labels_from_mat_file(os.path.join(root, fileName))
                ytrain.append(labelImg)
    return Xtrain, ytrain

def get_single_mode_data(dataList, modality, isNorm):
    singleModeDataList = []
    for img in dataList:
        temp = img[:, :, :, modality]
        if isNorm:
            maxTemp = np.max(temp)
            temp = temp / maxTemp
        singleModeDataList.append(temp)
    return singleModeDataList

def get_shapes(list):
    for item in list:
        logging.info(np.shape(item))


# Load single pickle file
# path = '/Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/PickledData/train_data0.p'
# img = load_pickle_file(path)

# Load mat files and dunp into pickle files:
# x, y = get_data_and_labels_from_folder()
# get_shapes(x)
# get_shapes(y)
# dump_data_struct_to_pickle_batches(x, 'train_data')
# dump_data_struct_to_pickle(y, 'train_labels')

# Load pickled files:
# data = load_pickle_file_batch('/Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/PickledData/', 'train_data.*')
# labels = load_pickle_file('/Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/PickledData/train_labels.p')
# get_shapes(data)
# get_shapes(labels)


