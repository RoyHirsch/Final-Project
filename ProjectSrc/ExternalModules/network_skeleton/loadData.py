
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import skimage.transform as ski

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def load_data(rootDir, maxNum = 2000, isResize = True):
	Xtrain = []
	ind = 0
	for root, dirs, files in os.walk(rootDir):
		for fileName in files:
			# print(filename)
			match = re.search(r'.*.jpg', fileName)
			if match:
				img = plt.imread(os.path.join(root, fileName))
				if len(np.shape(img)) > 2:
					img = rgb2gray(np.array(img))
				img = img / 255 #normalize
				if isResize:
					img = ski.resize(img, [64,64])
				Xtrain.append(img)
				ind += 1
				if ind >= maxNum:
					return Xtrain
	return Xtrain

def load_labels(rootDir, maxNum = 2000, isResize = True):
	ytrain = []
	ind = 0
	for root, dirs, files in os.walk(rootDir):
		for fileName in files:
			# print(filename)
			match = re.search(r'.*.jpg', fileName)
			if match:
				img = plt.imread(os.path.join(root, fileName))
				img = img/255 # normalize
				if isResize:
					img = ski.resize(img, [64,64])
				classImg = np.zeros(np.shape(img))
				classImg[img >= 0.5] = 1  # binary image
				ytrain.append(classImg)
				ind += 1
				if ind >= maxNum:
					return ytrain
	return ytrain
