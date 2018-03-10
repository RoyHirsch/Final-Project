# import matplotlib as mpl
# mpl.use('TkAgg')
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import math

class MetaDataCollector(object):

	def __init__(self):
		self.trainLossArray = []
		self.trainAccArray = []
		self.valAccArray = []

	def getStepValues(self, trainLoss, trainAcc, valAcc):
		self.trainLossArray.append(trainLoss)
		self.trainAccArray.append(trainAcc)
		self.valAccArray.append(valAcc)

	def printTrainLossGraph(self):
		plt.figure()
		plt.plot(range(len(self.trainLossArray)), self.trainLossArray)
		plt.show()

def resultDisplay(predictions, labels, images, sampleInd, imageSize, imageMod, thresh = 0.5):

	# get specific sample
	prediction = predictions[sampleInd,:,:]
	label = labels[sampleInd,:,:]
	image = images[sampleInd,:,:,imageMod]

	# clip prediction into bineary mask
	prediction[prediction > thresh] = 1
	prediction[prediction <= thresh] = 0

	# print results
	plt.figure(1)
	plt.subplot(131)
	plt.title('Label')
	plt.imshow(np.reshape(label, [imageSize, imageSize]), cmap='gray')
	plt.subplot(132)
	plt.title('Prediction')
	plt.imshow(np.reshape(prediction, [imageSize, imageSize]), cmap='gray')
	plt.subplot(133)
	plt.title('Image')
	plt.imshow(np.reshape(image, [imageSize, imageSize]), cmap='gray')
	plt.show()

def sigmoid(x):
  return 1 / (1 + math.exp(-x))