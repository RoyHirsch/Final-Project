import matplotlib as mpl
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
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

	# def getTrainLoss(self, lossVal):
	# 	self.trainLossArray.append(lossVal)
	#
	# def getValLoss(self, lossVal):
	# 	self.trainLossArray.append(lossVal)

	def printTrainLossGraph(self):
		plt.figure()
		plt.plot(range(len(self.trainLossArray)), self.trainLossArray)
		plt.show()

def printPredictionSample(predictionImage, validationImage):
	# predictionImage is sigmoid output (logits)
	f, (ax1, ax2) = plt.subplots(2)
	ax1.imshow(validationImage, cmap='gray')
	ax1.set_title('ground truth:')
	tmp = np.zeros_like(predictionImage)
	meanVal = np.mean(predictionImage)
	tmp[predictionImage >= meanVal] = 1
	ax2.imshow(tmp, cmap='gray')
	ax2.set_title('prediction:')
	plt.show()

def sigmoid(x):
  return 1 / (1 + math.exp(-x))