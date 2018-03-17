from UnetModel import *

# -------------------- Logging --------------------

def startLogging(logDir, debug=False):
	# this function starts logging the software outputs.
	# two levels logging: DEBUG and INFO
	# logPrint = logging.getLogger(__package__)

	# init a logger set logging level
	logFormat = '%(asctime)s - %(levelname)s - %(module)s : %(message)s'
	if debug:
		logging.basicConfig(format=logFormat, stream=sys.stdout, level=logging.DEBUG)
		logLevel = logging.DEBUG
	else:
		logging.basicConfig(format=logFormat, stream=sys.stdout, level=logging.INFO)
		logLevel = logging.INFO

	logStr = time.strftime('logFile_%H_%M__%d_%m_%y') + '.log'

	# connect the two streams
	# streamHandler = logging.StreamHandler()
	# streamHandler.setFormatter(logFormat)
	# streamHandler.setLevel(logLevel)

	fileHandler = logging.FileHandler(logDir+'/'+logStr)

	fileHandler.setFormatter(logging.Formatter(logFormat))
	fileHandler.setLevel(logLevel)

	logging.getLogger('').addHandler(fileHandler)
	# logPrint.addHandler(streamHandler)


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


def createFolder(homeDir, folderName):
	directory = homeDir + '/' + folderName
	if not os.path.exists(directory):
		os.makedirs(directory)


