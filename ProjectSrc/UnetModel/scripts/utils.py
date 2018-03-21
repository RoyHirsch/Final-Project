from UnetModel import *

##############################
# INIT FUNCTIONS
##############################

def initLoggingFolder(FLAGS):

	# Use perilously defined folder for Test or Restore run modes
	if FLAGS.runMode in ['Test', 'Restore']:
		if FLAGS.restoreFile:
			itemsList = FLAGS.restoreFile.split('/')
			FLAGS.logFolder = '/'.join(itemsList[:-1])
		else:
			print('Error - undefined restoreFile for runMode Test or Restore')

	# Make new logging folder only in Train mode
	elif FLAGS.runMode == 'Train':
		createFolder(os.path.realpath(__file__ + "/../../"), 'runData')
		runFolderStr = time.strftime('RunFolder_%d_%m_%y__%H_%M')
		createFolder(os.path.realpath(__file__ + "/../../") + "/runData/", runFolderStr)
		runFolderDir = os.path.realpath(__file__ + "/../../") + "/runData/" + runFolderStr
		FLAGS.logFolder = runFolderDir

	else:
		print('Error - undefined runMode')

	return FLAGS

def startLogging(logDir, debug=False):
	# this function starts logging the software outputs.
	# two levels logging: DEBUG and INFO

	# init a logger set logging level
	logFormat = '%(asctime)s - %(levelname)s - %(module)s : %(message)s'
	if debug:
		logging.basicConfig(format=logFormat, stream=sys.stdout, level=logging.DEBUG)
		logLevel = logging.DEBUG
	else:
		logging.basicConfig(format=logFormat, stream=sys.stdout, level=logging.INFO)
		logLevel = logging.INFO

	logStr = time.strftime('logFile_%d_%m_%y__%H_%M') + '.log'

	# connect the two streams
	fileHandler = logging.FileHandler(logDir+'/'+logStr)

	fileHandler.setFormatter(logging.Formatter(logFormat))
	fileHandler.setLevel(logLevel)

	logging.getLogger('').addHandler(fileHandler)


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


