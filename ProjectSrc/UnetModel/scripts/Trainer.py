from UnetModel import *

class Trainer(object):
	def __init__(self, net, argsDict):
		logging.info('#### -------- Trainer object was created -------- ####\n')

		self.net = net
		self.argsDict = argsDict

	def __del__(self):
		pass
	def train(self, dataPipe, batchSize, numSteps, printInterval, logPath, restore=False, restorePath=''):

		with tf.Session(graph=self.net.graph) as session:
			tf.global_variables_initializer().run()
			train_writer = tf.summary.FileWriter(logPath, session.graph)
			saver = tf.train.Saver()
			logging.info('Session begun\n')

			# restoring data from model file
			if restore:
				logging.info('Loading data from {}'.format((restorePath)))
				saver.restore(session, "{}".format((restorePath)))

			for step in range(numSteps):
				batchData, batchLabels = dataPipe.next_train_random_batch(batchSize)
				feed_dict = {self.net.X: batchData, self.net.Y: batchLabels}
				_, loss, predictions, summary = session.run(
					[self.net.optimizer, self.net.loss, self.net.predictions, self.net.merged], feed_dict=feed_dict)
				if step % printInterval == 0:
					train_writer.add_summary(summary, step)
					epochAccuracy = accuracy(predictions, batchLabels)
					epochDice = diceScore(predictions, batchLabels)
					logging.info("++++++ Iteration number {:} ++++++".format(step))
					logging.info('Minibatch Loss : {:.4f}'.format(loss))
					logging.info('Training Accuracy : {:.4f}'.format(epochAccuracy))
					logging.info('Dice score: {:.4f}\n'.format(epochDice))
					# resultDisplay(predictions=predictions, labels=batchLabels, images=batchData, sampleInd=1,
					# 			  imageSize=240, imageMod=1, thresh=0.5)

			save_path = saver.save(session, str(logPath)+"/{}_{}_{}_{}.ckpt".format('unet', self.net.layers, self.net.argsDict['weightVal'], time.strftime('%H_%M__%d_%m_%y')))
			logging.info('Saving variables in : %s' % save_path)
			with open('model_file.txt', 'a') as file1:
				file1.write(save_path)
				file1.write(' dice={}\n'.format(epochDice))


def diceScore(predictions, labels):
	eps = 1e-5
	predictions = tf.round(predictions)
	intersection = tf.reduce_sum(tf.multiply(predictions, labels))
	union = eps + tf.reduce_sum(predictions) + tf.reduce_sum(labels)
	res = 2 * intersection / (union + eps)
	return res.eval()

def accuracy(predictions, labels):
	predictions = tf.round(predictions)
	eq = tf.equal(predictions, labels)
	res = tf.reduce_mean(tf.cast(eq, tf.float32))
	return res.eval()




