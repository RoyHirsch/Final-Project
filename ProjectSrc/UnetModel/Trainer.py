from UnetModel.utils import *
import tensorflow as tf
import time

class Trainer(object):
	def __init__(self, net, batchSize, argsDict={}):

		print('\n#### -------- Trainer object was created -------- ####\n')

		self.net = net
		self.batchSize = batchSize
		self.argsDict = argsDict

	def __del__(self):
		print('\n#### -------- Trainer object was deleted -------- ####\n')

	def train(self, dataPipe, logPath, outPath, numSteps, restore=False, restorePath=''):

		with tf.Session(graph=self.net.graph) as session:
			tf.global_variables_initializer().run()
			train_writer = tf.summary.FileWriter(logPath, session.graph)
			saver = tf.train.Saver()
			print('Initialized')

			# restoring data from model file
			if restore == 1:
				print('Loading data from {}'.format((restorePath)))
				saver.restore(session, "{}".format((restorePath)))

			for step in range(numSteps):
				batchData, batchLabels = dataPipe.next_train_random_batch(self.batchSize)
				feed_dict = {self.net.X: batchData, self.net.Y: batchLabels}
				_, loss, predictions, summary = session.run(
					[self.net.optimizer, self.net.loss, self.net.predictions, self.net.merged], feed_dict=feed_dict)
				if step % 5 == 0:
					train_writer.add_summary(summary, step)
					epochAccuracy = accuracy(predictions, batchLabels)
					epochDice = diceScore(predictions, batchLabels)
					print(
						"++++++ Iteration number {:} ++++++ \nMinibatch Loss : {:.4f}\nTraining Accuracy : {:.4f}\nDice score: {:.4f}\n".format(
							step, loss, epochAccuracy, epochDice))
					resultDisplay(predictions=predictions, labels=batchLabels, images=batchData, sampleInd=1,
					              imageSize=240, imageMod=1, thresh=0.5)
			save_path = saver.save(session, "/variables/{}_{}.ckpt".format('unet', time.strftime('%d%m%y')))
			print('Saving variables in : %s' % save_path)
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


