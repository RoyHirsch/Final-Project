'''
helper script to read results from log files to compare hyper-params
outputs a csv file with the relevant hyper-params values and the last val and test statistics
'''

import os
import re
import pandas as pd

# csvPath = ''
# debug = '/Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/runData/RunFolder_24_04_18__10_49_iter_num_22/logFile_10_49__24_04_18.log'
# runDataRootDir = '/Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/runData'

runDataRootDir = '/Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/runDataFromTheServer/06_05__8_36'
csvPath = runDataRootDir + '/summery.csv'

# get all permutaion params as dicts:
logsDicts = []
for root, dirs, files in os.walk(runDataRootDir):
	for fileName in files:
		match = re.search(r'logFile_', fileName)
		if match:
			file = open(os.path.join(root, fileName),'r')
			# file = open(debug, 'r')
			logText = file.read()
			file.close()
			filterText = re.findall('parameters_search : (\w.*)', logText)[2:-2]
			splitedText = [item.split(' : ') for item in filterText]
			dictParams = dict()
			for item in splitedText:
				if len(item) != 2:
					pass
				elif item[1] in ['True', 'False']:
					dictParams[str(item[0])] = item[1]
				else:
					dictParams[str(item[0])] = float(item[1])

			dictParams['aa_test_name'] = root.split('/')[-1]
			valText = re.findall('(Validation for step num )([0-9])(.*)([\n])(.*)(Training Accuracy : )(.*\d)([\n])(.*)(Dice score: )(.*\d)', logText)
			if len(valText):
				valText = valText[-1]
				dictParams['val_acc'] = valText[6]
				dictParams['val_dice'] = valText[-1]
			else:
				dictParams['val_acc'] = ''
				dictParams['val_dice'] = ''
			testText = re.findall('(Test data)(.*)([\n])(.*)(Training Accuracy : )(.*\d)([\n])(.*)(Dice score: )(.*\d)',logText)
			if len(testText):
				testText = testText[-1]
				dictParams['test_acc'] = testText[5]
				dictParams['test_dice'] = testText[-1]
			else:
				dictParams['test_acc'] = ''
				dictParams['test_dice'] = ''

			logsDicts.append(dictParams)

table = pd.DataFrame([] ,columns=dictParams.keys())
for dict in logsDicts:
	newRow = pd.DataFrame([dict], columns=dict.keys())
	table = pd.concat([table, newRow], axis=0, ignore_index=True)

table.to_csv(csvPath)

#
# sampleText = '2018-05-02 08:47:42,154 - INFO - Trainer : ++++++ Validation for step num 1000 ++++++\n2018-05-02 08:47:42,155 - INFO - Trainer : Training Accuracy : 0.9795\n2018-05-02 08:47:42,155 - INFO - Trainer : Dice score: 0.1710\n'
# valText = re.findall('(Validation for step num )([0-9])(.*)([\n])(.*)(Training Accuracy : )(.*\d)([\n])(.*)(Dice score: )(.*\d)', sampleText)
