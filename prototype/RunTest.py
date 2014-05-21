from Display import *
from Task2 import *
from Reservoir import *
from Loader import *
from matplotlib.pyplot import show
import sys, json

networks = {
	'ESN': ESN,
	'BPDC': BPDC
}

loaders = {
	'MackeyGlass': loadMG,
	'Sequence': loadSeq,
	'Trajectory': loadTraj
}

tasks = {
	'generation': generation,
	'rappelGeneration': rappelGeneration,
	'prediction': prediction,
	'rappelPrediction': rappelPrediction,
	'classification': classification,
	'rappelClassification': rappelClassification,
	'classificationPrediction': classificationPrediction
}

displays = {
	'RMSE' : displayRMSE,
	'Accuracy': displayAcc,
	'F-Measure': displayF
}

def main():
	for test_file in sys.argv[1:]:
		fd = open(test_file)
		test_data = json.load(fd)
		fd.close()

		process(test_data)

def process(test_data):
	print ''.join(['=' for _ in xrange(len(test_data['title'])+10)])
	print '====', test_data['title'], "===="
	print ''.join(['=' for _ in xrange(len(test_data['title'])+10)])

	for task in test_data['task']:
		print ''.join(['-' for _ in xrange(len(task['type'])+8)])
		print '---', task['type'], '---'
		print ''.join(['-' for _ in xrange(len(task['type'])+8)])

		sys.stdout.write('Generating reservoir... ')
		network = networks[test_data['type']](**test_data['reservoir'])
		print 'done'

		sys.stdout.write('Loading data... ')
		data = loaders[task['data']['type']](**task['data']['param'])
		print 'done'

		sys.stdout.write('Running task... ')
		Ytarget, Y = tasks[task['type']](network, *data, **task['param'])
		print 'done'

		for display in task['display']:
			displays[display['type']]("%s: %s of %s"%(test_data['title'], task['type'], task['data']['type']), 
									  Ytarget, Y, 
									  **display['param'])

	show()

def loadData(type, signal_path, target_path=None, encode=None):
	if type == 'MackeyGlass' :
		tmp = loadtxt(signal_path)
		data = zeros((1, len(tmp)))
		for i in range(len(tmp)):
			data[:, i] = tmp[i]
		data = (data,)

	elif type == 'Sequence' :
		tmp = loadtxt(signal_path, dtype=string0)
		data = zeros((len(encode), len(tmp)))
		for i in range(len(tmp)):
			data[:, i] = array(encode[tmp[i]])
		data = (data,)

	elif type == 'Trajectory':
		data = (loadtxt(signal_path), loadtxt(target_path))

	return data

if __name__ == "__main__":
	main()
