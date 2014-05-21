from Display import *
from Task import *
from Reservoir import *
from DataLoader import *
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
	'F-Measure': displayF,
	'Trajectory-2D': displayTraj2D,
	'Trajectory-3D': displayTraj3D
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
		network = networks[test_data['reservoir']['type']](**test_data['reservoir']['param'])
		print '\t[done]'

		sys.stdout.write('Loading data... ')
		data = loaders[task['data']['type']](**task['data']['param'])
		print '\t\t[done]'

		sys.stdout.write('Running task... ')
		Ytarget, Y = tasks[task['type']](network, *data, **task['param'])
		print '\t\t[done]'

		for display in task['display']:
			displays[display['type']]("%s: %s of %s"%(test_data['title'], task['type'], task['data']['type']), 
									  Ytarget, Y, 
									  **display['param'])

	show()

if __name__ == "__main__":
	main()
