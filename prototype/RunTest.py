from Display import *
from TaskESN import *
from TaskBPDC import *
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
	'ESN': {
		'generation': generationESN,
		'rappelGeneration': rappelGenerationESN,
		'prediction': predictionESN,
		'rappelPrediction': rappelPredictionESN,
		'classification': classificationESN,
		'rappelClassification': rappelClassificationESN,
		'classificationPrediction': classificationPredictionESN
	},
	'BPDC': {
		'generation':generationBPDC
	}
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
	sys.stdout.write('\t\t')
	print ''.join(['=' for _ in xrange(len(test_data['title'])+8)])
	sys.stdout.write('\t\t')
	print '===', test_data['title'], "==="
	sys.stdout.write('\t\t')
	print ''.join(['=' for _ in xrange(len(test_data['title'])+8)])

	for task in test_data['task']:
		sys.stdout.write('\t')
		print '---', task['type'], '---'

		sys.stdout.write('Generating reservoir... ')
		sys.stdout.flush()
		network = networks[test_data['reservoir']['type']](**test_data['reservoir']['param'])
		print '\t[done]'

		sys.stdout.write('Loading data... ')
		sys.stdout.flush()
		data = loaders[task['data']['type']](**task['data']['param'])
		print '\t\t[done]'

		sys.stdout.write('Running task... ')
		sys.stdout.flush()
		Ytarget, Y = tasks[test_data['reservoir']['type']][task['type']](network, *data, **task['param'])
		print '\t\t[done]'

		for display in task['display']:
			displays[display['type']](Ytarget, Y, **display['param'])

	show()

if __name__ == "__main__":
	main()
