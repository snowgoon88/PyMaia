from Display import *
from TaskESN import *
from TaskBPDC import *
from TaskDR import *
from Reservoir import *
from DataLoader import *
from matplotlib.pyplot import show
import sys, json, getopt

networks = {
	'ESN': ESN,
	'BPDC': BPDC,
	'DR': DR
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
		'generation': generationBPDC
	},
	'DR': {
		'generation': generationDR,
		'prediction': predictionDR
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
	opt, files = getopt.getopt(sys.argv[1:], '', ['display'])

	for test_file in files:
		fd = open(test_file)
		test_data = json.load(fd)
		fd.close()

		process(test_data)

	if ('--display', '') in opt:
		show()


def process(test_data):
	sys.stdout.write('\t\t')
	print ''.join(['=' for _ in xrange(len(test_data['title'])+8)])
	sys.stdout.write('\t\t')
	print '===', test_data['title'], "==="
	sys.stdout.write('\t\t')
	print ''.join(['=' for _ in xrange(len(test_data['title'])+8)])

	print test_data['reservoir']

	for task in test_data['task']:
		sys.stdout.write('\t')
		print '---', task['title'], '---'

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
			displays[display['type']](Ytarget, Y, title=task['title'], **display['param'])

if __name__ == "__main__":
	main()
