from Reservoir import *
from TaskESN import *
from TaskBPDC import *
from TaskDR import *
from DataLoader import *
from Displayer import *

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