from numpy import *

def loadMG(data):
	tmp = loadtxt(data)
	load = zeros((1, len(tmp)))
	for i in range(len(tmp)):
		load[:, i] = tmp[i]
	return (load,)

def loadSeq(data, encode):
	tmp = loadtxt(data, dtype=string0)
	load = zeros((len(encode), len(tmp)))
	for i in range(len(tmp)):
		load[:, i] = array(encode[tmp[i]])
	return (load,)

def loadTraj(data, target):
	return (loadtxt(data).T, loadtxt(target).T)