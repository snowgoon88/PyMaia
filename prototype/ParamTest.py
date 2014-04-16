import ESN as esn
from scipy import optimize
import numpy as np
import json, getopt, sys


def mackeyGlass(arg, *params):
	variableParam = arg
	paramToSlide, K, N, L, leaking_rate, rho_factor, regul_coef, data, init_len, train_len, test_len = params 

	if paramToSlide == "N":
		N = variableParam
	elif paramToSlide == "leaking_rate":
		leaking_rate = variableParam
	elif paramToSlide == "rho_factor":
		rho_factor = variableParam
	elif paramToSlide == "regul_coef":
		regul_coef = variableParam
	elif paramToSlide == "seed":
		seed = variableParam
	elif paramToSlide == "train_len":
		train_len = variableParam

	Ytarget, Y = esn.prediction(K, 
								N, 
								L, 
								seed, 
								leaking_rate, 
								rho_factor, 
								regul_coef, 
								data, 
								init_len, 
								train_len, 
								test_len)

	print "Error with", variableParam, ":", sum(abs(Y.T - Ytarget.T),1) / len(Y.T)
	return sum(abs(Y.T - Ytarget.T),1) / len(Y.T)

def main():
	try:
		opts, files = getopt.getopt(sys.argv[1:], "s:", ["help", 
        												 "N", 
        												 "leaking_rate", 
        												 "rho_factor", 
        												 "seed", 
        												 "regul_coef", 
        												 "train_len"])
	except getopt.GetoptError as err:
		print str(err)
		usage()
		sys.exit(1)

	rranges = None
	paramToSlice = None
	for opt, arg in opts :
		if opt == "--help":
			usage()
			sys.exit(0)
		elif opt == "-s":
			tmp = arg.split(':')
			rranges = ( slice(float(tmp[0]), float(tmp[1]), float(tmp[2])), )
		elif opt in ("--N", "--leaking_rate", "--rho_factor", "--seed", "--regul_coef", "--train_len"):
			if paramToSlice == None:
				paramToSlice = opt[2:]
			else:
				usage()
				sys.exit(1)

	if rranges == None or paramToSlice == None :
		usage()
		sys.exit(1)

	for json_file in files:
		print "=====", json_file, "====="

		fd=open(json_file, 'r')
		json_data = json.load(fd)
		fd.close()

		if json_data['data']['type'] == 'MackeyGlass' :
			tmp = np.loadtxt(json_data['data']['path'])
			data = np.zeros((1, len(tmp)))
			for i in range(len(tmp)):
				data[:, i] = tmp[i]

			params = (	paramToSlice, 
            			json_data['esn']['K'], 
            			json_data['esn']['N'], 
            			json_data['esn']['L'],
            			json_data['esn']['leaking_rate'],
            			json_data['esn']['rho_factor'],
            			json_data['esn']['regul_coef'],
            			data,
            			json_data['data']['init_len'],
            			json_data['data']['train_len'],
            			json_data['data']['test_len'])

			resbrute = optimize.brute(mackeyGlass, rranges, args=params, full_output=True)

			print resbrute
			print "min(global error) =", resbrute[1]
			print "with argument:", resbrute[0]

		elif json_data['data']['type'] == 'Sequence' :
			tmp = np.loadtxt(json_data['data']['path'], dtype=string0)
			data = np.zeros((json_data['esn']['K'], len(tmp)))
			for i in range(len(tmp)):
				data[:, i] = np.array(json_data['data']['encode'][tmp[i]])

            #TODO !!
			print 'Unimplemented yet'
			sys.exit(99)

		else :
			print 'Unsupported data type: ',json_data['data']['type']
			sys.exit(2)


def usage():
	print 'usage: guess!'

if __name__ == "__main__":
	main()