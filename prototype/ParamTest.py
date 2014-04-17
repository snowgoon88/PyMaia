import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import ESN as esn
import json, getopt, sys

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
			rranges = (float(tmp[0]), float(tmp[1]), float(tmp[2]))
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
		fd=open(json_file, 'r')
		json_data = json.load(fd)
		fd.close()

		if json_data['data']['type'] == 'MackeyGlass' :
			tmp = np.loadtxt(json_data['data']['path'])
			data = np.zeros((1, len(tmp)))
			for i in range(len(tmp)):
				data[:, i] = tmp[i]

			K = json_data['esn']['K']
			N = json_data['esn']['N'] 
			L = json_data['esn']['L']
			a = json_data['esn']['leaking_rate']
			rho = json_data['esn']['rho_factor']
			seed = json_data['esn']['seed']
			b = json_data['esn']['regul_coef']
			init = json_data['data']['init_len']
			train = json_data['data']['train_len']
			test = json_data['data']['test_len']

			rmse = []
			xticks = []
			for i in np.arange(rranges[0], rranges[1]+rranges[2], rranges[2]):
				print json_file, "with", paramToSlice, "=", i
				xticks.append(i)
				if paramToSlice == "N":
					Ytarget, Y = esn.generation(K, i, L, seed, a, rho, b, data, init, train, test)
				elif paramToSlice == "seed":
					Ytarget, Y = esn.generation(K, N, L, i, a, rho, b, data, init, train, test)
				elif paramToSlice == "leaking_rate":
					Ytarget, Y = esn.generation(K, N, L, seed, i, rho, b, data, init, train, test)
				elif paramToSlice == "rho_factor":
					Ytarget, Y = esn.generation(K, N, L, seed, a, i, b, data, init, train, test)
				elif paramToSlice == "regul_coef":
					Ytarget, Y = esn.generation(K, N, L, seed, a, rho, i, data, init, train, test)
				elif paramToSlice == "train_len":
					Ytarget, Y = esn.generation(K, N, L, seed, a, rho, b, data, init, i, test) 	
				else:
					Ytarget, Y = esn.generation(K, N, L, seed, a, rho, b, data, init, train, test)
				rmse.append(np.sqrt(np.sum(np.power(Y.T - Ytarget.T, 2))/len(Y.T)))

				
				print "RMSE=", rmse[-1]

			plt.figure().canvas.set_window_title(json_file)
			plt.plot(rmse, 'k')
			plt.xticks(range(len(xticks)), xticks)
			plt.xlabel(paramToSlice)
			plt.ylabel("RMSE")
			plt.show()


		elif json_data['data']['type'] == 'Sequence' :
			tmp = np.loadtxt(json_data['data']['path'], dtype=np.string0)
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