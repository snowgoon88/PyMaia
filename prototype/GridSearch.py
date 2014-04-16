from ESN import runPredictionESN
from scipy import optimize
import numpy as np


def minimizeMackeyGlass(arg, *params):
	 = arg
	K, N, L, leaking_rate, rho_factor, seed, regul_coef, data, init_len, test_len, train_len = params 

	Ytarget, Y = runPredictionESN(K, 
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

	tmp = []
	for k in range(len(Y.T)):
		tmp.append(abs(Y.T[k] - Ytarget.T[k]))

	return sum(tmp) / len(tmp)

def main():
	tmp = np.loadtxt("data/MackeyGlass_t17_data")
	data = np.zeros((1, len(tmp)))
	for i in range(len(tmp)):
		data[:, i] = tmp[i]

	rranges = (slice(0.25, 0.35, 0.05), slice(1, 1.5, 0.25), slice(40, 45, 1))
	params = (1, 200, 1, 1e-8, data , 100, 1000, 1000)
	
	print "Grid Search..."
	resbrute = optimize.brute(minimizeMackeyGlass, rranges, args=params, full_output=True, finish=optimize.fmin)

	print "min(global error) =", resbrute[1]
	print "with argument:", resbrute[0]

if __name__ == "__main__":
	main()