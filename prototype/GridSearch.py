from ESN import runPredictionESN
from scipy import optimize
import numpy as np

i = 1 

def minimizeMackeyGlass(arg, *params):
	global i
	print "Iteration", i
	i+=1

	leaking_rate, rho_factor, seed = arg
	K, N, L, regul_coef, data, init_len, test_len, train_len = params 
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
	for k in range(len(Y)):
		tmp = abs(Y[k] - Ytarget[k])

	if np.isinf(sum(tmp) / len(tmp)):
		print "PROBLEME with ",arg, "\nYtarget:", Ytarget, "\nY:", Y


	return sum(tmp) / len(tmp)

def main():
	tmp = np.loadtxt("data/MackeyGlass_t17_data")
	data = np.zeros((1, len(tmp)))
	for i in range(len(tmp)):
		data[:, i] = tmp[i]

	print "GRID SEARCH"
	rranges = (slice(0.25, 0.35, 0.05), slice(1, 1.5, 0.25), slice(40, 45, 1))
	params = (1, 200, 1, 1e-8, data , 100, 1000, 1000)
	
	resbrute = optimize.brute(minimizeMackeyGlass, rranges, args=params, full_output=True, finish=optimize.fmin)

	print "min(global error) =", resbrute[1]
	print "with argument:", resbrute[0]

if __name__ == "__main__":
	main()