from ESN import generation
from scipy import optimize
import numpy as np

def minimizeMackeyGlass(arg, *params):
	leaking_rate, rho_factor, seed = arg
	print "leaking_rate:",leaking_rate
	print "rho_factor:", rho_factor
	print "seed:", seed

	K, N, L, regul_coef, data, init_len, train_len, test_len = params 
	Ytarget, Y = generation(K, 
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

	rmse = np.sqrt(np.sum(np.power(Ytarget.T - Y.T, 2))/len(Y.T))
	print "=> RMSE:", rmse
	return rmse

def main():
	tmp = np.loadtxt("data/MackeyGlass_t17_data")
	data = np.zeros((1, len(tmp)))
	for i in range(len(tmp)):
		data[:, i] = tmp[i]

	rranges = (slice(0.25, 0.35, 0.05), slice(1, 1.5, 0.25), slice(40, 45, 1))
	params = (1, 1000, 1, 1e-8, data , 100, 1900, 2000)
	
	resbrute = optimize.brute(minimizeMackeyGlass, rranges, args=params, full_output=True, finish=None)

	print "min(RMSE) =", resbrute[1]
	print "with:", resbrute[0]

if __name__ == "__main__":
	main()
