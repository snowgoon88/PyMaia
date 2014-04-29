from ESN import generation
from scipy import optimize
import numpy as np

def minimizeMackeyGlass(arg, *params):
	print "With arg:",arg
	leaking_rate, rho_factor = arg
	K, N, L, seed, regul_coef, data, init_len, train_len, test_len = params 
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

	rranges = (slice(0.30, 0.35, 0.01), slice(1.20, 1.25, 0.01))
	params = (1, 1000, 1, 42, 1e-8, data , 100, 1900, 2000)
	
	resbrute = optimize.brute(minimizeMackeyGlass, rranges, args=params, full_output=True, finish=None)

	print "min(RMSE) =", resbrute[1]
	print "with:", resbrute[0]

if __name__ == "__main__":
	main()
