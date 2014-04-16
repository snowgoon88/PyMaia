import numpy as np
import ESN as esn
from matplotlib.pyplot import *

def main():
    tmp = np.loadtxt("data/MackeyGlass_t17_data")
    data = np.zeros((1, len(tmp)))
    for i in range(len(tmp)):
        data[:, i] = tmp[i]

    Ytarget, Y = esn.modelisation(1, 1000, 1, 42, 0.3, 1.25, 1e-8, data, 100, 2000)

    # err = [] #
    # avg = []
    # var = []
    # for i in range(len(Y.T)):
    #     err.append((Y.T[i] - Ytarget.T[i])**2)
    #     avg.append(sum(err)/len(err))
    #     var.append(sum(np.power(err - avg[i], 2))/len(err))

    print "RMSE:", np.sqrt(np.sum(np.power(Y.T - Ytarget.T, 2)))

    fig = figure(0)
    fig.clear()
    fig.canvas.set_window_title("MackeyGlass_t17")
    # subplot(211)
    plot(Ytarget.T, 'k')
    plot(Y.T, 'r')
    legend(['Target', 'Prediction'])
    # subplot(212)
    # plot(err, 'k')
    # plot(avg, 'r')
    # plot(var, 'r--')
    # legend(['SE', 'RMSE', 'Variance'])
    show()

if __name__ == "__main__":
	main()