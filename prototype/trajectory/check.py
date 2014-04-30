import sys, getopt
import matplotlib.pyplot as plt
import numpy as np

def main():
    try:
        opts, files = getopt.getopt(sys.argv[1:], "", ["help"])
    except getopt.GetoptError as err:
        print str(err)
        usage()
        sys.exit(1)

    for f in files:
    	data = np.loadtxt(f).T
    	plt.figure()
    	plt.plot(data[0, :], data[1, :])

    plt.show()

if __name__ == "__main__":
	main()
