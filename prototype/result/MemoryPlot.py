from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D
import sys, getopt

TEST = 0
N = 1
SEED = 2
ACC = -1

colors = ('r', 'g', 'b', 'k', 'y', 'c', 'm', 'w')

def main():
	opt, files = getopt.getopt(sys.argv[1:], '', [])

	for result_file in files:
		test = []
		tmp = {}

		fd = open(result_file, 'r')
		line = fd.readline()
		while line:
			col = line[:-2].split(',')
			if not col[TEST] in test:
				test.append(col[TEST])
			if not col[N] in tmp:
				tmp[col[N]] = {}
			if not col[TEST] in tmp[col[N]]:
				tmp[col[N]][col[TEST]] = []
			tmp[col[N]][col[TEST]].append(float(col[ACC]))
			line = fd.readline()
		fd.close()

		fig = figure()
		fig.clear()
		fig.canvas.set_window_title(result_file)
		ylabel('Accuracy')
		xlabel('Memory')
		xticks(range(len(test)), [x[x.rfind(' ')+1:] for x in test])
		
		k=0
		label = []
		for i in tmp:
			plot(range(len(test)), [sum(tmp[i][x])/len(tmp[i][x]) for x in test], colors[k])
			label.append("N: %s"%i)
			k = (k+1)%len(colors)

		legend(label)

	show()

if __name__ == "__main__":
	main()