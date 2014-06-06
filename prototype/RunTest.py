from Binder import *

from matplotlib.pyplot import show
import sys, json, getopt

def main():
	opt, files = getopt.getopt(sys.argv[1:], '', ['display'])

	toShow = False
	for o, a in opt:
		if o == '--display':
			toShow = True

	for test_file in files:
		fd = open(test_file, 'r')
		test_data = json.load(fd)
		fd.close()

		sys.stdout.write('\t\t')
		print ''.join(['=' for _ in xrange(len(test_data['title'])+8)])
		sys.stdout.write('\t\t')
		print '===', test_data['title'], "==="
		sys.stdout.write('\t\t')
		print ''.join(['=' for _ in xrange(len(test_data['title'])+8)])

		for task in test_data['task']:
			sys.stdout.write('\t')
			print '---', task['title'], '---'

			sys.stdout.write('Generating reservoir... ')
			sys.stdout.flush()
			network = networks[test_data['reservoir']['type']](**test_data['reservoir']['param'])
			print '\t[done]'

			sys.stdout.write('Loading data... ')
			sys.stdout.flush()
			data = loaders[task['data']['type']](**task['data']['param'])
			print '\t\t[done]'

			sys.stdout.write('Running task... ')
			sys.stdout.flush()
			Ytarget, Y = tasks[test_data['reservoir']['type']][task['type']](network, *data, **task['param'])
			print '\t\t[done]'

			for display in test_data['display']:
				displays[display['type']](Ytarget, Y, title=task['title'], **display['param'])

	if toShow:
		show()


if __name__ == "__main__":
	main()
