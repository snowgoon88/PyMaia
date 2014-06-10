from Binder import *

import sys, json, getopt

def main():
	opt, files = getopt.getopt(sys.argv[1:], '', ['erase'])

	mode = 'a'
	if ('--erase', '') in opt:
		mode = 'w'

	for test_file in files:
		print "\t\t=== %s ==="%test_file

		fd = open(test_file, 'r')
		test_data = json.load(fd)
		fd.close()

		#TODO : generateur de seed, init avec test_data['perform']['seed']

		fd = open(test_data['perform']['file'], mode)

		nbTest = test_data['perform']['repeat'] * len(test_data['task'])
		nbTestExec = 0

		print "Have to perform %i test..."%nbTest

		for i in xrange(test_data['perform']['repeat']):

			# TODO : .next() sur le generateur :)
			seed = test_data['perform']['seed'][i%len(test_data['perform']['seed'])]

			for j in xrange(len(test_data['task'])):

				task = test_data['task'][j]

				network = networks[test_data['reservoir']['type']](seed=seed, **test_data['reservoir']['param'])

				data = loaders[task['data']['type']](**task['data']['param'])

				Ytarget, Y = tasks[test_data['reservoir']['type']][task['type']](network, *data, **task['param'])

				acc = 0 
				for k in xrange(len(Y.T)):
					tmp1 = where(Ytarget[:, k]==max(Ytarget[:, k]))[0][0]
					if isnan(sum(Y[:, k])):
						print "[WARNING] NaN in reservoir output"
						break
					tmp2 = where(Y[:, k]==max(Y[:, k]))[0][0]

					if tmp1 == tmp2:
						acc+=1

				fd.write("%s,%i,%i,%f\n"%( task['title'], 
										   test_data['reservoir']['param']['N'], 
										   seed, 
										   float(acc)/len(Y.T)))

				nbTestExec +=1
				sys.stdout.write("progress: %d%%   \r" % ((nbTestExec*100)/nbTest) )
				sys.stdout.flush()

		fd.close()

if __name__ == "__main__":
	main()
