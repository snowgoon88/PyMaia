import numpy as np
import sys, getopt

def class1(t, alpha, beta):
	return alpha*np.sin(t+beta)*np.abs(np.sin(t)), alpha*np.cos(t+beta)*np.abs(np.sin(t))

def class2(t, alpha, beta):
	pass

def class3(t, alpha, beta):
	pass

def main():
    try:
        opts, files = getopt.getopt(sys.argv[1:], "", ["help", "class="])
    except getopt.GetoptError as err:
        print str(err)
        usage()
        sys.exit(1)

    generator = {
    			'1' : class1,
    			'2' : class2,
    			'3' : class3
    }

    clazz = None
    for opt, arg in opts:
    	if opt == "--help":
    		usage()
    		sys.exit(0)
    	elif opt == "--class":
    		if arg not in ('1', '2', '3'):
    			usage()
    			print arg
    			sys.exit(2)
    		clazz = generator[arg]
    		print "class:", clazz, arg


   	timestep = 2*np.pi / 30
   	alpha = 0.7

   	for dataFile in files :
   		fd = open(dataFile, 'w')
   		t = 2*np.pi*np.random.random()
   		beta = 2*np.pi*np.random.random()
    	for i in xrange(30):
    		fd.write("%s %s\n"%clazz(t, alpha, beta))
    		t += timestep
    	fd.close()

def usage():
	print "usage: Guess !"

if __name__ == "__main__":
	main()