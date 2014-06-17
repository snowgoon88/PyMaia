import numpy as np
import sys, getopt, random


timestep = 2.0*np.pi / 30.0
alpha = 0.7

def class1(t, beta):
	return alpha*np.sin(t+beta)*np.abs(np.sin(t)), alpha*np.cos(t+beta)*np.abs(np.sin(t))

def class2(t, beta):
	return alpha*np.sin(t/2.0+beta)*np.sin(3.0*t/2.0), alpha*np.cos(t/2.0+beta)*np.sin(3.0*t/2.0)

def class3(t, beta):
	return alpha*np.sin(t+beta)*np.sin(2.0*t), alpha*np.cos(t+beta)*np.sin(2.0*t)

def generate():
    clazz = {
    	'1' : class1,
    	'2' : class2,
    	'3' : class3
    }

    for classID in clazz :
        for i in xrange(50):
            fd = open("dataset/%s_%s"%(classID, i+1), 'w')
            beta = 2*np.pi*np.random.random()
            t = 2*np.pi*np.random.random()
            for _ in xrange(30):
    	       fd.write("%s %s\n"%clazz[classID](t, beta))
    	       t += timestep
            fd.close()

def compil():
    dataset = ["dataset/%s_%s"%(c+1, i+1) for c in range(3) for i in range(50)]
    random.shuffle(dataset)

    fdData = open("trajectory_data", 'w')
    fdClass = open("trajectory_class", 'w')

    for data in dataset:
        for line in open(data, 'r'):
            fdData.write(line)
            if data[8] == '1':
                fdClass.write("%s -%s -%s\n"%(alpha, alpha, alpha))
            elif data[8] == '2':
                fdClass.write("-%s %s -%s\n"%(alpha, alpha, alpha))
            elif data[8] == '3':
                fdClass.write("-%s -%s %s\n"%(alpha, alpha, alpha))

    fdData.close()
    fdClass.close()


def main():
    try:
        opts, files = getopt.getopt(sys.argv[1:], "", ["help", "dataset", "compil"])
    except getopt.GetoptError as err:
        print str(err)
        usage()
        sys.exit(1)

    for opt, arg in opts:
        if opt == '--help':
            usage()
            sys.exit(0)
        elif opt == '--dataset':
            generate()
        elif opt == '--compil':
            compil()

def usage():
    print "usage: guess !"

if __name__ == "__main__":
	main()
