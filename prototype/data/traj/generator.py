import numpy as np
import itertools as it

def class1(t, alpha, beta):
	return alpha*np.sin(t+beta)*np.abs(np.sin(t)), alpha*np.cos(t+beta)*np.abs(np.sin(t))

def class2(t, alpha, beta):
	return alpha*np.sin(t/2+beta)*np.sin(3*t/2), alpha*np.cos(t+beta)*np.sin(2*t)

def class3(t, alpha, beta):
	return alpha*np.sin(t+beta)*np.sin(2*t), alpha*np.cos(t+beta)

def main():
    clazz = {
    		'1' : class1,
    		'2' : class2,
    		'3' : class3
            }

    timestep = 2*np.pi / 30
    alpha = 0.7

    for clazzID in clazz :
        for i in xrange(1, 51):
    	   fd = open("dataset/%s_%s"%(clazzID, i), 'w')
    	   beta = 2*np.pi*np.random.random()
        t = 2*np.pi*np.random.random()
        for _ in it.repeat(None, 30):
    	       fd.write("%s %s\n"%clazz[clazzID](t, alpha, beta))
    	       t += timestep
    	fd.close()

if __name__ == "__main__":
	main()