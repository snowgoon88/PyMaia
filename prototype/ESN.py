from numpy import *
from matplotlib.pyplot import *
import scipy.linalg
import json, getopt, sys

import pdb

class ESN:
    def generate(self, K, N, L, seed, leaking_rate, rho_factor):
        self.a = leaking_rate
        self.x = zeros((N, 1))
        self.Wout = zeros((L, K+N+L))
        # Reservoir generation
        random.seed(seed)
        self.Win = random.rand(N, 1+K)-0.5
        self.W = random.rand(N, N)-0.5
        #self.Win = random.randn(N, 1+K)
        #self.W = random.randn(N,N)
        # Spectral radius tuning
        rhoW = max( abs( linalg.eig(self.W)[0] ) )
        self.W *= rho_factor / rhoW

    def feed(self, data):
        self.x = (1-self.a)*self.x + self.a*tanh( dot(self.Win, vstack((1, data))) + dot(self.W, self.x) )

    def solve(self, data):
        self.feed(data)
        return dot(self.Wout, vstack((1,data,self.x)) )

    def train(self, Ytarget, Xmem, regMatrix):
        # ridge regression
        self.Wout = dot( dot(Ytarget, Xmem.T), linalg.inv( dot(Xmem, Xmem.T) + regMatrix ) )
        pdb.set_trace()


def runESN(verbose, K, N, L, seed, leaking_rate, rho_factor, regul_coef, data, initLen, trainLen, testLen):
    Xmem = zeros((1+K+N, trainLen - initLen))
    Ytarget = data[None, initLen+1:trainLen+1]

    # init network
    network = ESN()
    # generate a reservoir
    if verbose :
        print 'Step 1/5: Reservoir generation'
    network.generate(K, N, L, seed, leaking_rate, rho_factor)

    # initialisation transient
    if verbose :
        print 'Step 2/5: Initialisation transient'
    for t in range(initLen):
        network.feed(data[t])

    # feed
    if verbose :
        print 'Step 3/5: Learning'
    for t in range(initLen, trainLen):
        network.feed(data[t])
        Xmem[:, t-initLen] = vstack((1, data[t], network.x))[:,0]

    # train
    if verbose:
        print 'Step 4/5: Compute Wout'
    network.train(Ytarget, Xmem, regul_coef*eye(1+K+N))

    # test
    if verbose:
        print 'Step 5/5: Testing'
    Ymem = zeros((L, testLen))
    u = data[trainLen]
    for t in range(testLen):
        Ymem[:, t] = network.solve(u)
        u = Ymem[:, t]

    return data[trainLen+1:trainLen+testLen+1], Ymem

def main():
    try:
        opts, files = getopt.getopt(sys.argv[1:], "v", ["help"])
    except getopt.GetoptError as err:
        print str(err)
        usage()
        sys.exit(1)

    verbose = False
    for opt, arg in opts:
        if opt == '-v':
            verbose = True
        elif opt == '--help':
            usage()
            sys.exit(0)
    
    for json_file in files :
        fd=open(json_file, 'r')
        json_data = json.load(fd)
        fd.close()

        # data setup
        data = None
        if json_data['data']['type'] == 'MackeyGlass' :
            data = loadtxt(json_data['data']['path'])
        elif json_data['data']['type'] == 'Sequential' :
            fd = open(json_data['data']['path'], 'r')
            #data = loadtxt(json_data['data']['path'], dtype=string0)
            data = fd.read().split()
            fd.close()
            for i in range(len(data)):
                data[i] = json_data['data']['encode'][data[i]]
            data = array(data)
        else :
            print 'Unsupported data type: ',json_data['data']['type']
            sys.exit(2)
 
        if verbose:
            print '=== Running', json_file, "==="

        Ytarget, Y = runESN(verbose,
                            json_data['esn']['K'], 
                            json_data['esn']['N'], 
                            json_data['esn']['L'], 
                            json_data['esn']['seed'], 
                            json_data['esn']['leaking_rate'], 
                            json_data['esn']['rho_factor'], 
                            json_data['esn']['regul_coef'],
                            data, 
                            json_data['data']['init_len'], 
                            json_data['data']['train_len'], 
                            json_data['data']['test_len'])
        
        if verbose:
            print '===================='

        figure(1).clear()
        plot(Ytarget, 'g')
        plot(Y.T, 'b')
        title(json_file)
        legend(['Target signal', 'Predicted signal'])
        show()

def usage():
    print 'python ESN.py [--help] [-v] <test_file>* )'

if __name__ == "__main__":
    main()
