from numpy import *
from matplotlib.pyplot import *
import scipy.linalg
import json, getopt, sys

import pdb

class ESN:
    def generate(self, K, N, L, seed, leaking_rate, rho_factor):
        self.x = zeros((N, 1))
        self.Wout = zeros((L, K+N+L))
        self.a = leaking_rate
        # Reservoir generation
        random.seed(seed)
        # With an uniform distribution on [0;1)
        self.Win = random.rand(N, 1+K)-0.5
        self.W = random.rand(N, N)-0.5
        # With a gaussian distribution on [-1;1]
        #self.Win = random.randn(N, 1+K)
        #self.W = random.randn(N,N)

        # Spectral radius tuning
        rhoW = max( abs( linalg.eig(self.W)[0] ) )
        self.W *= rho_factor / rhoW

    def feed(self, data):
        self.x = (1-self.a)*self.x + self.a*tanh( dot(self.Win, vstack((1, vstack(data)))) + dot(self.W, self.x) )

    def solve(self, data):
        self.feed(data)
        return dot(self.Wout, vstack((1,vstack(data),self.x)) )

    def train(self, Ytarget, Xmem, regMatrix):
        # ridge regression
        self.Wout = dot( dot(Ytarget, Xmem.T), linalg.inv( dot(Xmem, Xmem.T) + regMatrix ) )


def runESN(K, N, L, seed, leaking_rate, rho_factor, regul_coef, data, initLen, trainLen, testLen):
    Xmem = zeros((1+K+N, trainLen - initLen))
    Ytarget = data[:, initLen+1:trainLen+1]
    # init network
    network = ESN()
    # generate a reservoir
    print 'Step 1/5: Reservoir generation'
    network.generate(K, N, L, seed, leaking_rate, rho_factor)

    # initialisation transient
    print 'Step 2/5: Initialisation transient'
    for t in range(initLen):
        network.feed(data[:, t])

    # feed
    print 'Step 3/5: Learning'
    for t in range(initLen, trainLen):
        network.feed(data[:, t])
        Xmem[:, t-initLen] = vstack((1, vstack(data[:, t]), network.x))[:,0]

    # train
    print 'Step 4/5: Compute Wout'
    network.train(Ytarget, Xmem, regul_coef*eye(1+K+N))

    # test
    print 'Step 5/5: Testing'
    Ymem = zeros((L, testLen))
    u = data[:, trainLen]
    for t in range(testLen):
        Ymem[:, t] = hstack(network.solve(u))
        u = Ymem[:, t]

    return data[:, trainLen+1:trainLen+testLen+1], Ymem

def main():
    try:
        opts, files = getopt.getopt(sys.argv[1:], "do:", ["help"])
    except getopt.GetoptError as err:
        print str(err)
        usage()
        sys.exit(1)

    display = False
    for opt, arg in opts:
        if opt == '--help':
            usage()
            sys.exit(0)
        elif opt == '-d':
            display = True
            
    k = 0
    for json_file in files :
        fd=open(json_file, 'r')
        json_data = json.load(fd)
        fd.close()

        # data setup
        data = None
        if json_data['data']['type'] == 'MackeyGlass' :
            tmp = loadtxt(json_data['data']['path'])
            data = zeros((1, len(tmp)))
            for i in range(len(tmp)):
                data[:, i] = tmp[i]
        elif json_data['data']['type'] == 'Sequential' :
            tmp = loadtxt(json_data['data']['path'], dtype=string0)
            data = zeros((2, len(tmp)))
            for i in range(len(tmp)):
                data[:, i] = array(json_data['data']['encode'][tmp[i]])
        else :
            print 'Unsupported data type: ',json_data['data']['type']
            sys.exit(2)
 
        print '=== Running', json_file, "==="

        Ytarget, Y = runESN(json_data['esn']['K'], 
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

        if display:
            k+=1
            figure(k).clear()
            figure(k).canvas.set_window_title(json_file)
            subplot(211)
            plot(Ytarget.T, 'k')
            plot(Y.T, 'r')
            legend(['Ytarget', 'Y'])
            subplot(212)
            plot(Ytarget.T - Y.T, 'k')
            legend(['Error'])
            show()

def usage():
    print 'python ESN.py [--help] [-v] <test_file>* )'

if __name__ == "__main__":
    main()
