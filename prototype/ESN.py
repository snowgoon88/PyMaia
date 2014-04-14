from numpy import *
from matplotlib.pyplot import *
import scipy.linalg

class ESN:
    def generate(self, K, N, L, seed, leaking_rate, rho_factor):
        self.x = zeros((N, 1))
        self.Wout = zeros((L, K+N+L))
        self.a = leaking_rate
        ## Reservoir generation
        random.seed(seed)
        ### With an uniform distribution on [0;1)
        self.Win = random.rand(N, 1+K)-0.5
        self.W = random.rand(N, N)-0.5
        ### With a gaussian distribution on [-1;1]
        #self.Win = random.randn(N, 1+K)
        #self.W = random.randn(N,N)
        ## Spectral radius tuning
        rhoW = max( abs( linalg.eig(self.W)[0] ) )
        self.W *= rho_factor / rhoW

    def input(self, data):
        self.x = (1-self.a)*self.x + self.a*tanh( dot(self.Win, vstack((1, vstack(data)))) + dot(self.W, self.x) )

    def train(self, Ytarget, Xmem, regMatrix):
        # Compute Wout with a ridge regression
        self.Wout = dot( dot(Ytarget, Xmem.T), linalg.inv( dot(Xmem, Xmem.T) + regMatrix ))

    def output(self, data):
        self.input(data)
        return dot(self.Wout, vstack((1,vstack(data),self.x)) )


def runESN(K, N, L, seed, leaking_rate, rho_factor, regul_coef, data, initLen, trainLen, testLen):
    print 'Step 1/5: Reservoir generation'
    network = ESN()
    network.generate(K, N, L, seed, leaking_rate, rho_factor)

    print 'Step 2/5: Transient initialization'
    for t in range(initLen):
        network.input(data[:, t])

    print 'Step 3/5: Learning phase'
    Xmem = zeros((1+K+N, trainLen - initLen))
    for t in range(initLen, trainLen):
        network.input(data[:, t])
        Xmem[:, t-initLen] = vstack((1, vstack(data[:, t]), network.x))[:,0]

    print 'Step 4/5: Wout computation'
    network.train(data[:, initLen+1:trainLen+1], Xmem, regul_coef*eye(1+K+N))

    print 'Step 5/5: Testing phase'
    Ymem = zeros((L, testLen))
    u = data[:, trainLen]
    for t in range(testLen):
        Ymem[:, t] = hstack(network.output(u))
        u = Ymem[:, t]

    return data[:, trainLen+1:trainLen+testLen+1], Ymem
