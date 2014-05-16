from numpy import *
import scipy.linalg

def uniformDistribution(shape, param):
    if "seed" in param:
        random.seed(param["seed"])
    return (param["max"]-param["min"])*random.rand(shape[0], shape[1]) + param["min"]

def gaussianDistribution(shape, param):
    if "seed" in param:
        random.seed(param["seed"])
    return param["sigma"] * random.rand(shape[0], shape[1]) + param["mu"]

def sparseDistribution(shape, param):
    if "seed" in param:
        random.seed(param["seed"])
    M = zeros(shape)
    for i in xrange(shape[0]):
        for j in xrange(shape[1]):
            prob = random.random()
            k = 0
            while k < len(param["prob"]) and prob > sum(param["prob"][:k]):
                k+=1
            if k < len(param["value"]):
                M[i,j] = param["value"][k]
    return M

distribution = {
    "uniform": uniformDistribution,
    "gaussian": gaussianDistribution,
    "sparse": sparseDistribution
}

class ESN:
    def generate(self, K, N, L, Win, W, leaking_rate, rho_factor):
        ## Reservoir generation
        # leaking rate
        self.a = leaking_rate
        # internal state
        self.X = zeros((N, 1))
        # input weight
        self.Win = distribution[Win["type"]]((N, 1+K), Win)
        # internal weight
        self.W = distribution[W["type"]]((N, N), W)
        # output weight
        self.Wout = zeros((L, K+N+L))
        ## Spectral radius tuning
        rhoW = max( abs( linalg.eig(self.W)[0] ) )
        self.W *= rho_factor / rhoW

    def input(self, data):
        # Update X with input
        self.X = (1-self.a)*self.X + self.a*tanh( dot(self.Win, vstack((1, vstack(data)))) + dot(self.W, self.X) )

    def batchTraining(self, Ytarget, Xmem, regMatrix):
        # Compute Wout with a ridge regression
        self.Wout = dot( dot(Ytarget, Xmem.T), linalg.inv( dot(Xmem, Xmem.T) + regMatrix ))

    def onlineTraining(self, data):
        self.input(data)
        # BPDC here

    def compute(self, data):
        self.input(data)
        return dot(self.Wout, vstack((1,vstack(data),self.X)) )

