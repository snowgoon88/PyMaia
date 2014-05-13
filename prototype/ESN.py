from numpy import *
import scipy.linalg

def uniform(shape, param):
    if "seed" in param:
        random.seed(param["seed"])
    return (param["max"]-param["min"])*random.rand(shape[0], shape[1]) + param["min"]

def gaussian(shape, param):
    if "seed" in param:
        random.seed(param["seed"])
    return param["sigma"] * random.rand(shape[0], shape[1]) + param["mu"]

def sparse(shape, param):
    if "seed" in param:
        random.seed(param["seed"])

distribution = {
    "uniform": uniform,
    "gaussian": gaussian,
    "sparse": sparse
}

class ESN:
    def generate(self, K, N, L, Win, W, leaking_rate, rho_factor):
        ## Reservoir generation
        self.a = leaking_rate
        self.X = zeros((N, 1))
        self.Win = distribution[Win["type"]]((N, 1+K), Win)
        self.W = distribution[Win["type"]]((N, N), W)
        self.Wout = zeros((L, K+N+L))
        ## Spectral radius tuning
        rhoW = max( abs( linalg.eig(self.W)[0] ) )
        self.W *= rho_factor / rhoW

    def input(self, data):
        self.X = (1-self.a)*self.X + self.a*tanh( dot(self.Win, vstack((1, vstack(data)))) + dot(self.W, self.X) )

    def train(self, Ytarget, Xmem, regMatrix):
        # Compute Wout with a ridge regression
        self.Wout = dot( dot(Ytarget, Xmem.T), linalg.inv( dot(Xmem, Xmem.T) + regMatrix ))

    def output(self, data):
        self.input(data)
        return dot(self.Wout, vstack((1,vstack(data),self.X)) )

