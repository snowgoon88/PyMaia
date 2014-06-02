from numpy import *
from scipy.linalg import eig, inv, pinv

def uniformDistribution(shape, param):
    if 'seed' in param:
        random.seed(param['seed'])
    return (param["max"]-param['min'])*random.rand(shape[0], shape[1]) + param['min']

def gaussianDistribution(shape, param):
    if 'seed' in param:
        random.seed(param['seed'])
    return param['sigma'] * random.randn(shape[0], shape[1]) + param['mu']

def sparseDistribution(shape, param):
    if 'seed' in param:
        random.seed(param['seed'])
    M = zeros(shape)
    for i in xrange(shape[0]):
        for j in xrange(shape[1]):
            prob = random.random()
            k = 0
            while k < len(param['prob']) and prob > sum(param['prob'][:k]):
                k+=1
            if k < len(param['value']):
                M[i,j] = param['value'][k]
    return M

distribution = {
    "uniform": uniformDistribution,
    "gaussian": gaussianDistribution,
    "sparse": sparseDistribution
}

function = {
    "tanh": tanh
}

def tanhprim(x):
    return 1 - power(tanh(x), 2)

derived = {
    "tanh": tanhprim
}

class Reservoir:
    def __init__(self, K, N, L, Win, W, f, leaking_rate, rho_factor):
        ## Reservoir generation
        # constant
        self.a = leaking_rate
        self.K = K
        self.N = N
        self.L = L
        self.f = function[f]
        # internal state
        self.X = zeros((N, 1))
        # input weight
        self.Win = distribution[Win["type"]]((N, 1+K), Win)
        # internal weight
        self.W = distribution[W["type"]]((N, N), W)
        # output weight
        self.Wout = random.rand(L, 1+K+N)-0.5
        ## Spectral radius tuning
        rhoW = max( abs( eig(self.W)[0] ) )
        self.W *= rho_factor / rhoW

    def input(self, data):
        # Update X with input
        self.X = (1-self.a)*self.X + self.a*self.f( dot(self.Win, vstack((1, vstack(data)))) + dot(self.W, self.X) )

    def compute(self, data):
        self.input(data)
        return dot(self.Wout, vstack((1,vstack(data),self.X)) )

    def train(self, **params):
        raise NotImplementedError("A raw reservoir can't be train !")

# Batch training
class ESN(Reservoir):
    def train(self, **params):
        Ytarget = params['Ytarget']
        Xmen = params['Xmen']
        if 'regul_matrix' in params:
            # Compute Wout with a ridge regression
            self.Wout = dot( dot(Ytarget, Xmen.T), inv( dot(Xmen, Xmen.T) + params['regul_matrix'] ))
        else:
            # Compute Wout with a pseudoinverse
            self.Wout = dot( Ytarget, pinv(Xmen) )


# Online training
class DR(Reservoir):
    def train(self, **params):
        n = params['learning_rate']
        data = params['data']
        target = params['target']

        y = self.compute(data)
        err = target - y 
        Xwrap = vstack((1, vstack(data),self.X))

        for j in xrange(self.L):
           for i in xrange(1+self.K+self.N):
               self.Wout[j, i] += n*err[j]*Xwrap[i]

        return y




class BPDC(Reservoir):
    def __init__(self, K, N, L, Win, W, f, leaking_rate, rho_factor):
        Reservoir.__init__(self, K, N, L, Win, W, f, leaking_rate, rho_factor)
        self.fprim = derived[f]
        self.err_mem = [0 for _ in xrange(L)]

    def train(self, **params):
        data = params['data']
        target = params['target']
        n = params['learning_rate']
        e = params['regul_const']

        y = self.compute(data)
        err = y - target
        x = vstack((1, vstack(data), self.X))

        for i in xrange(self.L):
            gamma = (1-self.a)*sum(self.err_mem)-err[i]
            for j in xrange(1+self.K+self.N):
                self.Wout[i, j] += (n/self.a)*(self.f(x[j])/(sum(power(self.f(x[-self.N:]), 2))+e))*gamma

        self.err_mem = err
