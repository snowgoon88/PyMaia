from scipy.linalg import eig, inv, pinv, lstsq
from numpy import *
from distribution import *

class ESN:
    def __init__(self, K, N, L, Win, W, leaking_rate, rho_factor):
        ## Reservoir generation
        # constant
        self.a = leaking_rate
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
        self.X = (1-self.a)*self.X + self.a*tanh( dot(self.Win, vstack((1, vstack(data)))) + dot(self.W, self.X) )

    def compute(self, data):
        self.input(data)
        return dot(self.Wout, vstack((1,vstack(data),self.X)) )

    def train(self, Ytarget, Xmem, regul_matrix):
        if not regul_matrix is None:
            # Compute Wout with a ridge regression
            self.Wout = dot( dot(Ytarget, Xmem.T), inv( dot(Xmem, Xmem.T) + regul_matrix ))
        else:
            # Compute Wout with numpy.linalg.lstsq
            self.Wout = lstsq(Xmem.T, Ytarget.T)[0].T
            # Compute Wout with a pseudoinverse
            #self.Wout = dot( Ytarget, pinv(Xmem) )

        if isnan(sum(self.Wout)):
            print "[WARNING] Wout contains NaN !"

        if isinf(sum(self.Wout)):
            print "[WARNING] Wout contains inf !"