from numpy import *
from matplotlib.pyplot import *
import scipy.linalg

class ESN:
    def generate(self, K, N, L, a):
        self.a = a
        self.x = zeros((N, 1))
        self.Wout = zeros((L, K+N+L))
        # random value [-0.5;0.5]
        self.Win = random.rand(N, 1+K)-0.5
        # random value [-0.5;0.5]
        self.W = random.rand(N, N)-0.5

    def feed(self, data):
        self.x = (1-a)*self.x + a*tanh( dot(self.Win, vstack((1, data))) + dot(self.W, self.x) )

    def solve(self, data):
        self.feed(data)
        return dot(self.Wout, vstack((1,data,self.x)) )

    def train(self, Ytarget, Xmem, regMatrix):
        # ridge regression
        self.Wout = dot( dot(Ytarget, Xmem.T), linalg.inv( dot(Xmem, Xmem.T) + regMatrix ) )

K=1
N=1000
L=1
a=0.3

initLen=100
trainLen=2000
testLen=2000

print 'Loading data... '
data = loadtxt('MackeyGlass_t17.txt')
Xmem = zeros((1+K+N, trainLen - initLen))
Ytarget = data[None, initLen+1:trainLen+1]

# init network
network = ESN()
# generate a reservoir
print 'Generation... '
network.generate(K, N, L, a)
# Spectral radius tuning
rhoW = max( abs( linalg.eig(network.W) ) )
network.W *= 1.25 / rhoW
# initialisation transient
print 'Initialisation...'
for t in range(initLen):
    network.feed(data[t])
# feed
print 'Feeding...'
for t in range(initLen, trainLen):
    network.feed(data[t])
    Xmem[:, t-initLen] = vstack((1, data[t], network.x))[:,0]
# train
print 'Training...'
network.train(Ytarget, Xmem, 1e-8*eye(1+K+N))
# test
print 'Testing...'
Ymem = zeros((L, testLen))
u = data[trainLen]
for t in range(testLen):
    Ymem[:, t] = network.solve(u)
    u = Ymem[:, t]

figure(1).clear()
plot(data[trainLen+1:trainLen+testLen+1], 'g')
plot(Ymem.T, 'b')
legend(['Target signal', 'Predited signal'])
show()
