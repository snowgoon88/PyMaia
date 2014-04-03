from numpy import *
from matplotlib.pyplot import *
import scipy.linalg
import simplejson, getopt, sys

verbose = False

class ESN:
    def generate(self, K, N, L, a, rho_factor):
        self.a = a
        self.x = zeros((N, 1))
        self.Wout = zeros((L, K+N+L))
        # random value [-0.5;0.5]
        self.Win = random.rand(N, 1+K)-0.5
        # random value [-0.5;0.5]
        self.W = random.rand(N, N)-0.5
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


def runESN(K, N, L, a, rho_factor, data, initLen, trainLen, testLen):
    global verbose

    Xmem = zeros((1+K+N, trainLen - initLen))
    Ytarget = data[None, initLen+1:trainLen+1]

    # init network
    network = ESN()
    # generate a reservoir
    if verbose :
        print 'Generating a reservoir... '
    network.generate(K, N, L, a, rho_factor)

    # initialisation transient
    if verbose :
        print 'Initialisation step...'
    for t in range(initLen):
        network.feed(data[t])

    # feed
    if verbose :
        print 'Feeding step...'
    for t in range(initLen, trainLen):
        network.feed(data[t])
        Xmem[:, t-initLen] = vstack((1, data[t], network.x))[:,0]

    # train
    if verbose:
        print 'Training step...'
    network.train(Ytarget, Xmem, 1e-8*eye(1+K+N))

    # test
    if verbose:
        print 'Testing step...'
    Ymem = zeros((L, testLen))
    u = data[trainLen]
    for t in range(testLen):
        Ymem[:, t] = network.solve(u)
        u = Ymem[:, t]

    if verbose:
        print 'done'

    figure(1).clear()
    plot(data[trainLen+1:trainLen+testLen+1], 'g')
    plot(Ymem.T, 'b')
    legend(['Target signal', 'Predited signal'])
    show()

def main():
    try:
        opts, files = getopt.getopt(sys.argv[1:], "vh", ["help", "verbose"])
    except getopt.GetoptError as err:
        print str(err)
        usage()
        sys.exit(1)

    global verbose
    for opt, arg in opts:
        if opt in ('-v','--verbose'):
            verbose = True
        elif opt in ('-h','--help'):
            usage()
            sys.exit(0)
    
    for json_file in files :
        fd=open(json_file)
        json_data = simplejson.load(fd)
        fd.close()

        # ESN setup
        K = json_data['esn']['K']
        N = json_data['esn']['N']
        L = json_data['esn']['L']
        a = json_data['esn']['a']
        rho_factor = json_data['esn']['rho_factor']

        # data setup
        data = None
        if json_data['data']['type'] == 'txt' :
            data = loadtxt(json_data['data']['path'])
        else :
            print 'Unsupported data type: ',json_data['data']['type']
            sys.exit(2)
        init_len=json_data['data']['initLen']
        train_len=json_data['data']['trainLen']
        test_len=json_data['data']['testLen']

        if verbose:
            print 'Running ', json_file
        runESN(K, N, L, a, rho_factor, data, init_len, train_len, test_len)

def usage():
    print 'python ESN.py [--help|-h] [--verbose|-v] testFiles...'

if __name__ == "__main__":
    main()
