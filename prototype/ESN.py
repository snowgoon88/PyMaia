from numpy import *
from matplotlib.pyplot import *
import scipy.linalg
import json, getopt, sys

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


def main():
    try:
        opts, files = getopt.getopt(sys.argv[1:], "d", ["help"])
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

        print '=== Running', json_file, "==="

        data = None
        if json_data['data']['type'] == 'MackeyGlass' :
            tmp = loadtxt(json_data['data']['path'])
            data = zeros((1, len(tmp)))
            for i in range(len(tmp)):
                data[:, i] = tmp[i]

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

        elif json_data['data']['type'] == 'Sequential' :
            tmp = loadtxt(json_data['data']['path'], dtype=string0)
            data = zeros((json_data['esn']['K'], len(tmp)))
            for i in range(len(tmp)):
                data[:, i] = array(json_data['data']['encode'][tmp[i]])

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

            print_Ytarget = []
            print_Y = []
            acc = []
            predictOK = 0
            for i in range(json_data['data']['test_len']):
                print_Ytarget.append(where(Ytarget[:, i]==1)[0][0])
                print_Y.append(where(Y[:, i]==max(Y[:, i]))[0][0])
                if print_Y[i] == print_Ytarget[i]:
                    predictOK+=1
                acc.append(float(predictOK)/(i+1))

            if display:
                k+=1
                figure(k).clear()
                figure(k).canvas.set_window_title(json_file)
                subplot(211)
                yticks(range(26), [chr(97 + x) for x in range(26)])
                plot(print_Ytarget, 'wo')
                plot(print_Y, 'r+')
                legend(['Ytarget', 'Y'])
                subplot(212)
                plot(acc, 'k')
                legend(['Accuracy'])

        else :
            print 'Unsupported data type: ',json_data['data']['type']
            sys.exit(2)

    if display:
        show()


def usage():
    print 'python ESN.py [--help] [-d] TEST_FILE... )'


if __name__ == "__main__":
    main()
