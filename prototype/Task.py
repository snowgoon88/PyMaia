from numpy import *
from Reservoir import ESN

def classification(K, N, L, Win, W, leaking_rate, rho_factor, regul_coef, data, target, initLen, trainLen, testLen):
    #print 'Step 1/5: Reservoir generation'
    network = ESN(K, N, L, Win, W, leaking_rate, rho_factor)

    #print 'Step 2/5: Transient initialization'
    for t in range(initLen):
        network.input(data[:, t])

    #print 'Step 3/5: Learning phase'
    Xmem = zeros((1+K+N, trainLen))
    for t in range(initLen, int(trainLen+initLen)):
        network.input(data[:, t])
        Xmem[:, t-initLen] = vstack((1, vstack(data[:, t]), network.X))[:,0]

    #print 'Step 4/5: Wout computation'
    network.train(Ytarget=target[:, initLen:trainLen+initLen], Xmem=Xmem, regMatrix=regul_coef*eye(1+K+N))

    #print 'Step 5/5: Testing phase'
    Ymem = zeros((L, testLen))
    for t in range(testLen):
        u = data[:, initLen+trainLen+t]
        Ymem[:, t] = hstack(network.compute(u))

    return target[:, initLen+trainLen:initLen+trainLen+testLen], Ymem

def rappelClassification(K, N, L, Win, W, leaking_rate, rho_factor, regul_coef, data, target, initLen, trainLen):
    #print 'Step 1/5: Reservoir generation'
    network = ESN(K, N, L, Win, W, leaking_rate, rho_factor)

    #print 'Step 2/5: Transient initialization'
    for t in range(initLen):
        network.input(data[:, t])

    # save the internal state
    xbak = network.X

    #print 'Step 3/5: Learning phase'
    Xmem = zeros((1+K+N, trainLen))
    for t in range(initLen, int(trainLen+initLen)):
        network.input(data[:, t])
        Xmem[:, t-initLen] = vstack((1, vstack(data[:, t]), network.X))[:,0]

    #print 'Step 4/5: Wout computation'
    network.train(Ytarget=target[:, initLen:trainLen+initLen], Xmem=Xmem, regMatrix=regul_coef*eye(1+K+N))

    network.X = xbak
    #print 'Step 5/5: Testing phase'
    Ymem = zeros((L, trainLen))
    for t in range(trainLen):
        u = data[:, initLen+t]
        Ymem[:, t] = hstack(network.compute(u))

    return target[:, initLen:initLen+trainLen], Ymem

def classificationPrediction(K, N, L, Win, W, leaking_rate, rho_factor, regul_coef, data, target, initLen, trainLen, testLen):
    #print 'Step 1/5: Reservoir generation'
    network = ESN(K, N, L+K, Win, W, leaking_rate, rho_factor)

    #print 'Step 2/5: Transient initialization'
    for t in range(initLen):
        network.input(data[:, t])

    #print 'Step 3/5: Learning phase'
    Xmem = zeros((1+K+N, trainLen))
    for t in range(initLen, int(trainLen+initLen)):
        network.input(data[:, t])
        Xmem[:, t-initLen] = vstack((1, vstack(data[:, t]), network.X))[:,0]

    #print 'Step 4/5: Wout computation'
    network.train(Ytarget=concatenate((data[:, initLen+1:trainLen+initLen+1], target[:, initLen:trainLen+initLen])), Xmem=Xmem, regMatrix=regul_coef*eye(1+K+N))

    #print 'Step 5/5: Testing phase'
    Ymem = zeros((L+K, testLen))
    for t in range(testLen):
        u = data[:, initLen+trainLen+t]
        Ymem[:, t] = hstack(network.compute(u))

    return target[:, initLen+trainLen:initLen+trainLen+testLen], Ymem[K:,:]

def generation(K, N, L, Win, W, leaking_rate, rho_factor, regul_coef, data, initLen, trainLen, testLen):
    #print 'Step 1/5: Reservoir generation'
    network = ESN(K, N, L, Win, W, leaking_rate, rho_factor)

    #print 'Step 2/5: Transient initialization'
    for t in range(initLen):
        network.input(data[:, t])

    #print 'Step 3/5: Learning phase'
    Xmem = zeros((1+K+N, trainLen))
    for t in range(initLen, int(trainLen+initLen)):
        network.input(data[:, t])
        Xmem[:, t-initLen] = vstack((1, vstack(data[:, t]), network.X))[:,0]

    #print 'Step 4/5: Wout computation'
    network.train(Ytarget=data[:, initLen+1:trainLen+initLen+1], Xmem=Xmem, regMatrix=regul_coef*eye(1+K+N))

    #print 'Step 5/5: Testing phase'
    Ymem = zeros((L, testLen))
    u = data[:, trainLen+initLen]
    for t in range(testLen):
        Ymem[:, t] = hstack(network.compute(u))
        u = Ymem[:, t]

    return data[:, initLen+trainLen+1:initLen+trainLen+testLen+1], Ymem


def rappelGeneration(K, N, L, Win, W, leaking_rate, rho_factor, regul_coef, data, initLen, trainLen):
    #print 'Step 1/5: Reservoir generation'
    network = ESN(K, N, L, Win, W, leaking_rate, rho_factor)
    
    #print 'Step 2/5: Transient initialization'
    for t in range(initLen):
        network.input(data[:, t])

    # save the internal state
    xbak = network.X

    #print 'Step 3/5: Learning phase'
    Xmem = zeros((1+K+N, trainLen))
    for t in range(initLen, trainLen + initLen):
        network.input(data[:, t])
        Xmem[:, t-initLen] = vstack((1, vstack(data[:, t]), network.X))[:,0]

    #print 'Step 4/5: Wout computation'
    network.train(Ytarget=data[:, initLen+1:trainLen+initLen+1], Xmem=Xmem, regMatrix=regul_coef*eye(1+K+N))

    network.X = xbak
    #print 'Step 5/5: Testing phase'
    Ymem = zeros((L, trainLen))
    u = data[:, initLen]
    for t in range(trainLen):
        Ymem[:, t] = hstack(network.compute(u))
        u = Ymem[:, t]

    return data[:, initLen+1:trainLen+initLen+1], Ymem   


def prediction(K, N, L, Win, W, leaking_rate, rho_factor, regul_coef, data, initLen, trainLen, testLen):
    #print 'Step 1/5: Reservoir generation'
    network = ESN(K, N, L, Win, W, leaking_rate, rho_factor)

    #print 'Step 2/5: Transient initialization'
    for t in range(initLen):
        network.input(data[:, t])

    #print 'Step 3/5: Learning phase'
    Xmem = zeros((1+K+N, trainLen))
    for t in range(initLen, int(trainLen+initLen)):
        network.input(data[:, t])
        Xmem[:, t-initLen] = vstack((1, vstack(data[:, t]), network.X))[:,0]

    #print 'Step 4/5: Wout computation'
    network.train(Ytarget=data[:, initLen+1:trainLen+initLen+1], Xmem=Xmem, regMatrix=regul_coef*eye(1+K+N))

    #print 'Step 5/5: Testing phase'
    Ymem = zeros((L, testLen))
    for t in range(testLen):
        u = data[:, initLen+trainLen+t]
        Ymem[:, t] = hstack(network.compute(u))

    return data[:, initLen+trainLen+1:initLen+trainLen+testLen+1], Ymem


def rappelPrediction(K, N, L, Win, W, leaking_rate, rho_factor, regul_coef, data, initLen, trainLen):
    #print 'Step 1/5: Reservoir generation'
    network = ESN(K, N, L, Win, W, leaking_rate, rho_factor)
    
    #print 'Step 2/5: Transient initialization'
    for t in range(initLen):
        network.input(data[:, t])

    # save the internal state
    xbak = network.X

    #print 'Step 3/5: Learning phase'
    Xmem = zeros((1+K+N, trainLen))
    for t in range(initLen, trainLen + initLen):
        network.input(data[:, t])
        Xmem[:, t-initLen] = vstack((1, vstack(data[:, t]), network.X))[:,0]

    #print 'Step 4/5: Wout computation'
    network.train(Ytarget=data[:, initLen+1:trainLen+initLen+1], Xmem=Xmem, regMatrix=regul_coef*eye(1+K+N))

    network.X = xbak
    #print 'Step 5/5: Testing phase'
    Ymem = zeros((L, trainLen))
    for t in range(trainLen):
        u = data[:, initLen+t]
        Ymem[:, t] = hstack(network.compute(u))
        
    return data[:, initLen+1:trainLen+initLen+1], Ymem   
