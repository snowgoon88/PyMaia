from numpy import *

def generation(network, data, init_len, train_len, test_len, regul_coef):
    for t in range(init_len):
        network.input(data[:, t])

    Xmem = zeros((1+network.N+network.K, train_len))
    for t in range(init_len, int(train_len+init_len)):
        network.input(data[:, t])
        Xmem[:, t-init_len] = vstack((1, vstack(data[:, t]), network.X))[:,0]

    network.train(Ytarget=data[:, init_len+1:train_len+init_len+1], Xmem=Xmem, regMatrix=regul_coef*eye(1+network.N+network.K))

    Ymem = zeros((network.L, test_len))
    u = data[:, train_len+init_len]
    for t in range(test_len):
        u = Ymem[:, t] = hstack(network.compute(u))

    return data[:, init_len+train_len+1:init_len+train_len+test_len+1], Ymem

def rappelGeneration(network, data, init_len, train_len, regul_coef):
    for t in range(init_len):
        network.input(data[:, t])

    Xbak = network.X

    Xmem = zeros((1+network.K+network.N, train_len))
    for t in range(init_len, train_len + init_len):
        network.input(data[:, t])
        Xmem[:, t-init_len] = vstack((1, vstack(data[:, t]), network.X))[:,0]

    network.train(Ytarget=data[:, init_len+1:train_len+init_len+1], Xmem=Xmem, regMatrix=regul_coef*eye(1+network.K+network.N))

    network.X = Xbak

    Ymem = zeros((network.L, train_len))
    u = data[:, init_len]
    for t in range(train_len):
        u = Ymem[:, t] = hstack(network.compute(u))

    return data[:, init_len+1:train_len+init_len+1], Ymem   

def prediction(network, data, init_len, train_len, test_len, regul_coef):
    for t in range(init_len):
        network.input(data[:, t])

    Xmem = zeros((1+network.N+network.K, train_len))
    for t in range(init_len, int(train_len+init_len)):
        network.input(data[:, t])
        Xmem[:, t-init_len] = vstack((1, vstack(data[:, t]), network.X))[:,0]

    network.train(Ytarget=data[:, init_len+1:train_len+init_len+1], Xmem=Xmem, regMatrix=regul_coef*eye(1+network.N+network.K))

    Ymem = zeros((network.L, test_len))
    for t in range(test_len):
        Ymem[:, t] = hstack(network.compute(data[:, init_len+train_len+t]))

    return data[:, init_len+train_len+1:init_len+train_len+test_len+1], Ymem

def rappelPrediction(network, data, init_len, train_len, regul_coef):
    for t in range(init_len):
        network.input(data[:, t])

    Xbak = network.X

    Xmem = zeros((1+network.K+network.N, train_len))
    for t in range(init_len, train_len + init_len):
        network.input(data[:, t])
        Xmem[:, t-init_len] = vstack((1, vstack(data[:, t]), network.X))[:,0]

    network.train(Ytarget=data[:, init_len+1:train_len+init_len+1], Xmem=Xmem, regMatrix=regul_coef*eye(1+network.K+network.N))

    network.X = Xbak

    Ymem = zeros((network.L, train_len))
    for t in range(train_len):
        Ymem[:, t] = hstack(network.compute(data[:, init_len+train_len+t]))

    return data[:, init_len+1:train_len+init_len+1], Ymem

def classification(network, data, target, init_len, train_len, test_len):
    for t in range(init_len):
        network.input(data[:, t])

    Xmem = zeros((1+network.K+network.N, train_len))
    for t in range(init_len, int(train_len+init_len)):
        network.input(data[:, t])
        Xmem[:, t-init_len] = vstack((1, vstack(data[:, t]), network.X))[:,0]

    network.train(Ytarget=target[:, init_len:train_len+init_len], Xmem=Xmem, regMatrix=regul_coef*eye(1+network.K+network.N))

    Ymem = zeros((network.L, test_len))
    for t in range(test_len):
        u = data[:, init_len+train_len+t]
        Ymem[:, t] = hstack(network.compute(u))

    return target[:, init_len+train_len:init_len+train_len+test_len], Ymem

def rappelClassification(network, data, target, init_len, train_len):
    for t in range(init_len):
        network.input(data[:, t])

    xbak = network.X

    Xmem = zeros((1+network.K+network.N, train_len))
    for t in range(init_len, int(train_len+init_len)):
        network.input(data[:, t])
        Xmem[:, t-init_len] = vstack((1, vstack(data[:, t]), network.X))[:,0]

    network.train(Ytarget=target[:, init_len:train_len+init_len], Xmem=Xmem, regMatrix=regul_coef*eye(1+network.K+network.N))

    Ymem = zeros((network.L, train_len))
    for t in range(train_len):
        u = data[:, init_len+t]
        Ymem[:, t] = hstack(network.compute(u))

    return target[:, init_len:init_len+train_len], Ymem

def classificationPrediction(K, N, L, Win, W, leaking_rate, rho_factor, regul_coef, data, target, init_len, train_len, test_len):
    for t in range(init_len):
        network.input(data[:, t])

    Xmem = zeros((1+network.K+network.N, train_len))
    for t in range(init_len, int(train_len+init_len)):
        network.input(data[:, t])
        Xmem[:, t-init_len] = vstack((1, vstack(data[:, t]), network.X))[:,0]

    network.train(Ytarget=concatenate((data[:, init_len+1:train_len+init_len+1], target[:, init_len:train_len+init_len])), Xmem=Xmem, regMatrix=regul_coef*eye(1+network.K+network.N))

    Ymem = zeros((network.L+network.K, test_len))
    for t in range(test_len):
        u = data[:, init_len+train_len+t]
        Ymem[:, t] = hstack(network.compute(u))

    return target[:, init_len+train_len:init_len+train_len+test_len], Ymem[K:,:]