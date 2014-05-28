from numpy import *

def generationBPDC(network, data, init_len, train_len, test_len, learning_rate, regul_const):
    for t in range(init_len):
        network.input(data[:, t])

    for t in range(init_len, train_len+init_len):
        network.train(data=data[:, t], target=data[:, t+1], learning_rate=learning_rate, regul_const=regul_const)

    Ymem = zeros((network.L, test_len))
    u = data[:, train_len+init_len]
    for t in range(test_len):
        u = Ymem[:, t] = hstack(network.compute(u))

    return data[:, init_len+train_len+1:init_len+train_len+test_len+1], Ymem