from numpy import *
from matplotlib.pyplot import *

def generationDR(network, data, init_len, train_len, test_len, learning_rate):
    for t in range(init_len):
        network.input(data[:, t])

    Ymem = zeros((network.L, test_len+train_len))

    X = vstack((network.X.T))

    W = vstack((network.Wout))
    for t in range(init_len, train_len+init_len):
        Ymem[:, t-init_len] = network.train(data=data[:, t], target=data[:, t+1], learning_rate=learning_rate)
        W = vstack((W, network.Wout.flatten()))
        X = vstack((X, network.X.T))
    
    u = data[:, train_len+init_len]
    for t in range(train_len, train_len+test_len):
        u = Ymem[:, t] = hstack(network.compute(u))
        X = vstack((X, network.X.T))

    figure().canvas.set_window_title("W")
    plot(W)

    figure().canvas.set_window_title("X")
    plot(X)

    return data[:, init_len+1:init_len+train_len+test_len+1], Ymem


def predictionDR(network, data, init_len, train_len, test_len, learning_rate):
    for t in range(init_len):
        network.input(data[:, t])

    Ymem = zeros((network.L, test_len+train_len))

    X = vstack((network.X.T))

    W = vstack((network.Wout))
    for t in range(init_len, train_len+init_len):
        Ymem[:, t-init_len] = network.train(data=data[:, t], target=data[:, t+1], learning_rate=learning_rate)
        W = vstack((W, network.Wout.flatten()))
        X = vstack((X, network.X.T))
    
    for t in range(train_len+init_len, train_len+init_len+test_len):
        Ymem[:, t-init_len] = hstack(network.compute(data[:, t]))
        X = vstack((X, network.X.T))

    figure().canvas.set_window_title("W")
    plot(W)

    figure().canvas.set_window_title("X")
    plot(X)

    return data[:, init_len+1:init_len+train_len+test_len+1], Ymem