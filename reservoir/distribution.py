from numpy import *

def uniformDistribution(shape, param):
    return (param["max"]-param['min'])*random.rand(shape[0], shape[1]) + param['min']

def gaussianDistribution(shape, param):
    return param['sigma'] * random.randn(shape[0], shape[1]) + param['mu']

def sparseDistribution(shape, param):
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