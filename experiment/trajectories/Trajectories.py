#! /usr/bin/python

import argparse
import json
import numpy as np
from reservoir.network import ESN

alpha = 0.7

def trajX(t, beta, petals):
    return alpha*np.sin(t+beta)*np.sin(petals*t)

def trajY(t, beta, petals):
    return alpha*np.cos(t+beta)*np.sin(petals*t)

def main():
    parser = argparse.ArgumentParser(description="Learning of sequence rules with Echo State Network")
    parser.add_argument('--esn', action='store', type=argparse.FileType('r'))
    parser.add_argument('--esnSeed', action='store', type=int)
    parser.add_argument('--cardinal', action='store', type=int)
    parser.add_argument('--trajSeed', action='store', type=int)
    parser.add_argument('--init', action='store', type=int)
    parser.add_argument('--train', action='store', type=int)
    parser.add_argument('--test', action='store', type=int)
    parser.add_argument('--regul', action='store', type=float)
    parser.add_argument('--output', action='store', type=argparse.FileType('a'))
    args = parser.parse_args()

    # generating data
    Xinput = np.zeros((2, args.init+args.train+args.test+30-(args.init+args.train+args.test)%30))
    Ytarget = -0.7*np.ones((args.cardinal, args.init+args.train+args.test))

    np.random.seed(args.trajSeed)
    t = np.linspace(0, 2*np.pi, 30)
    for i in xrange((args.init+args.train+args.test)/30+1):
        k = np.random.randint(1, args.cardinal+1)
        Xinput[0, 30*i:30*(i+1)] = trajX(t, 2*np.pi*np.random.random(), k)
        Xinput[1, 30*i:30*(i+1)] = trajY(t, 2*np.pi*np.random.random(), k)
        Ytarget[k-1, 30*i:30*(i+1)] = -1*Ytarget[k-1, 30*i:30*(i+1)]

    # Generating esn
    esn_param = json.load(args.esn)
    args.esn.close()

    np.random.seed(args.esnSeed)
    esn = ESN(**esn_param)

    # transient
    for i in xrange(args.init):
        esn.input(Xinput[:, i])
    # training step
    Xmem = np.zeros((1+esn_param['K']+esn_param['N'], args.train))
    for i in xrange(args.train):
        esn.input(Xinput[:, i+args.init])
        Xmem[:, i] = np.vstack((1, np.vstack(Xinput[:, i+args.init]), esn.X))[:,0]
    # learning
    esn.train(Ytarget[:, args.init:args.init+args.train], 
              Xmem, 
              None if args.regul is None else args.regul*np.eye(1+esn_param['K']+esn_param['N']))
    # testing
    Y = np.zeros((esn_param['L'], args.test))
    for i in xrange(args.test):
        Y[:, i] = np.hstack(esn.compute(Xinput[:, args.init+args.train+i]))

    # print perf and caetera
    acc = 0
    Ytest = Ytarget[:, args.init+args.train:args.init+args.train+args.test]
    for i in xrange(args.test):
        tmp1 = np.where(Ytest[:, i]==max(Ytest[:, i]))[0][0]
        tmp2 = np.where(Y[:, i]==max(Y[:, i]))[0][0]
        if tmp1 == tmp2:
            acc+=1

    print "Accuracy:", float(acc)/args.test

if __name__ == "__main__":
    main()