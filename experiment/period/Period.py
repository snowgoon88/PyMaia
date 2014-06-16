#! /usr/bin/python

import argparse
import sys
import json
import numpy as np
from reservoir.network import ESN


def main():
    parser = argparse.ArgumentParser(description="Learning of sequence rules with Echo State Network")
    parser.add_argument('--esn', action='store', type=argparse.FileType('r'))
    parser.add_argument('--period', action='store', type=int)
    parser.add_argument('--cardinal', action='store', type=int)
    parser.add_argument('--init', action='store', type=int)
    parser.add_argument('--train', action='store', type=int)
    parser.add_argument('--test', action='store', type=int)
    parser.add_argument('--regul', action='store', type=float)
    parser.add_argument('--esnSeed', action='store', type=int)
    parser.add_argument('--seqSeed', action='store', type=int)
    parser.add_argument('--output', action='store', type=argparse.FileType('a'))
    args = parser.parse_args()

    # Generating esn
    esn_param = json.load(args.esn)
    args.esn.close()

    if args.esnSeed:
        np.random.seed(args.esnSeed)
    esn = ESN(**esn_param)

    # Generating data
    np.random.seed(args.seqSeed)
    period = np.zeros((args.cardinal, args.period))
    for i in xrange(len(period.T)):
        period[np.random.randint(args.cardinal), i]=1

    data = np.zeros((args.cardinal, args.init+args.train+args.test+1))
    for i in xrange(len(data.T)):
        data[:, i] = period[:, i%len(period.T)]

    # transient
    for i in xrange(args.init):
        esn.input(data[:, i])
    # training step
    Xmem = np.zeros((1+esn_param['K']+esn_param['N'], args.train))
    for i in xrange(args.train):
        esn.input(data[:, i+args.init])
        Xmem[:, i] = np.vstack((1, np.vstack(data[:, i+args.init]), esn.X))[:,0]
    # learning
    esn.train(data[:, args.init+1:args.init+args.train+1], 
              Xmem, 
              None if args.regul is None else args.regul*np.eye(1+esn_param['K']+esn_param['N']))
    # testing
    Y = np.zeros((esn_param['L'], args.test))
    for i in xrange(args.test):
        Y[:, i] = np.hstack(esn.compute(data[:, args.init+args.train+i]))

    # print perf and caetera
    acc = 0
    Ytarget = data[:, args.init+args.train+1:args.init+args.train+args.test+1]
    for i in xrange(args.test):
        tmp1 = np.where(Ytarget[:, i]==max(Ytarget[:, i]))[0][0]
        tmp2 = np.where(Y[:, i]==max(Y[:, i]))[0][0]
        if tmp1 == tmp2:
            acc+=1

    args.output.write("%s,%i,%i,%f\n"%(args.period, 
                                       esn_param['N'], 
                                       args.esnSeed, 
                                       float(acc)/args.test))
    args.output.close()


if __name__ == "__main__":
    main()
