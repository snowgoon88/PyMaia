#! /usr/bin/python

import argparse
import json
import numpy as np
from reservoir.network import ESN


def main():
    parser = argparse.ArgumentParser(description="Learning of sequence rules with Echo State Network")
    parser.add_argument('--esn', action='store', type=argparse.FileType('r'))
    parser.add_argument('--esnSeed', action='store', type=int)
    parser.add_argument('--seq', action='store', type=argparse.FileType('r'))
    parser.add_argument('--seqSeed', action='store', type=int)
    parser.add_argument('--init', action='store', type=int)
    parser.add_argument('--train', action='store', type=int)
    parser.add_argument('--test', action='store', type=int)
    parser.add_argument('--regul', action='store', type=float)
    args = parser.parse_args()

    # Generating data
    seq_param = json.load(args.seq)
    args.seq.close()

    np.random.seed(args.seqSeed)
    seq = []
    for i in xrange(max([seq_param['rules'][x]['delay'] for x in seq_param['rules']])+1):
        seq.append(chr(65+np.random.randint(0, len(seq_param['encode']))))
    while len(seq) < args.init + args.train + args.test +1:
        seq.append(seq_param['rules'][seq[-1]][seq[-1-seq_param['rules'][seq[-1]]['delay']]])

    data = np.zeros((len(seq_param['encode']), len(seq)))
    for i in xrange(len(seq)):
        data[:, i] = np.array(seq_param['encode'][seq[i]])

    learnQ = {}
    for i in seq_param['encode']:
        learnQ[i] = {}
        for j in seq_param['encode']:
            learnQ[i][j] = 0

    # Generating esn
    esn_param = json.load(args.esn)
    args.esn.close()

    np.random.seed(args.esnSeed)
    esn = ESN(**esn_param)

    # transient
    for i in xrange(args.init):
        esn.input(data[:, i])
    # training step
    Xmem = np.zeros((1+esn_param['K']+esn_param['N'], args.train))
    for i in xrange(args.train):
        esn.input(data[:, i+args.init])
        learnQ[seq[args.init + i]][seq[args.init + i +1]] +=1
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
    accByRule = {}
    lenByRule = {}
    for i in xrange(len(seq_param['encode'])):
        accByRule[i]={}
        lenByRule[i]={}
        for j in xrange(len(seq_param['encode'])):
            accByRule[i][j]=0
            lenByRule[i][j]=0
    
    Ytarget = data[:, args.init+args.train+1:args.init+args.train+args.test+1]
    Xinput = data[:, args.init+args.train:args.init+args.train+args.test]
    for i in xrange(args.test):
        xletter = np.where(Xinput[:, i]==max(Xinput[:, i]))[0][0]
        tmp1 = np.where(Ytarget[:, i]==max(Ytarget[:, i]))[0][0]

        lenByRule[xletter][tmp1]+=1
        if np.isnan(sum(Y[:, i])):
            print "[WARNING] NaN in reservoir output"
            break

        tmp2 = np.where(Y[:, i]==max(Y[:, i]))[0][0]
        if tmp1 == tmp2:
            accByRule[xletter][tmp1]+=1

    print "LEARNING"
    for i in xrange(len(seq_param['encode'])):
        print "* %c:"%(65+i), sum(learnQ[chr(65+i)].values())
        for j in xrange(len(seq_param['encode'])):
            print "\t -> %c: %i"%(chr(65+j), learnQ[chr(65+i)][chr(65+j)])
    print "ACCURACY"
    for i in xrange(len(seq_param['encode'])):
        print "* %c:"%(65+i), float(sum(accByRule[i].values()))/sum(lenByRule[i].values())
        for j in xrange(len(seq_param['encode'])):
            if lenByRule[i][j] == 0 :
                print "\t-> %c (0): _"%(65+j)
            else:
                print "\t-> %c (%i):"%((65+j), lenByRule[i][j]), float(accByRule[i][j])/lenByRule[i][j]
    print "Global:", float(sum([ sum(accByRule[x].values()) for x in accByRule ])) / sum([sum(lenByRule[x].values()) for x in lenByRule])

if __name__ == "__main__":
    main()
