#! /usr/bin/python

import argparse
import json
import numpy as np
from reservoir.network import ESN


def main():
    parser = argparse.ArgumentParser(description="Learning of sequence rules with Echo State Network",
                                     epilog="Write by NiZiL")
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 2.0')
    parser.add_argument('--esn', action='store', type=argparse.FileType('r'))
    parser.add_argument('--seq', action='store', type=argparse.FileType('r'))
    parser.add_argument('--init', action='store', type=int)
    parser.add_argument('--train', action='store', type=int)
    parser.add_argument('--test', action='store', type=int)
    parser.add_argument('--regul', action='store', type=float)
    parser.add_argument('--seed', action='store', type=int)
    args = parser.parse_args()

    # Generating esn
    esn_param = json.load(args.esn)
    args.esn.close()

    print "ESN:", esn_param
    if args.seed:
        np.random.seed(args.seed)
    esn = ESN(**esn_param)

    # Generating data
    seq_param = json.load(args.seq)
    args.seq.close()

    delay = []
    for l in seq_param['rules']:
        delay.append(seq_param['rules'][l]['delay'])

    np.random.seed(seq_param['seed'])
    tmp = []
    for i in xrange(max(delay)):
        tmp.append(chr(65+np.random.randint(0, len(delay))))
    while len(tmp) < args.init + args.train + args.test +1:
        tmp.append(seq_param['rules'][tmp[-1]][tmp[-1*seq_param['rules'][tmp[-1]]['delay']]])

    data = np.zeros((len(seq_param['encode']), len(tmp)))
    for i in xrange(len(tmp)):
        data[:, i] = np.array(seq_param['encode'][tmp[i]])


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
    accByLetter = {}
    lenByLetter = {}
    for i in xrange(len(delay)):
        accByLetter[i]=0
        lenByLetter[i]=0
    
    Ytarget = data[:, args.init+args.train+1:args.init+args.train+args.test+1]
    Xinput = data[:, args.init+args.train:args.init+args.train+args.test]
    for i in xrange(args.test):
        xletter = np.where(Xinput[:, i]==max(Xinput[:, i]))[0][0]
        tmp1 = np.where(Ytarget[:, i]==max(Ytarget[:, i]))[0][0]

        lenByLetter[xletter]+=1
        if np.isnan(sum(Y[:, i])):
            print "[WARNING] NaN in reservoir output"
            break

        tmp2 = np.where(Y[:, i]==max(Y[:, i]))[0][0]
        if tmp1 == tmp2:
            acc+=1
            accByLetter[xletter]+=1

    for i in accByLetter:
        if lenByLetter[i] == 0 :
            print "There is no %c"%(65+i)
        else:
            print "Accuracy for %i %c:"%(lenByLetter[i],(65+i)), float(accByLetter[i])/lenByLetter[i]
    print "Accuracy:", float(acc)/args.test


if __name__ == "__main__":
    main()
