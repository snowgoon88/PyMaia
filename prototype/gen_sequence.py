#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import json
import numpy as np

# ****************************************************************************
# ******************************************************************** SEQ_GEN
# 20/06/2014 - tiré de experiment/rules/Rules.py 
# ****************************************************************************
def main():
    parser = argparse.ArgumentParser(description="Generate a sequence of letter from a json file")
    parser.add_argument('--seq', action='store', type=argparse.FileType('r'))
    parser.add_argument('--seqSeed', action='store', type=int, default=None)
    parser.add_argument('--length', action='store', type=int, default=10)
    args = parser.parse_args()

    # Generating data
    seq_param = json.load(args.seq)
    args.seq.close()

    seq,data = gen_sequence( seq_param, args.length, args.seqSeed)
    print "SEQ=",seq
    print "DATA=",data

# *************************************************************** gen_sequence
def gen_sequence(seq_param, length=10, seed=None):
    """
    Génère une séquence de donnée de taille 'length' en utilisant les 'rules'
    de 'seq_param'. Une séquence initiale aléatoire est générée d'abord dont la
    taille dépend de la règle.
    
    Params:
    - `seq_param`: dictionnaire qui détaille les règles (voir "seq.json")
    - `length` : longueur de la séquence
    - `seed`: seed pour random
    Return:
    - `seq` : la sequence de lettres générée.
    - `data` : la séquence encodée. Un vecteur par colonne.
    """
    np.random.seed(seed)
    seq = []
    # Initialisation aléatoire pendant 'init' pas de temps
    for i in xrange(max([seq_param['rules'][x]['delay'] for x in seq_param['rules']])+1):
        seq.append(chr(65+np.random.randint(0, len(seq_param['encode']))))
    # Génération de la séquence
    while len(seq) < length+1:
        seq.append(seq_param['rules'][seq[-1]][seq[-1-seq_param['rules'][seq[-1]]['delay']]])

    # encodage de la séquence en fonction (suite de vecteur de valeurs)
    data = np.zeros((len(seq_param['encode']), len(seq)))
    for i in xrange(len(seq)):
        data[:, i] = np.array(seq_param['encode'][seq[i]])

    return seq, data
# ******************************************************************* get_rule
def get_rule(seq, pos, rules):
    """
    Trouve la règle qui a été utilisée pour produire l'élément seq[pos+1].
    
    Params:
    - `seq`: séquence de lettres
    - `pos`: position courante
    - `rules`: ensemble des règles
    
    Return:
    - `rule` : la règles utilisée
    - `next` : le prochain élement
    """
    s = seq[pos]
    delay = rules[s]['delay']
    prev = seq[pos-delay]
    next = rules[s][prev]
    rule_str = prev+"["+str(delay)+"]"+s+":"+next
    return rule_str,next
# ************************************************************* get_delay_init
def get_delay_init(rules):
    """
    Calcule a partir de quelle indice les règles ont commencé à être
    appliquées.
    
    Params:
    - `rules`: règles utilisées
    Returns:
    - indice premier élément généré
    """
    return max([rules[x]['delay'] for x in rules])+1
# ****************************************************************************
# *********************************************************************** MAIN
# ****************************************************************************
if __name__ == "__main__":
    main()


