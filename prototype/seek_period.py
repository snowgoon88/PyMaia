#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

# ************************************************************* SEARCH PATTERN
def dphi(a, b, phi_a=0, phi_b=0, verb=False):
    """
    Distance entre a avec un décalage de phi_a et 
    b avec un décalage de phi_b.
    phi_a et phi_b sont des entiers
    """
    d = 0
    for i in xrange(len(a)):
        if a[(i+phi_a) % len(a)] != b[(i+phi_b) % len(a)]:
            d += 1
    if verb:
        sdebug = ""
        for i in xrange(len(a)):
            sdebug += a[(i+phi_a) % len(a)]
        sdebug += "-"
        for i in xrange(len(a)):
            sdebug += b[(i+phi_b) % len(a)]
        print "d({0})={1}".format(sdebug,d)
    return d

def test_phi(seq, phi=0):
    for i in xrange(len(seq)):
        print seq[(i+phi) % len(seq)]

def iter_align(seq, length_p):
    """
    En posant phi[0] = k[0] = 0 on recherche les autres 'k'
    de manière itérative :
    Les 'm' premiers phi étant fixé, cherche le 'm+1' phi.

    Params:
    - `length_p' : longueur de la période présummée
    """
    k = [0]
    # Pour chaque m, cherche le décalage phi qui minimise la distance
    # des j premieres subsequence à la m-eme subséquence.
    for m in xrange(1,len(seq)/length_p):
        dphim = []
        for phi in xrange(length_p):
            # print "j in ",str(range(0,m))
            sumdist = [dphi(seq[j*length_p:(j+1)*length_p], seq[m*length_p:(m+1)*length_p], k[j], phi, False) for j in xrange(0,m)]
            # print "dist pour m={0} et phi={1} : {2}".format(m,phi,sumdist)
            dphim.append(np.sum(sumdist))
        k.append( np.argmin(dphim) )
        # print "k=",k
    return k

def consensus_pattern(seq, phi, length_p, alphabet):
    """
    On recherche, lettre par lettre, le pattern consensus.
    """
    # plus simple de chercher vs un sequence "cible" virtuelle
    seq_target = []
    for i in xrange(len(seq)/length_p):
        for j in xrange(length_p):
            seq_target.append( seq[i*length_p+(j+phi[i])%length_p] )
    # print "target=",seq_target
    # Le pattern le plus consensuel, lettre par lettre.
    pat = []
    for i in range(length_p):
        l_dist = []
        for lettre in alphabet:
            distlettre = [(lettre == seq_target[i+j*length_p]) for j in xrange(len(seq)/length_p)]
            # print "pos={0}, {1} dist={2} {3}".format(i,lettre,length_p-np.sum(distlettre),distlettre)
            l_dist.append( np.sum(distlettre) )
        pat.append( alphabet[np.argmax(l_dist)] )
    #
    dtot = 0
    for i in range(len(seq)/length_p):
        dtot += dphi( seq[i*length_p:(i+1)*length_p], pat, phi[i], 0)
    #
    return pat, dtot

def search_period(seq, alphabet, p_max=None):
    """
    Pour chaque taille, recherche quelle est le meilleur pattern 
    et calcule la distance normé (nb de différence / nb répétition du pattern)

    Params:
    - `p_max`: taille max d'une période ou len(seq)/2
    """
    if p_max is None:
        p_max = len(seq)/2
    #
    result = []
    for i in xrange(1,p_max):
        seq_search = seq[0:i*(len(seq)/i)]
        k = iter_align(seq_search, i)
        pat,dpat = consensus_pattern( seq_search, k, i, alphabet)
        result.append( (pat,dpat/(float)(len(seq)/i)) )
        # print "{0} -> {1} {2} ({3})".format(i,pat,dpat,k)
    return result
    
# *********************************************************************** MAIN
# ****************************************************************************
# ****************************************************************************
if __name__ == '__main__':
    alphabet = ['A','B','C']

    s = ['A','B','C']
    s1 = ['B','C','A']
    S2 = ['A','B','A']
    seq = ['A','B','C','A','B','A','B','C','A']

    # --------
    res = search_period(seq)
    print res
    # --------
    # k = iter_align(seq, 3)
    # print "k=",k
    # pat = consensus_pattern( seq, k, 3)
    # print "pat=",pat
    # --------
    # def f_dphi(s,a,b,phi_a,phi_b):
    #     return s.format(phi_a,a,phi_b,b,dphi(a,b,phi_a,phi_b))
    # print f_dphi("d({0}-{1},{2}-{3})={4}",s,s1,0,0)
    # print f_dphi("d({0}-{1},{2}-{3})={4}",s,s1,0,1)
    # print f_dphi("d({0}-{1},{2}-{3})={4}",s,s1,0,2)
    # print f_dphi("d({0}-{1},{2}-{3})={4}",s,s1,1,0)
    # print f_dphi("d({0}-{1},{2}-{3})={4}",s,s1,2,0)
    # --------
    # test_phi( s, 0)
    # test_phi( s, 1)
    # test_phi( s, 2)
    # test_phi( s, 3)
# **************************************************************************
# ****************************************************************************

    
