#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import gen_sequence as gs
import plot_seq as ps

import matplotlib
import matplotlib.pyplot as plt

# ********************************************************************* RECMAP
class RECMAP(object):
    """
    """
    # ------------------------------------------------------------------- init
    def __init__(self, name="RECMAP"):
        """ NN de type (D)SOM recurrente.
        """
        self._name = name

    # --------------------------------------------------------------- generate
    def generate(self, dim_input, size_grid, seed=None, rand=True):
        """
        
        Params:
        - `dim_input`: dimension en input
        - `size_grid`: le résseau aura seize_grid x size_grid neurones
        - `seed`: pour la génération aléatoire des poids
        - `rand`: poids aléatoire ou 0
        """
         # Les poids d'entrée uniformes entre -1 et 1
        if rand:
            np.random.seed( seed )
            self._win = np.random.rand( size_grid*size_grid, dim_input) * 2.0 - 1.0
        else:
            # ou tous égaux à 0
            self._win = np.zeros( (size_grid*size_grid, dim_input) )
        self._dim = size_grid
        #
        # Poids du contexte : les même que ceux d'entrée
        self._wcon = self._win.copy()
        self._context = np.zeros( (dim_input,) )
        self._context = np.zeros( (dim_input,) )
        # 
        # Normalized Square of Manhattan Distance between neurones
        # => Sauf que c'est pas normalisé entre 0 et 1 !!!!!!!!!!! ______@todo
        self._d_topo = np.zeros((size_grid*size_grid,size_grid*size_grid))
        max_dist = np.square((size_grid-1)+(size_grid-1))
        for id1 in xrange(size_grid*size_grid):
            px1 = id1 % size_grid
            py1 = id1 / size_grid
            for id2 in xrange(size_grid*size_grid):
                px2 = id2 % size_grid
                py2 = id2 / size_grid
                dist = np.abs(px1-px2) + np.abs(py1-py2)
                # print "id1={0} ({1}, {2}) id2={3} ({4}, {5}) d={6}".format(id1,px1,py1,id2,px2,py2,
                #                                                            dist)
                self._d_topo[id1,id2] = (float)(dist * dist) / (float)(max_dist)

    # ----------------------------------------------------------------- output
    def output(self, data, alpha=0.5, beta=0.5):
        """
        1 passe forward du réseau
        
        Params:
        - `data`: échantillon, same dimensions as dim_input
        """
        # distance courante au poids : w_i - data, ou w_i et un vecteur ligne
        diff = self._win - data
        # Puis une distance par ligne => vecteur de distance
        dw = np.sum( np.square(diff), 1)
        # Puis le contexte, qui est fait le data précédent
        dc = np.sum( np.square( self._wcon -self._context ), 1)
        # Neurone vainqueur
        idx_win = np.argmin( alpha * dw + beta * dc )
        #
        self._old_context = self._context
        self._context = data
        #
        return idx_win, dw, dc
    # ---------------------------------------------------------- dist_neigh
    def dist_neigh(self, i, j, sigma=1.0):
        """ Distance h(t,i,j) = exp( -||p_i - p_j||^2 / (2*sigma^2)

        Params:
        - `i`: neuron i
        - `j`: neuron j
        - `sigma`: largeur de la gaussienne
        """
        dtopo = self._d_topo[i,j]
        h = np.exp( - dtopo / (2.0 * sigma * sigma) )
        return h
    # ------------------------------------------------------------------ learn
    def learn(self, samples, epsilon=0.1, eta=1.0 ):
        """
        Fait apprentissage avec le sample, en fonction du contexte et du voisinage.
        
        Params:
        - `samples`: donnée d'apprentissage, avec une entrée en ligne.
        - `epsilon`:
        - `eta`:
        
        Return:
        - `delta_in` : modif poids entrée de la carte
        - `delta_con` : modif des poids du contexte
        - `win` : indice du neurone vainqueur
        """
        data = np.array(samples)
        print "DATA="+str(data)
        # Quel est le node vainqueur ? : SOM mode.
        win,dw,dc = self.output( data )
        #
        delta_in = []
        delta_con = []
        for i in range(len(self._win)):
            delta_i = epsilon * self.dist_neigh(i,win,eta) * np.subtract(data,self._win[i,:])
            delta_in.append( delta_i )
            delta_c = epsilon * self.dist_neigh(i,win,eta) * np.subtract(self._old_context, self._wcon[i,:])
            delta_con.append( delta_c )
        #
        # modif des poids
        self._win = np.add( self._win, np.array(delta_in) )
        self._wcon = np.add( self._wcon, np.array(delta_con) )
        # for i in xrange(len(self._win)):
        #     self._win[i,:] = np.add( self._win[i,:], delta_in[i])
        #     self._wcon[i,:] = np.add( self._wcon[i,:], delta_con[i])
        return np.array(delta_in), np.array(delta_con), win
    # ------------------------------------------------------------ str_dump
    def str_dump(self, ):
        """ Dump everything
        """
        str_dump = self._name+" dim_input={0}, with {1} neurones\n".format( self._win.shape[1], self._win.shape[0] )
        str_dump += "_context = " + str(self._context)
        str_dump += "\n_win = " + str(self._win)
        str_dump += "\n_d_topo = " + str(self._d_topo)
        str_dump += "\n_wcon = " + str(self._wcon)
        return str_dump
# ****************************************************************************
# *********************************************************************** PLOT
# ****************************************************************************
def plot_sequence(seq, outfilename = None):
    """
    Affiche une séquence de lettre.
    
    Params:
    - `seq`:
    - `outfilename` : si pas None, sauvegarde dans un fichier.
    """
    # prépare la figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #
    code = {"A":1.0, "B":2.0, "C":3.0, "D":4.0, "E":5.0}
    seq_plot = [ code[s] for s in seq ]
    ax.plot(seq_plot, '-')
    #
    # et une autre figure qui plot avec des matrices de couleur
    p = ps.SeqPlot( seq_plot, maxcol=100)
    plt.show()
    #
    # save ??
    if outfilename is not None:
        fname = outfilename+str('%03d' % (length))+'.png'
        print fname
        plt.savefig( fname )
    

# ****************************************************************************
# *********************************************************************** TEST
# ****************************************************************************
def test_creation():
    """
    Creation d'une carte et test sur une séquence de quelques entrées.
    """
    rsom = RECMAP()
    rsom.generate( 3, 2)
    print "CREATION\n"+rsom.str_dump()
    #
    x0 = np.array( [0.5, 0.5, 0.5] )
    x1 = np.array( [0.1, 0.2, -0.5] )
    x2 = np.array( [0.9, 0.0, 0.5] )
    #
    win,dw,dc = rsom.output( x0 )
    print "OUTPUT"
    print "win = {0}".format( win )
    print "dw = {0}".format( str(dw) )
    print "dc = {0}".format( str(dc) )
    #
    win,dw,dc = rsom.output( x1 )
    print "OUTPUT"
    print "win = {0}".format( win )
    print "dw = {0}".format( str(dw) )
    print "dc = {0}".format( str(dc) )
    #
    win,dw,dc = rsom.output( x2 )
    print "OUTPUT"
    print "win = {0}".format( win )
    print "dw = {0}".format( str(dw) )
    print "dc = {0}".format( str(dc) )
def test_learn():
    """
    Test d'un apprentissage simple.
    """
    rsom = RECMAP()
    rsom.generate( 3, 2)
    print "CREATION\n"+rsom.str_dump()
    #
    x0 = np.array( [0.5, 0.5, 0.5] )
    x1 = np.array( [0.1, 0.2, -0.5] )
    x2 = np.array( [0.9, 0.0, 0.5] )
    #
    di,dc = rsom.learn( x0, epsilon=0.1, eta=1.0 )
    print "delta_in = "+str(di)
    print "delta_con = "+str(dc)
    print "LEARNED_n"+rsom.str_dump()
def test_sequence():
    """
    Test d'un apprentissage avec des séquences de lettres.
    """
    import gen_sequence as gs
    #
    print "***** CREATION"
    rsom = RECMAP()
    rsom.generate( dim_input=5, size_grid=5)
    #
    # Génération de séqence à partir de fichier "seq_test.json"
    file_param = open('seq_test.json', 'r')
    seq_param = gs.json.load(file_param)
    file_param.close()
    seq,data = gs.gen_sequence( seq_param, length=1000 )
    pos_start = gs.get_delay_init(seq_param['rules'])
    print "***** SEQ\n",seq
    #
    # Pour évaluer/comprendre l'apprentissage
    d_eval = {}
    # Initialisation
    for i in xrange(pos_start):
        rsom.learn(data[:,i].transpose(), epsilon=0.0, eta=1.0 )
    # Apprentissage
    for i in xrange(pos_start, data.shape[1]):
        di,dc,win = rsom.learn( data[:,i].transpose(), epsilon=0.1, eta=1.0 )
        # ajouter le résultat à d_eval
        # D'abord trouver la règle en cours dans la séquence
        rule,next = gs.get_rule(seq,i,seq_param['rules'])
        # Augmenter compteur
        if not d_eval.has_key( rule):
            d_eval[rule] = []
        d_eval[rule].append( win )
    #
    for k,v in d_eval.iteritems():
        print "{0} => {1}".format(k, str(v))
def test_pattern():
    """
    Génère une séquence et cherche les "meilleurs" patterns qui peuvent
    être répétitif.
    """
    import gen_sequence as gs
    import seek_period as sp
    #
    # Génération de séqence à partir de fichier "seq_test.json"
    file_param = open('seq_test.json', 'r')
    seq_param = gs.json.load(file_param)
    file_param.close()
    seq,data = gs.gen_sequence( seq_param, length=1000 )
    #
    res = sp.search_period(seq, ['A','B','C','D','E'],120)
    for i in xrange(len(res)):
        p,d = res[i]
        print "{0:0.3f} - {1:3d} {2}".format( d,i+1,''.join(p) )
    #
    # prépare la figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(len(res))+1,[d for p,d in res], '-')
    plt.show()

def test_plot():
    """
    Différentes façon d'afficher une séqence de lettres.
    """
    # Génération de séqence à partir de fichier "seq_test.json"
    file_param = open('seq_test.json', 'r')
    seq_param = gs.json.load(file_param)
    file_param.close()
    seq,data = gs.gen_sequence( seq_param, length=500 )
    pos_start = gs.get_delay_init(seq_param['rules'])
    #
    plot_sequence( seq )
    
# ****************************************************************************
# *********************************************************************** MAIN
# ****************************************************************************
if __name__ == '__main__':
    # test_creation()
    # test_learn()
    # test_sequence()
    test_pattern()
    # test_plot()









        
