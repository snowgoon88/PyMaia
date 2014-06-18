#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, sys

import numpy as np
import matplotlib
matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt

# ************************************************************************ MAP
class MAP(object):
    """ Classe générique pour NN de type SOM.
    """
    
    # ---------------------------------------------------------------- init
    def __init__(self, name="MAP"):
        """ NN de type SOM
        
        :Param
        - `name`: nom du réseau
        """
        self._name = name
    # -----------------------------------------------------------  generate
    def generate(self, dim_input, size_grid=2, seed=None, rand=True):
        """
        - dim_input : nb de dimension en input
        - size_grid : network will have size_grid x size_grid neurones
        
        """
        # Les poids uniformes entre -1 et 1
        if rand:
            np.random.seed( seed )
            self._w = np.random.rand( size_grid*size_grid, dim_input) * 2.0 - 1.0
        else:
            self._w = np.zeros( (size_grid*size_grid, dim_input) )
        self._dim = size_grid
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
    # ---------------------------------------------------------------- copy
    def copy_from(self, other):
        """
        Génère une MAP avec les même poids que 'other'
        """
        self._w = other._w.copy()
        self._dim = other._dim
        self._d_topo = other._d_topo.copy()
    # -------------------------------------------------------------- output
    def output(self, data):
        """
        - data : same dimension as dim_input
        Return:
         - idx_win : index of winning neurone
         - dist : vector of distance to neurones
        """
        # Calcule w_i - data, ou w_i et un vecteur ligne
        diff = self._w - data
        # Puis une distance par ligne => vecteur de distance
        dist = np.sum( np.square(diff), 1)
        # Neurone vainqueur
        idx_win = np.argmin(dist)
        #
        return idx_win, dist
    # ------------------------------------------------------------ str_dump
    def str_dump(self, ):
        """ Dump everything
        """
        str_dump = self._name+" dim_input={0}, with {1} neurones\n".format( self._w.shape[1], self._w.shape[0] )
        str_dump += "_w = " + str(self._w)
        str_dump += "\n_d_topo = " + str(self._d_topo)
        return str_dump
    # --------------------------------------------------------------- draw_net
    def draw_net(self, ax, col='k'):
        """
        Affiche les points en 2D avec les liens entre les points
        Param:
        - ax : matplotlib.axes.Axes
        """
        # Les points
        # ax.plot( self._w[:,0], self._w[:,1], 'ob')
        # for i in range(self._dim * self._dim):
        #     ax.text( self._w[i,0], self._w[i,1], str(i), color=col_p)
        ax.scatter( self._w[:,0], self._w[:,1], s=50, c= 'w', edgecolors=col, zorder=10)
        # Les lignes
        # parcoure une dimension, affiche 2 lignes : "hori" et "vert"
        for i in xrange(self._dim):
            # "horizontal" line
            id = i*self._dim
            l_id = [id+d for d in xrange(self._dim)]
            # print l_id
            # ax.plot( self._w[id:id+self._dim,0], self._w[id:id+self._dim,1], '-b')
            ax.plot( self._w[l_id,0], self._w[l_id,1], '-'+col) #'-b')
            # "vertical" line
            id = i
            l_id = [id+self._dim*d for d in xrange(self._dim)]
            # print l_id
            # ax.plot( self._w[id:id+self._dim,0], self._w[id:id+self._dim,1], '-b')
            ax.plot( self._w[l_id,0], self._w[l_id,1], '-'+col)#'-b')
    # ------------------------------------------------------------- draw_delta
    def draw_delta(self, ax, delta):
        """
        Affiche des flèches entre les points et leur future nouvelle position.
        Param:
        - ax : matplotlib.axes.Axes
        - delta : mat nx2 de delta des poids (retorn of self.train)
        """
        print "w=",self._w
        print "delta=",delta
        ax.quiver( self._w[:,0], self._w[:,1], delta[:,0], delta[:,1], color='r')#, angles='xy', scale_units='xy', scale=1, color='r')
    # ------------------------------------------------------------------ learn
    def learn(self, samples, nb_epoch=25000, epsilon=0.1, eta=1.0,
              freq_dist=100, freq_draw=-1, fig=None, ax1=None, ax2=None, col_net='k'):
        """
        Fait nb_epoch d'apprentissage avec les samples tirés dans le désordre.

        :Params:
        - `samples`:
        - `nb_epoch`:
        """
        ite = 1
        distortion = []
        dist = 0
        for s in samples:
            # print "diff=",self._w - s
            D = ((self._w - s)**2).sum(axis=-1)
            # print "D=",D
            # print "D.min()=".D.min()
            dist += D.min()
        dist /= float(samples.shape[0])
        distortion.append( dist )
        print "{0:5} : {1:5.4}".format(ite, dist)
        #
        # Figure if any
        if (ax1 or ax2):
            ax2.cla()
            ax2.plot(np.arange(len(distortion))*freq_dist, distortion, '.-', alpha=0.8, color="gray", markerfacecolor="red")
            ax2.set_title('Distortion')
            ax1.cla()
            ax1.set_title('samples and net')
            ax1.set_xlim(-1,1)
            ax1.set_ylim(-1,1)
            ax1.scatter( samples[:,0], samples[:,1], s=3.0, color='b')
            self.draw_net(ax1, col_net)
            fig.canvas.draw()
        #fig.show()
        #
        for epoch in range(nb_epoch):
            list_id = np.arange(samples.shape[0])
            np.random.shuffle( list_id )
            for id in list_id:
                self.learn_data( samples[id,:], epsilon, eta)
                # compute distorsion every 100 iterations
                if ite % freq_dist == 0:
                    dist = 0
                    for s in samples:
                        # print "diff=",self._w - s
                        D = ((self._w - s)**2).sum(axis=-1)
                        # print "D=",D
                        dist += D.min()
                    dist /= float(samples.shape[0])
                    distortion.append( dist )
                    print "{0:5} : {1:5.4}".format(ite, dist)
                # Plot
                if freq_draw > 0 and ite % freq_draw == 0:
                    ax2.cla()
                    ax2.set_title('Distortion')
                    ax2.plot(np.arange(len(distortion))*freq_dist, distortion, '.-', alpha=0.8, color="gray", markerfacecolor="red")
                    ax1.cla()
                    ax1.set_title('samples and net')
                    ax1.set_xlim(-1,1)
                    ax1.set_ylim(-1,1)
                    ax1.scatter( samples[:,0], samples[:,1], s=3.0, color='b')
                    self.draw_net(ax1, col_net)
                    fig.canvas.draw()
                    #fig.show()
                ite +=1
        return distortion

# *********************************************************************** DSOM
class DSOM(MAP):
    # ------------------------------------------------------------- dist_neigh
    def dist_neigh(self, i, s, v, eta=1.0):
        """ Distance h(i,s,v) = exp( -||p_i - p_s||^2 / (eta^2 ||v-w_i||)
        
        Params:
        - `i`: neuron i
        - `s`: neuron winner
        - 'v': valueof current input or data
        - `eta` : elasticity 
        """
        dtopo = self._d_topo[i,s]
        dweight = np.sum( np.square( v - self._w[s,:] ))
        h = np.exp( - dtopo / (eta*eta*dweight) )
        return h
    # ---------------------------------------------------------- learn_data
    def learn_data(self, data, epsilon=0.1, eta=1.0):
        # Compute winner node
        win,dist = self.output( data )
        delta = []
        for i in range(len(self._w)):
            delta_i = epsilon * self.dist_neigh(i,win,data,eta) * np.subtract(data,self._w[i,:])
            delta.append( delta_i )
            # print "id={0} n={1:4.3} delta_i={2}".format( i, self.dist_neigh(i,win,data), str(delta_i))
            self._w[i,:] = np.add( self._w[i,:], delta_i)
        # print self._name+" learn with {0} winner is {1} at d={2}".format(data, win,dist)
        # print "delta=",delta
        return np.array(delta)
# ****************************************************************** DSOM_NICO
class DSOM_NICO(MAP):
    # ------------------------------------------------------------- dist_neigh
    def dist_neigh(self, i, s, v, eta=1.0):
        """ Distance h(i,s,v) = exp( -||p_i - p_s||^2 / (eta^2 ||v-w_i||)
        
        Params:
        - `i`: neuron i
        - `s`: neuron winner
        - 'v': valueof current input or data
        - `eta` : elasticity 
        """
        dtopo = self._d_topo[i,s]
        dweight = np.sum( np.square( v - self._w[s,:] ))
        h = np.exp( - dtopo / (eta*eta*dweight) )
        return h
    # ---------------------------------------------------------- learn_data
    def learn_data(self,  data, epsilon=0.1, eta=1.0):
        """
        Met en place les équations de l'article, avec exactitude.
        """
        win,dist = self.output( data )
        delta = []
        for i in range(len(self._w)):
            delta_i = epsilon * norm( np.subtract(data,self._w[i,:])) * self.dist_neigh(i,win,data,eta) * np.subtract(data,self._w[i,:])
            delta.append( delta_i )
            # print "id={0} n={1:4.3} delta_i={2}".format( i, self.dist_neigh(i,win,data), str(delta_i))
            self._w[i,:] = np.add( self._w[i,:], delta_i)
        # print self._name+" learn with {0} winner is {1} at d={2}".format(data, win,dist)
        # print "delta=",delta
        return np.array(delta)
# ************************************************************************ SOM
class SOM(MAP):      
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
    # ------------------------------------------------------------- learn_data
    def learn_data(self, data, epsilon=0.1, sigma=1.0):
        """ Apprend 1 sample avec loi de type Kohonen.
        
        Params:
        - `data`: un échantillon 
        - `epsilon`: coef d'apprentissage
        - `sigma`: taille de la gaussienne
        """
        # Compute winner node
        win,dist = self.output( data )
        delta = []
        for i in range(len(self._w)):
            delta_i = epsilon * self.dist_neigh(i,win,sigma) * np.subtract(data, self._w[i,:])
            delta.append( delta_i )
            self._w[i,:] = np.add( self._w[i,:], delta_i)
        # print self._name+" learn with {0} winner is {1} at d={2}".format(data, win,dist)
        # print "delta=",delta
        return np.array(delta)
# ****************************************************************************
def dist_ring(nb_point, center=(0,0), r_min=0, r_max=1):
    """
    Génére des points 2D selon une distribution uniforme sur un anneau.
    """
    res = np.zeros((nb_point,2))
    xc,yc = center
    for i in range(nb_point):
        r = -1
        while r < r_min or r > r_max:
            x,y = np.random.random()*2*r_max-r_max, np.random.random()*2*r_max-r_max
            r = np.sqrt((x)*(x) + (y)*(y))
        res[i,:] = x+xc,y+yc
    return res
def norm( vec ):
    """
    Calcule la norme d'un vecteur
    """
    n = np.linalg.norm( vec, 2)
    return n
# ********************************************************************** LEARN
def epoch_learn(samples, l_map=None, nb_epoch=25000, l_epsilon=[], l_eta=[],
                freq_dist=100, freq_draw=-1, l_col=[]):
        """
        Fait nb_epoch d'apprentissage avec les samples tirés dans le désordre pour la liste
        de MAP donnée.

        :Params:
        - `samples`: les échantillons
        - `l_map` : un liste de map à apprendre
        - `nb_epoch`: 
        """
        # Figure if any
        fig = None
        l_ax= []
        if freq_draw >= 0:
            fig = plt.figure( figsize=(16,10) )
            nbcol_plot = max(2, len(l_map))
            ax_dglobal = plt.subplot2grid( (5, nbcol_plot), (0,0), colspan=nbcol_plot)
            ax_dlocal = plt.subplot2grid( (5, nbcol_plot), (1,0), colspan=nbcol_plot)
            for m in range(len(l_map)):
                l_ax.append( plt.subplot2grid((5, nbcol_plot),(2,m), rowspan=3) )
            # plt.tight_layout()
            plt.subplots_adjust( left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.2, hspace=0.2)

            plt.ion()
            plt.show(False) # Not blocking.
        #
        ite = 1
        distortion = []
        for map in l_map:
            dist = 0
            for s in samples:
                # print "diff=",self._w - s
                D = ((map._w - s)**2).sum(axis=-1)
                # print "D=",D
                # print "D.min()=".D.min()
                dist += D.min()
            dist /= float(samples.shape[0])
            distortion.append( dist )
        print "{0:5} : {1}".format(ite, distortion)
        l_distortion = np.array( distortion, ndmin=2)
        # print "l_dist=",l_distortion
        #
        # Figure if any
        if (fig):
            # global distortion plot
            ax_dglobal.cla()
            # ax_dglobal.set_title('Global Distortion')
            # local distortion plot
            ax_dlocal.cla()
            # ax_dlocal.set_title('Last Distortion')
            for m in xrange(l_distortion.shape[1]):                
                ax_dglobal.plot(np.arange(l_distortion.shape[0])*freq_dist, l_distortion[:,m], color=l_col[m])
                ax_dlocal.plot(l_distortion[-10:,m], color=l_col[m])
            ax_dlocal.legend( [m._name for m in l_map], loc='upper left', fontsize='x-small')
            for m in range(len(l_map)):
                map = l_map[m]
                # scatter plot
                ax1 = l_ax[m]
                ax1.cla()
                # ax1.set_title('samples and net')
                ax1.set_xlim(-1,1)
                ax1.set_ylim(-1,1)
                ax1.scatter( samples[:,0], samples[:,1], s=3.0, color='b', alpha=0.25)
                map.draw_net(ax1, col=l_col[m])
            fig.canvas.draw()
            filename = "f_{0:4}.png".format(l_distortion.shape[0])
            fig.savefig(filename)
            #fig.show()
        #
        for epoch in range(nb_epoch):
            list_id = np.arange(samples.shape[0])
            np.random.shuffle( list_id )
            for id in list_id:
                for m in xrange(len(l_map)):
                    map = l_map[m]
                    map.learn_data( samples[id,:], l_epsilon[m], l_eta[m])
                # compute distorsion every 100 iterations
                if ite % freq_dist == 0:
                    distortion = []
                    for mapl in l_map:
                        dist = 0
                        for s in samples:
                            # print "diff=",self._w - s
                            D = ((mapl._w - s)**2).sum(axis=-1)
                            # print "D=",D
                            # print "D.min()=".D.min()
                            dist += D.min()
                        dist /= float(samples.shape[0])
                        distortion.append( dist )
                    print "{0:05} : {1}".format(ite, distortion)
                    l_distortion = np.row_stack((l_distortion,np.array(distortion,ndmin=2)))
                    # print "l_dist=",l_distortion

                # Plot
                if fig and ite % freq_draw == 0:
                    # global distortion plot
                    ax_dglobal.cla()
                    # ax_dglobal.set_title('Global Distortion')
                    # local distortion plot
                    ax_dlocal.cla()
                    # ax_dlocal.set_title('Last Distortion')
                    for m in xrange(l_distortion.shape[1]):                
                        ax_dglobal.plot(np.arange(l_distortion.shape[0])*freq_dist, l_distortion[:,m], color=l_col[m])
                        ax_dlocal.plot(l_distortion[-10:,m], color=l_col[m])
                    ax_dlocal.legend( [m._name for m in l_map], loc='upper left',fontsize='x-small')
                    for m in range(len(l_map)):
                        map = l_map[m]
                        # scatter plot
                        ax1 = l_ax[m]
                        ax1.cla()
                        # ax1.set_title('samples and net')
                        ax1.set_xlim(-1,1)
                        ax1.set_ylim(-1,1)
                        ax1.scatter( samples[:,0], samples[:,1], s=3.0, color='b', alpha=0.25)
                        map.draw_net(ax1, col=l_col[m])
                    fig.canvas.draw()
                    filename = "f_{0:05}.png".format(l_distortion.shape[0])
                    fig.savefig(filename)
                    #fig.show()
                ite +=1
        if fig:
            print 'Making movie animation.mpg - this make take a while'
            os.system("mencoder 'mf://f_*.png' -mf type=png:fps=10 -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o animation.mpg")
            plt.ioff()  # Set interactive mode ON, so matplotlib will not be blocking the window
            plt.show(True)
        return l_distortion

# ****************************************************************************
def test_creation():
    """
    Crée une carte DSOM. Et calcule la distance de chaque neurone a une 
    entrée 'x'.
    """
    dsom = DSOM()
    dsom.generate(3, 2)
    print dsom.str_dump()
    #
    x = np.array( [0.5, 0.5, 0.5] )
    win,dist = dsom.output( x )
    print "win is {0}, with w={1}".format(win, str(dsom._w[win]) )
    print "dist={0}".format(str(dist))
def test_distance():
    """
    Crée une carte DSOM dim3 -> 4x4 neurones.
    Calcule les activations des neurones pour une entrée donnée.
    Calcule la distance 'eta' en fonction du winner, de l'entrée et de la dist_topo.
    
    """
    dsom = DSOM()
    dsom.generate(3, 4)
    #
    data = [0.1, 0.2, 0.3]
    win,dist = dsom.output( data )
    print "With d={0}, win={1}".format( dist, win)
    for i in xrange(len(dsom._w)):
        print "i={0} win={1} eta={2}".format( i, win, dsom.dist_neigh(i,win,data))
    dsom.train( data )
    dsom.train( data, eta=2.0)
def test_plot():
    """
    Crée DSOM dim3 -> 3x3 neurones.
    Affiche;
    """
    dsom = DSOM()
    dsom.generate(2, 3)
    data = [0.7, 0.2]
    #
    # prépare la figure
    fig = plt.figure()
    #ax = fig.add_subplot(111)
    plt.xlim( -1, 1 )
    plt.ylim( -1, 1 )
    plt.plot( data[0], data[1], '+r')
    dsom.draw_net(plt, 'b', '-b')
    delta = dsom.train( data )
    print "delta=",delta
    print "delta0=",delta[:,0]
    print "delta1=",delta[:,1]
    print "w=",dsom._w
    dsom.draw_delta(plt, delta) # Attention, du coup ca dessine au nouvel emplacement des neurones
    dsom.draw_net(plt, 'g', '-g')
    # affiche la figure
    plt.show()
def test_dist():
    """
    Affiche une distribution uniforme sur un anneau.
    """
    samples = dist_ring( 1000 )
    fig = plt.figure()
    plt.scatter( samples[:,0], samples[:,1], s=1.0, color='b')
    plt.show()
def test_learn():
    """ 
    Apprentissage avec 2 samples et qq epochs
    """
    dsom = DSOM("DSOM")
    dsom.generate(2, 8, rand=False)
    som = SOM("SOM")
    som.copy_from(dsom)
    #samples = np.array([[0.1, 0.2, 0.3], [-0.2, 0.0, 0.8]])
    samples = dist_ring( 1000 )
    #
    epoch_learn( samples, l_map=[dsom,som], nb_epoch=4, l_epsilon=[0.1,0.1], l_eta=[1.0,0.1], freq_draw=100, l_col=['k','r'])
    # # Figure if any
    # fig = plt.figure()
    # ax1 = plt.subplot2grid((4, 2),(1,0), rowspan=3)
    # ax2 = plt.subplot2grid((4, 2),(0,0))
    # ax3 = plt.subplot2grid((4, 2),(1,1), rowspan=3)
    # ax4 = plt.subplot2grid((4, 2),(0,1))
    # plt.ion()
    # plt.show(False) # Not blocking.
    # ## dist_dsom = dsom.learn(samples, nb_epoch=4, epsilon=0.1, eta=1.0, freq_draw=100, fig=fig, ax1=ax1, ax2=ax2)
    # dist_som = som.learn(samples, nb_epoch=4, epsilon=0.1, eta=0.1, freq_draw=100, fig=fig, ax1=ax3, ax2=ax4, col_net='r')
    # plt.ioff()  # Set interactive mode ON, so matplotlib will not be blocking the window
    # plt.show(True) 
    # print "dist=",distortion
def test_learn2():
    """
    Apprentissage avec samples de deux densité sur deux disques.
    """
    # Ratio de surface pour ratio de nb de points
    area_1 = np.pi*0.5**2 - np.pi*0.25**2
    area_2 = np.pi*0.25**2

    n1 = int(area_1*4000)
    n2 = int(area_2*1000)
    samples = np.zeros((n1+n2,2))
    samples[:n1] = dist_ring(nb_point=n1, r_min=0.25, r_max=0.50)
    samples[n1:] = dist_ring(nb_point=n2, r_min=0.00, r_max=0.25)
    dsom = DSOM("DSOM")
    dsom.generate(2, 10, rand=False)
    som1 = SOM("SOM_0.1")
    som1.copy_from(dsom)
    som2 = SOM("SOM_0.05")
    som2.copy_from(dsom)
    #samples = np.array([[0.1, 0.2, 0.3], [-0.2, 0.0, 0.8]])
    # distortion = dsom.learn(samples, nb_epoch=10, epsilon=0.1, eta=1.0) 
    epoch_learn( samples, l_map=[dsom,som1,som2], nb_epoch=4, l_epsilon=[0.1,0.1,0.1], l_eta=[1.0,0.1,0.05], freq_draw=100, l_col=['k','r','g'])
    # print "dist=",distortion
    print "DSOM"
    print dsom._w
    print "SOM"
    print som._w
def test_learn3():
    """
    Apprentissage avec samples de deux densité sur deux disques.
    """
    # Ratio de surface pour ratio de nb de points
    area_1 = np.pi*0.5**2 - np.pi*0.25**2
    area_2 = np.pi*0.25**2

    n1 = int(area_1*4000)
    n2 = int(area_2*1000)
    samples = np.zeros((n1+n2,2))
    samples[:n1] = dist_ring(nb_point=n1, r_min=0.25, r_max=0.50)
    samples[n1:] = dist_ring(nb_point=n2, r_min=0.00, r_max=0.25)
    dsom = DSOM("DSOM_1.5")
    dsom.generate(2, 10, rand=False)
    dsom_nic = DSOM_NICO("DSOM_NIC1.4")
    dsom_nic.copy_from(dsom)
    dsom_nic2 = DSOM_NICO("DSOM_NIC1.2")
    dsom_nic2.copy_from(dsom)
    som1 = SOM("SOM_0.05")
    som1.copy_from(dsom)
    som2 = SOM("SOM_0.1")
    som2.copy_from(dsom)
    #samples = np.array([[0.1, 0.2, 0.3], [-0.2, 0.0, 0.8]])
    # distortion = dsom.learn(samples, nb_epoch=10, epsilon=0.1, eta=1.0) 
    epoch_learn( samples, l_map=[dsom,dsom_nic,som2], nb_epoch=10, l_epsilon=[0.1,0.1,0.1], l_eta=[1.5,1.4,0.1], freq_draw=100, l_col=['k','r','g'])
def test_learn4():
    """
    Apprentissage avec samples de deux densité sur deux disques.
    """
    # Ratio de surface pour ratio de nb de points
    area_1 = np.pi*0.5**2 - np.pi*0.25**2
    area_2 = np.pi*0.25**2

    n1 = int(area_1*4000)
    n2 = int(area_2*1000)
    samples = np.zeros((n1+n2,2))
    samples[:n1] = dist_ring(nb_point=n1, r_min=0.25, r_max=0.50)
    samples[n1:] = dist_ring(nb_point=n2, r_min=0.00, r_max=0.25)
    dsom = DSOM("DSOM_1.0")
    dsom.generate(2, 10, rand=False)
    dsom1 = DSOM("DSOM_1.5")
    dsom1.copy_from(dsom)
    dsom2 = DSOM("DSOM_NIC0.5")
    dsom2.copy_from(dsom)
    #samples = np.array([[0.1, 0.2, 0.3], [-0.2, 0.0, 0.8]])
    # distortion = dsom.learn(samples, nb_epoch=10, epsilon=0.1, eta=1.0) 
    epoch_learn( samples, l_map=[dsom,dsom1,dsom2], nb_epoch=5, l_epsilon=[0.1,0.1,0.1], l_eta=[1.0,1.4,0.1], freq_draw=100, l_col=['k','r','g'])
def test_copy():
    dsom = DSOM("DSOM")
    dsom.generate(2, 3, rand=False)
    dsom_nic1 = DSOM_NICO("DSOM_NIC")
    dsom_nic1.copy_from(dsom)
    dsom_nic2 = DSOM_NICO("DSOM_NIC2")
    dsom_nic2.copy_from(dsom)
    #
    print "DSOM"
    print dsom.str_dump()
    print "DSOM_NIC"
    print dsom_nic.str_dump()
    samples = np.array([[0.1, 0.2,]])
    dist = epoch_learn( samples, l_map=[dsom,dsom_nic,dsom_nic2], nb_epoch=1, l_epsilon=[0.1,0.1,0.1], l_eta=[1.0,1.0,0.5])
    print "DIST=",dist
    print "---------------------------"
    print "DSOM"
    print dsom._w
    print "DSOM_NIC"
    print dsom_nic._w
# ****************************************************************************
if __name__ == '__main__':
    # test_creation()
    # test_distance()
    # test_plot()
    # test_dist()
    # test_learn()
    # test_learn2()
    test_learn3()
    # test_copy()
    # Dans le graphe, faire toutes les distorsions sur un seul graphe mais en ajouter
    # un second avec une fenêtre glissante                        ______________@todo
