#!/usr/bin/python
# -*- coding: utf-8 -*-

import random as rd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons


# ************************************************************* DiscreteSlider
class DiscreteSlider(Slider):
    """A matplotlib slider widget with discrete steps."""
    def __init__(self, *args, **kwargs):
        """Identical to Slider.__init__, except for the "increment" kwarg.
        "increment" specifies the step size that the slider will be discritized
        to."""
        self.inc = kwargs.pop('increment', 0.5)
        Slider.__init__(self, *args, **kwargs)

    def set_val(self, val):
        discrete_val = int(val / self.inc) * self.inc
        # We can't just call Slider.set_val(self, discrete_val), because this 
        # will prevent the slider from updating properly (it will get stuck at
        # the first step and not "slide"). Instead, we'll keep track of the
        # the continuous value as self.val and pass in the discrete value to
        # everything else.
        xy = self.poly.xy
        xy[2] = discrete_val, 1
        xy[3] = discrete_val, 0
        self.poly.xy = xy
        self.valtext.set_text(self.valfmt % discrete_val)
        if self.drawon: 
            self.ax.figure.canvas.draw()
        self.val = val
        if not self.eventson: 
            return
        for cid, func in self.observers.iteritems():
            func(discrete_val)
# ****************************************************************************
# ******************************************************************** SeqPlot
class SeqPlot(object):
    """
    Plot une séquence comme une matrice de pt de couleurs,
    en commençant en bas à gauche.
    """
    # ---------------------------------------------------------------  init
    def __init__(self, seq, maxcol=200):
        """
        Prépare le plot initial, en maxcol de large.
        
        :Param
        - `seq`: sequence d'entier
        - `maxcol`: largeur max de la matrice de couleur
        """
        self._seq = seq
        self._maxcol = maxcol        
        self._offset = 0
        self._nbcol = maxcol
        self._size = len(self._seq)
        self._nbpoint = (self._size / self._nbcol)*self._nbcol + self._offset
        
        # Ajoute des 0 en fin de séquence pour être sûr
        self._seq.extend( [0 for i in xrange(maxcol)] )
        
        self._fig, self._ax = plt.subplots()
        # Les marges habituelles sont de 0.1 partout
        # Fait en sorte qu'il y a de la place
        plt.subplots_adjust(bottom=0.25)
        self._ax.pcolor( np.reshape(self._seq[self._offset:self._nbpoint], (-1,self._nbcol) ))

        axcolor = 'lightgoldenrodyellow'
        self._axoffset  = plt.axes([0.1, 0.15, 0.8, 0.03], axisbg=axcolor)
        self._axsize  = plt.axes([0.1, 0.1, 0.8, 0.03], axisbg=axcolor)

        # sfreq = Slider(axfreq, 'Freq', 0.1, 30.0, valinit=f0)
        self._soffset = DiscreteSlider(self._axoffset, 'Off', 0.0, self._maxcol, increment=1.0, valinit=self._offset)
        self._ssize = DiscreteSlider(self._axsize, 'Size', 1.0, self._maxcol, increment=1.0, valinit=self._maxcol)
        self._ssize.on_changed(self.update)
        self._soffset.on_changed(self.update)

    # -------------------------------------------------------------- update
    def update(self,val):
        self._offset = int(self._soffset.val)
        self._nbcol = int(self._ssize.val)
        # print "offset=",offset," nbcol=",nbcol
        self._nbpoint = (self._size / self._nbcol)*self._nbcol + self._offset
        # print "nbpoint=",nbpoint
        self._ax.cla()
        self._ax.pcolor( np.reshape(self._seq[self._offset:self._nbpoint], (-1,self._nbcol) ))
        #    fig.canvas.draw()
        self._fig.canvas.draw_idle()
        
    # ------------------------------------------------------------------- show
    def show(self, ):
        """
        Affiche le plot
        """
        plt.show()
# ****************************************************************************

# *********************************************************************** MAIN
if __name__ == '__main__':
    # data with some zeros at the end
    data = [rd.randint(0,5) for i in xrange(396)]
    data.extend( [0 for i in xrange(4)] )
    p = SeqPlot( data, maxcol=50)
    # print data
    p.show()
# ****************************************************************************


