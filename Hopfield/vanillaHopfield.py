#!/usr/bin/env python

__author__ = "Daniel del Castillo"
__version__ = "0.0.0"

"""
This is a plain vanilla implementation of a Hopfield network.
"""

import numpy as np
from pylab import imshow, cm, show

class Hopfield:
    def __init__(self, patterns=[]):
        self.W = None
        if len(patterns) > 0: self.train(patterns)

    # build the weight matrix
    def train(self, patterns):
        n_rows, n_cols = patterns.shape
        self.W = np.zeros((n_cols,n_cols))
        for p in patterns:
            self.W += np.outer(p,p)
        self.W[range(n_cols),range(n_cols)] = 0

    # recall a stored pattern (synchronous update)
    def recall(self, patterns, n_steps=10):
        sign_vec = np.vectorize(lambda x: -1 if x < 0 else 1)
        for _ in range(n_steps):
            patterns = sign_vec(np.dot(patterns,self.W))
        return patterns

    # display a pattern as a binary image of a specified size
    def display(self, pattern, size=32):
        imshow(pattern.reshape((size,size)),cmap=cm.binary)
        show()

    # calculate Hopfield energy for a pattern
    def energy(self, patterns):
        return np.array([-0.5*np.dot(np.dot(p.T,W),p) for p in patterns])

    # read a comma-separated file containing a binary image
    def load_pict(self, n_patterns, dim_patterns, filename):
        patterns = np.loadtxt(open(filename, 'r'), delimiter=',', dtype=int)
        return patterns.reshape(n_patterns, dim_patterns)

    # flip a number of bits of a pattern
    def flip(self, pattern, n_flips):
        pat = np.copy(pattern)
        idxs = np.random.choice(range(len(pat)), n_flips, replace=False)
        for idx in idxs:
            pat[idx] *= -1
        return pat
