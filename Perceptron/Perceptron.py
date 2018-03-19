#!/usr/bin/env python

__author__ = "Daniel del Castillo"
__version__ = "0.0.0"

"""
This is a father class for the family of Perceptron architectures.
"""

class Perceptron:
    def __init__(self,name,size_in,size_out,W):
        self.name = name
        self.size_in = size_in
        self.size_out = size_out
        self.W = W

    def train(self):
        pass

    def predict(self):
        pass
