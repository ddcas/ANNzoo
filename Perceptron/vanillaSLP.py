#!/usr/bin/env python

__author__ = "Daniel del Castillo"
__version__ = "0.0.0"

"""
This is a plain vanilla implementation of a single-layer perceptron.
An example of use is also included.
"""

import numpy as np
import random
import time

from Perceptron import Perceptron

# only needed for the examples
from Dataset import Dataset

class SLP(Perceptron):
    def __init__(self,name,size_in=None,size_out=None,W=None):
        Perceptron.__init__(self,name,size_in,size_out,W)

    def train(self,X,T,eta=0.001,nepochs=20,learn_mode=[0,0], \
              header=True,wait=0):

        # print header?
        if header:
            lm = ''
            if learn_mode[0]:
                lm += 'delta&'
                if learn_mode[1]: lm += 'sequential'
                else: lm += 'batch'
            else:
                lm += 'perceptron&'
                if learn_mode[1]: lm += 'sequential'
                else: lm += 'batch'
            print('Training:',self.name,'| lr',eta,'|',nepochs,'epochs','|',lm)

        # initialize dimensions
        N = X.shape[1]
        if not self.size_in: self.size_in = X.shape[0]-1
        if not self.size_out: self.size_out = T.shape[0]
        if not self.W: self.W = np.random.randn(self.size_out,self.size_in+1)*0.01
        # initialize epoch array
        epochs = [0]
        # initialize metrics
        mc_ratios = []
        # check initial hypothesis
        WX = self.W@X
        found = np.array_equal(np.sign(WX),T)
        # training loop
        while not found:
            # find misclassified points
            mc_points = np.sign(WX) == T
            mc_ratios.append(1-np.sum(mc_points)/N)
            # calculate weight increment
            if learn_mode[0]: # delta learning rule
                if learn_mode[1]: # sequential learning
                    mc_index = random.choice([idx[0] \
                        for idx,x in np.ndenumerate(mc_points.reshape(N)) if not x])
                    delta = -eta*(WX[:,mc_index]-T[:,mc_index])*X[:,mc_index]
                else: # batch learning
                    e = WX-T
                    delta = -eta*(e/np.linalg.norm(e))@X.T
            else: # perceptron learning rule
                if learn_mode[1]: # sequential learning
                    mc_index = random.choice([idx[0] \
                        for idx,x in np.ndenumerate(mc_points.reshape(N)) if not x])
                    delta = -eta*(np.sign(WX[:,mc_index])-T[:,mc_index])*X[:,mc_index]
                else: # batch learning
                    delta = -eta*(np.sign(WX)-T)@X.T

            # update weights
            self.W += delta
            WX = self.W@X
            found = np.array_equal(np.sign(WX),T)
            # finished?
            if epochs[-1]==(nepochs-1):
                break
            # no? then increase epoch count
            epochs.append(epochs[-1]+1)
        # update animation with final hypothesis
        if found:
            mc_ratios.append(0)

        return epochs, mc_ratios, found

# EXAMPLES
# 1) Single-Layer Perceptron for LS and NLS data
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
N = 200 # the number of example patterns, must be an even number
d = 2 # the dimension of the input patterns
z = 1 # the dimension of the output patterns
c = 2 # the number of categories
# define the classes and their values
categories  = np.linspace(-d+1,d-1,d)
labels = np.array([[categories[0]]*(N//2)+[categories[1]]*(N//2)])
# generate dataset
LSdata = Dataset(N,d,z,c,labels,sd=0.25,mean_centre=0,mean_sd=1)
NLSdata = Dataset(N,d,z,c,labels,sd=2,mean_centre=0,mean_sd=1)

# Training parameters
e_max = 20

# create the perceptron instances
slpLS = SLP('My slpLS')
start_time = time.clock()
slpLS_epochs, slpLS_mc_ratios, slpLS_found = slpLS.train(LSdata.patterns,LSdata.targets,
        nepochs=e_max,learn_mode=[0,0])
slpLS_elapsed_time = time.clock() - start_time
if slpLS_found:
    print('The SLP found a solution in ' + str(slpLS_epochs[-1]+1) + ' epochs!')
else:
    print('The SLP could not find a solution in ' + str(slpLS_epochs[-1]+1) + ' epochs')
    print('Final ratio of misclassified points >>>',slpLS_mc_ratios[-1])

print('Final weights: '+'\n'+'     W >>> '+str(slpLS.W))
print('Time elapsed: '+str(slpLS_elapsed_time))

slpNLS = SLP('My slpNLS')
start_time = time.clock()
slpNLS_epochs, slpNLS_mc_ratios, slpNLS_found = slpNLS.train(
            NLSdata.patterns,NLSdata.targets,nepochs=e_max)
slpNLS_elapsed_time = time.clock() - start_time
if slpNLS_found:
    print('The SLP found a solution in ' + str(slpNLS_epochs[-1]+1) + ' epochs!')
else:
    print('The SLP could not find a solution in ' + str(slpNLS_epochs[-1]+1) + ' epochs')
    print('Final ratio of misclassified points >>>',slpNLS_mc_ratios[-1])

print('Final weights: '+'\n'+'     W >>> '+str(slpNLS.W))
print('Time elapsed: '+str(slpNLS_elapsed_time))


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
