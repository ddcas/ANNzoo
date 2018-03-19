#!/usr/bin/env python

__author__ = "Daniel del Castillo"
__version__ = "0.0.0"

"""
This is a plain vanilla implementation of a two-layer perceptron.
An example of use is also included.
"""

import numpy as np
import random
import time

from Perceptron import Perceptron

# only needed for the examples
from Dataset import Dataset


def phi(x):
    return 2.0/(1+np.exp(-x))-1

def dphi(x):
    return ((1+x)*(1-x))*0.5


class TLP(Perceptron):
    def __init__(self,name,size_in=None,size_out=None,W=None,size_h=None,V=None):
        Perceptron.__init__(self,name,size_in,size_out,W)
        self.size_h = W.shape[0] if W else None
        self.V = V

    def fw(self,X):
        N = X.shape[1]
        H_in = self.W@X
        # H_out = np.vstack((np.ones((1,N)),phi(H_in)))   # OPTION 1
        H_out = np.vstack((phi(H_in),np.ones((1,N)))) # OPTION 2
        O_in = self.V@H_out
        O_out = phi(O_in)

        return H_out, O_out

    def bw(self,O_out,T,H_out):
        N = T.shape[1]
        delta_o = (O_out-T)*dphi(O_out)
        delta_h_ = self.V.T@delta_o*dphi(H_out)
        delta_h = np.empty((self.size_h,N))
        # delta_h = delta_h_[self.size_h:,:]   # OPTION 1
        delta_h = delta_h_[:self.size_h,:]   # OPTION 2

        return delta_o, delta_h

    def train(self,X,T,h=1,eta=0.001,alpha=0.9,nepochs=20,learn_mode=[0,0],batch_size=0, \
              reshuffle=False,header=True,fig=[],wait=0):

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
            print('Training:',self.name,'| lr',eta,'| alpha',alpha,'|',nepochs,'epochs','|',lm)

        # initialize dimensions
        N = X.shape[1]
        if not self.size_in: self.size_in = X.shape[0]-1
        if not self.size_out: self.size_out = T.shape[0]
        if not self.size_h: self.size_h = h
        if not self.W: self.W = np.random.randn(self.size_h,self.size_in+1)*0.01
        if not self.V: self.V = np.random.randn(self.size_out,self.size_h+1)*0.01
        dW, dV = np.zeros_like(self.W), np.zeros_like(self.V)
        # initialize epoch array
        epochs = [0]
        # initialize metrics
        mc_ratios = []
        # initial forward pass
        H_out, O_out = self.fw(X)
        # initial prediction
        found = np.array_equal(np.sign(O_out),T)
        # training loop
        while not found:
            # find misclassified points
            mc_points = np.sign(O_out) == T
            mc_ratios.append(1-np.sum(mc_points)/N)
            # backward pass
            delta_o, delta_h = self.bw(O_out,T,H_out)
            # update weights
            dW = alpha*dW - delta_h@X.T*(1-alpha)
            dV = alpha*dV - delta_o@H_out.T*(1-alpha)
            # print('dW>>>',dW, end="\r")
            self.W += eta*dW
            self.V += eta*dV
            # new forward pass
            H_out, O_out = self.fw(X)
            found = np.array_equal(np.sign(O_out),T)
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
# 1) Two-Layer Perceptron for NLS data
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def gen_NLdataset(N,d,z,c,sd,labels):
    # generate patterns and targets
    N = int(N/2)
    if N%2==1:
        N += 1
    X = []
    X.append([])
    X.append([])
    covs = (sd**2)*np.eye(d,d)
    # generate patterns and targets
    for I in range(2):
        points = [np.random.multivariate_normal([0,0],covs,N//2) for i in range(c)]
        for i in range(N//2):
            X[0].append([points[0][i][0]+(5*((-1)**I)), points[0][i][1]+(-5*((-1)**I))])
            X[1].append([points[1][i][0]+(5*((-1)**I)), points[1][i][1]+(5*((-1)**I))])

    P = np.vstack(X)
    XT = np.hstack((P,labels.T))
    # shuffle input and output patterns and targets together
    np.random.shuffle(XT)
    patterns = np.ones((d+1,N*2))
    patterns[:-1,:] = np.copy(XT[:,:-z].T)
    targets = np.copy(XT[:,-z:].T)

    return patterns, targets, X

# dimensions
N = 200 # the number of example patterns, must be an even number
d = 2 # the dimension of the input patterns
z = 1 # the dimension of the output patterns
c = 2 # the number of categories
sd = 1

# define the classes and their values
categories  = np.linspace(-d+1,d-1,d)
labels = np.array([[categories[0]]*(N//2)+[categories[1]]*(N//2)])
if int(N/8)%2==1:
    TN = int(N/4)+2
else:
    TN = int(N/4)

labels2 = np.array([[categories[0]]*(TN//2)+[categories[1]]*(TN//2)])

# generate dataset
LSdata = Dataset(N,d,z,c,labels,sd=0.25,mean_centre=0,mean_sd=1)
tests, clas, testxy = gen_NLdataset(TN,d,z,c,sd,labels2)

# define the hidden layer size
hLS = 1
hNLS = 2
# specify maximum number of epochs
e_maxLS = 100
e_maxNLS = 200

tlpLS = TLP('My tlpLS')
start_time = time.clock()
tlpLS_epochs, tlpLS_mc_ratios, tlpLS_found = tlpLS.train(
        LSdata.patterns,LSdata.targets,hLS,nepochs=e_maxLS)
tlpLS_elapsed_time = time.clock() - start_time

if tlpLS_found:
    print('The MLP found a solution in ' + str(tlpLS_epochs[-1]+1) + ' epochs!')
else:
    print('The MLP could not find a solution in ' + str(tlpLS_epochs[-1]+1) + ' epochs')
    print('Final ratio of misclassified points >>>',tlpLS_mc_ratios[-1])
    print('Best ratio of misclassified points >>>',min(tlpLS_mc_ratios))

print('Final weights: '+'\n'+'     W >>> '+str(tlpLS.W),'\n','     V >>> '+str(tlpLS.V))
print('Time elapsed: '+str(tlpLS_elapsed_time))

tlpNLS = TLP('My tlpNLS')
start_time = time.clock()
tlpNLS_epochs, tlpNLS_mc_ratios, tlpNLS_found = tlpNLS.train(
        tests,clas,hNLS,nepochs=e_maxNLS,eta=0.001,alpha=0.9)
tlpNLS_elapsed_time = time.clock() - start_time

if tlpNLS_found:
    print('The MLP found a solution in ' + str(tlpNLS_epochs[-1]+1) + ' epochs!')
else:
    print('The MLP could not find a solution in ' + str(tlpNLS_epochs[-1]+1) + ' epochs')
    print('Final ratio of misclassified points >>>',tlpNLS_mc_ratios[-1])
    print('Best ratio of misclassified points >>>',min(tlpNLS_mc_ratios))
#
print('Final weights: '+'\n'+'     W >>> '+str(tlpNLS.W),'\n','     V >>> '+str(tlpNLS.V))
print('Time elapsed: '+str(tlpNLS_elapsed_time))
