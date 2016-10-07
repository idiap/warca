"""
 *  This is an implementation of tSNE dimensionality reduction algorithm 
 *  written in python.
 *
 *  Copyright (c) 2016 Idiap Research Institute, http://www.idiap.ch/
 *  Written by Cijo Jose <cijose@idiap.ch>
 *
 *  This is free software: you can redistribute it and/or modify it
 *  under the terms of the GNU General Public License version 3 as
 *  published by the Free Software Foundation.
 *
 *  This is distributed in the hope that it will be useful, but WITHOUT
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 *  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
 *  License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with selector.  If not, see <http://www.gnu.org/licenses/>.
 *

"""
import numpy as np
import colorsys
import scipy
from scipy.misc import imresize
import plot_utils as plt_utils
from fast_functions import _chi2_kernel_fast
        

class tSNE:
    def __init__(self, x, perplexity = 30, tol = 1e-7, dim=2, distance='euc'):
        n = x.shape[0]
        D = np.zeros((n, n), dtype=x.dtype)
        if distance == 'euc':
            D  = np.dot(x, x.T)
            d = np.diag(D)
            D =  d + d[:, np.newaxis] - 2. * D
        elif distance == 'chi2':
            _chi2_kernel_fast(x, x, D)
        elif distance == 'pre':
            assert x.shape[0] == x.shape[1]
            D = x
        D /= np.max(D)
        n = D.shape[0]
        entropy_target = np.log(perplexity)
        self.P = np.zeros((n, n), dtype = D.dtype)
        pj = np.zeros(n, dtype = D.dtype)
        max_tries = 50
        for i in range(n):
            gamma_min = -np.inf
            gamma_max = np.inf
            gamma = 1.

            num_tries = 0
            flag = True
            while( flag ):
                pj = np.exp(-D[i, :] * gamma)
                #print pj
                pj[i] = 0.
                psum = np.sum(pj)
                pj /= psum
                entropy_gamma = -np.sum(pj[pj > 1e-7] * np.log(pj[pj > 1e-7]))
                if entropy_gamma > entropy_target:
                    gamma_min = gamma
                    if gamma_max == np.inf:
                        gamma *= 2.
                    else:
                        gamma = (gamma_min + gamma_max) / 2.
                else:
                    gamma_max = gamma
                    if gamma_min == -np.inf:
                        gamma /= 2.
                    else:
                        gamma = (gamma_min + gamma_max) / 2.
                 
                num_tries += 1;
                #print np.abs(entropy_gamma - entropy_target)        
                if(np.abs(entropy_gamma - entropy_target) < tol or num_tries >= max_tries):
                    flag = False
            
            self.P[i, :] = pj

        self.P = (self.P + self.P.T) * 0.5 / n
        self.P[self.P < 1e-100] = 1e-100
        self.constant  = np.sum(self.P * np.log(self.P))
        self.Y = np.random.randn(n ,dim) * 1e-4
        self.gains = np.ones((n, dim), dtype = self.P.dtype)
        self.y_step = np.zeros((n, dim), dtype = self.P.dtype)
        self.grad =  np.zeros((self.Y.shape), dtype = self.Y.dtype)
        self.iter = 0
        self.momentum = 0.5                                     # initial momentum
        self.final_momentum = 0.8                               # value to which momentum is changed
        self.mom_switch_iter = 250                              # iteration at which momentum is changed
        self.stop_lying_iter = 100                              # iteration at which lying about P-values is stopped
        self.max_iter = 1000                                    # maximum number of iterations
        self.epsilon = 500                                      # initial learning rate
        self.min_gain = .01
        self.initial_solution = False


    def func_grad(self):
        Q = np.dot(self.Y, self.Y.T)
        q = np.diag(Q)
        Q = q + q[:, np.newaxis] - 2.*Q + 1.
        Q = 1. / Q
        np.fill_diagonal(Q, 0.)

        qsum = np.sum(Q)
        Q = Q / qsum
        
        Q[Q < 1e-100] = 1e-100

        kl_div = self.constant - np.sum(self.P * np.log(Q))
        L = (self.P - Q) * Q * qsum
        
        self.grad = 4. * np.dot((np.diag(np.sum(L, axis=0)) - L),  self.Y)
                    
        return kl_div

    def take_step(self):
        f = self.func_grad()
        if self.iter % 10 == 0:
            print 'Iteration '+ str(self.iter) +' Loss '+str(f)
        self.gains = (self.gains + 0.2) * (np.sign(self.grad) != np.sign(self.y_step)) + (self.gains * 0.8) *(np.sign(self.grad) == np.sign(self.y_step))
        self.gains[self.gains < self.min_gain] = self.min_gain
        self.y_step = self.momentum * self.y_step - self.epsilon * (self.gains * self.grad)
        self.Y += self.y_step
        self.Y -= np.mean(self.Y, axis=0)
        self.iter += 1
        if self.iter == self.mom_switch_iter:
            self.momentum = self.final_momentum


    def get_points(self):
        return self.Y
    

import sys
import os
import re
if __name__ == '__main__':
    feature_file = sys.argv[1]
    _,fname = os.path.split(feature_file)
    plot_name = re.split(r'[.](?![^][]*\])', fname)[0]
    print plot_name
    label_file = sys.argv[2]
    distance_ = sys.argv[3]
    X = np.loadtxt(feature_file, delimiter=',')
    y = np.loadtxt(label_file)
    print X.shape
    print y.shape
    obj = tSNE(X, distance = distance_)
    for i in range(2000):
        obj.take_step()
    Xtsne = obj.get_points()
    plt_utils.plot_dataset(Xtsne, y, name = plot_name, title = "tSNE visualization of " + plot_name)
        
