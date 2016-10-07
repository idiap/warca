"""
 *  Cython function for computing chi2 distance
 *
 *  Copyright (c) 2016 Idiap Research Institute, http://www.idiap.ch/
 *  Written by Cijo Jose <cijose@idiap.ch>
 *
 *  This file is part of tSNE.
 *
 *  tSNE is free software: you can redistribute it and/or modify it
 *  under the terms of the GNU General Public License version 3 as
 *  published by the Free Software Foundation.
 *
 *  tSNE is distributed in the hope that it will be useful, but WITHOUT
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 *  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
 *  License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with selector.  If not, see <http://www.gnu.org/licenses/>.
 *
"""
from libc.string cimport memset
import numpy as np
cimport numpy as np

cdef extern from "cblas.h":
    double cblas_dasum(int, const double *, int) nogil

ctypedef float [:, :] float_array_2d_t 
ctypedef double [:, :] double_array_2d_t

ctypedef unsigned short [:] int_array_1d_t

cdef fused floating_array_2d_t:
    float_array_2d_t
    double_array_2d_t

cdef fused integer_array_1d_t:
    int_array_1d_t


np.import_array()


def _chi2_kernel_fast(floating_array_2d_t X,
                      floating_array_2d_t Y,
                      floating_array_2d_t result):
    cdef np.npy_intp i, j, k
    cdef np.npy_intp n_samples_X = X.shape[0]
    cdef np.npy_intp n_samples_Y = Y.shape[0]
    cdef np.npy_intp n_features = X.shape[1]
    cdef double res, nom, denom
    with nogil:
        for i in range(n_samples_X):
            for j in range(n_samples_Y):
                res = 0
                for k in range(n_features):
                    denom = (X[i, k] - Y[j, k])
                    nom = (X[i, k] + Y[j, k])
                    if nom != 0:
                        res  += denom * denom / nom
                        result[i, j] = res

                        
def _sigmoid(x):
    return 1./(1. + np.exp(-x))
                            
                        
def _pcca_gradient_helper(floating_array_2d_t K,
                        floating_array_2d_t A,
                        integer_array_1d_t Y,
                          floating_array_2d_t result):
    cdef np.npy_intp i, j
    cdef np.npy_intp n_samples_K =  K.shape[0]
    cdef np.npy_intp n_samples_A =  A.shape[0]
    cdef double res, yij, tmp
    cdef double loss = 0.
    for i in range(n_samples_K):
        for j in range(i+1, n_samples_K):
            if i==j:
                continue
            res = np.linalg.norm(np.dot(A, np.subtract(K[:, i], K[:, j])))**2 - 1.
            yij = 1. if Y[i] == Y[j] else -1. 
            res *= yij
            tmp = _sigmoid(res)
            loss += np.log(1. +(1./(1.-tmp)))
            res =  tmp * yij
            result[i, i] += res
            result[j, j] += res
            result[i, j] -= res
            result[j, i] -= res
    return loss
        
