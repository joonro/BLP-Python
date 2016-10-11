#    BLP-Python provides an implementation of random coefficient logit model of 
#    Berry, Levinsohn and Pakes (1995)
#    Copyright (C) 2011, 2013 Joon H. Ro
#
#    This file is part of BLP-Python.
#
#    BLP-Python is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    BLP-Python is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as np

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np

from cython.parallel import prange
from libc.math cimport abs, exp, fabs, log

cimport cython

def cal_delta(double[:] delta,
              double[:] theta_v,
              double[:, :] theta_D,
              double[:] ln_s_jt,
              double[:, :] v,
              double[:, :] D,
              double[:, :] x2,
              int nmkt, int nsimind, int nbrand,
              double etol, int iter_limit):
    """
    calculate delta (mean utility) through contraction mapping
    """
    cdef:
        np.ndarray[np.float64_t, ndim=1] diff = np.empty(delta.shape[0])
        np.ndarray[np.float64_t, ndim=1] mktshr = np.empty(delta.shape[0])
        np.ndarray[np.float64_t, ndim=2] mu = np.zeros((x2.shape[0], nsimind))

    _cal_mu(theta_v, theta_D, v, D, x2, mu, nmkt, nsimind, nbrand)

    cdef:    
        np.ndarray[np.float64_t, ndim=2] exp_mu = np.exp(mu)
        np.ndarray[np.float64_t, ndim=2] exp_xb = np.empty_like(exp_mu)

        int i, j, ix, mkt, ind, brand
        int niter = 0

        double denom
        double diff_max, diff_mean

    # contraction mapping
    while True:
        diff_mean = 0 
        diff_max = 0
        
        # calculate market share
        for mkt in range(nmkt): # each market
            for ind in range(nsimind): # each simulated individual
                denom = 1

                # calculate denominator
                ix = nbrand * mkt
                for brand in range(nbrand):
                    exp_xb[ix, ind] = exp(delta[ix]) * exp_mu[ix, ind]
                    denom += exp_xb[ix, ind]

                    ix += 1
    
                ix = nbrand * mkt
                for brand in range(nbrand):
                    if ind == 0:  # initialize mktshr
                        mktshr[ix] = 0
                
                    mktshr[ix] += exp_xb[ix, ind] / (denom * nsimind)
                    
                    if ind + 1 == nsimind:
                        # the last individual - mktshr calculation is done
                        # calculate the difference here to save some loop
                        diff[ix] = ln_s_jt[ix] - log(mktshr[ix])
                        
                        delta[ix] += diff[ix]
                    
                        if abs(diff[ix]) > diff_max:
                            diff_max = abs(diff[ix])
                            
                        diff_mean += diff[ix]

                    ix += 1
        
        diff_mean /= delta.shape[0]
        
        if (diff_max < etol) and (diff_mean < 1e-3) or niter > iter_limit:
            break

        niter += 1

    print('contraction mapping finished in {} iterations'.format(niter)) 

def cal_mu(double[:] theta_v,
           double[:, :] theta_D,
           double[:, :] v,
           double[:, :] D,
           double[:, :] x2,
           int nmkt, 
           int nsimind, 
           int J):
    '''
    calculate mu: the individual specific utility

    Delta is the effect of demographics on the preference parameter
    Z is the demographics

    v is the vector of draws from the \( N(0, I_{k+1}) \)
    Simga is the scaling parameter

    mu = dot(Delta, Z) + dot(Sigma, v)

    here v is nmkt-by-nsimind-by-nvar
    '''
    cdef np.ndarray[np.float64_t, ndim=2] mu = np.zeros((x2.shape[0], nsimind))

    _cal_mu(theta_v, theta_D, v, D, x2, mu, nmkt, nsimind, J)

    return mu

cdef double _cal_mu(double[:] theta_v,
                    double[:, :] theta_D,
                    double[:, :] v,
                    double[:, :] D,
                    double[:, :] x2,
                    double[:, :] mu,
                    int nmkt, 
                    int nsimind, 
                    int J) nogil except -1:

    cdef:
        int mkt, ind, k, d, j, ix # indices
        double tmp_mu

    for mkt in range(nmkt): # each market
        for ind in range(nsimind): # each simulated individual
            for k in range(theta_v.shape[0]): # each betas
                tmp_mu = theta_v[k] * v[mkt, nsimind * k + ind]

                for d in range(theta_D.shape[1]): # demographics(Z)
                    tmp_mu += theta_D[k, d] * D[mkt, nsimind * d + ind]

                ix = J * mkt
                for j in range(J):
                    mu[ix, ind] += x2[ix, k] * tmp_mu
                    ix += 1

def cal_mktshr(
        double[:, :] exp_xb, int nmkt, int nsimind, int nbrand):
    '''
    calculate market share

    Parameters
    ----------
    N : exp_xb
        exp(delta + mu)

    nmkt : int
        number of markets

    nsimind : int
        number of simulated individuals

    nbrand : int
        number of brands (alternatives)

    Returns
    -------
    mktshr : ndarray
        (N, ) Output array of market share
    '''
    # given mu, calculate delta
    cdef np.ndarray[np.float64_t, ndim=1] mktshr = np.zeros((exp_xb.shape[0], ))

    cdef int mkt, ind, brand, ix

    cdef double denom
    
    for mkt in prange(nmkt, nogil=True, schedule='guided'):  # each market
        for ind in range(nsimind):  # each simulated individual
            denom = 1

            ix = nbrand * mkt
            for brand in range(nbrand):
                denom += exp_xb[ix, ind]
                ix += 1

            ix = nbrand * mkt
            for brand in range(nbrand):
                mktshr[ix] += exp_xb[ix, ind] / (denom * nsimind)
                ix += 1

    return mktshr

def cal_ind_choice_prob(
        double[:, :] exp_xb, int nmkt, int nsimind, int nbrand):
    '''
    calculate individual choice probability

    Parameters
    ----------
    N : exp_xb
        exp(delta + mu)

    nmkt : int
        number of markets

    nsimind : int
        number of simulated individuals

    nbrand : int
        number of brands (alternatives)

    Returns
    -------
    mktshr : ndarray
        (N, ) Output array of market share
    '''
    # given mu, calculate delta
    cdef:
        np.ndarray[np.float64_t, ndim=2] ind_choice_prob = np.empty((exp_xb.shape[0], exp_xb.shape[1]))
        int mkt, ind, brand
        int ix_base
        double denom

    for mkt in range(nmkt):  # each market
        ix_base = nbrand * mkt

        for ind in range(nsimind):  # each simulated individual
            denom = 1

            for brand in range(nbrand):
                denom += exp_xb[ix_base + brand, ind]

            for brand in range(nbrand):
                ind_choice_prob[ix_base + brand, ind] = exp_xb[ix_base + brand, ind]
                ind_choice_prob[ix_base + brand, ind] /= denom

    return ind_choice_prob
