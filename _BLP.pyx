#    BLP-Python provides an implementation of random coefficient logit model of
#    Berry, Levinsohn and Pakes (1995)
#    Copyright (C) 2011, 2013, 2016 Joon H. Ro
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
              double[:, :] theta2,
              double[:] ln_s_jt,
              double[:, :, :] v,
              double[:, :, :] D,
              double[:, :, :] X2,
              double etol, int iter_limit):
    """
    calculate delta (mean utility) through contraction mapping
    """
    cdef:
        int nmkts = v.shape[0]
        int nsiminds = v.shape[1]
        int nbrands = X2.shape[1]

    cdef:
        np.ndarray[np.float64_t, ndim=1] diff = np.empty(delta.shape[0])
        np.ndarray[np.float64_t, ndim=1] mktshr = np.empty(delta.shape[0])
        np.ndarray[np.float64_t, ndim=3] mu = np.zeros((nmkts, nsiminds, nbrands))

    _cal_mu(theta2, v, D, X2, mu)

    cdef:
        np.ndarray[np.float64_t, ndim=3] exp_mu = np.exp(mu)
        np.ndarray[np.float64_t, ndim=3] exp_xb = np.empty_like(exp_mu)

        int i, j, ix, mkt, ind, brand
        int niter = 0

        double denom
        double diff_max, diff_mean

    # contraction mapping
    while True:
        diff_mean = 0
        diff_max = 0

        # calculate market share
        for mkt in range(nmkts): # each market
            for ind in range(nsiminds): # each simulated individual
                denom = 1

                # calculate denominator
                for brand in range(nbrands):
                    exp_xb[mkt, brand, ind] = exp(delta[ix]) * exp_mu[mkt, brand, ind]
                    denom += exp_xb[mkt, brand, ind]

                ix = nbrands * mkt
                for brand in range(nbrands):
                    if ind == 0:  # initialize mktshr
                        mktshr[ix] = 0

                    mktshr[ix] += exp_xb[mkt, brand, ind] / (denom * nsiminds)

                    if ind + 1 == nsiminds:
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

def cal_mu(double[:, :] theta2,
           double[:, :, :] v,
           double[:, :, :] D,
           double[:, :, :] X2,
           ):
    '''
    calculate mu: the individual specific utility

    Delta is the effect of demographics on the preference parameter
    D is the demographics

    v is the vector of draws from the \( N(0, I_{k+1}) \)
    Simga is the scaling parameter

    mu = Delta @ D + Sigma @ v

    here v is nmkts-by-nsiminds-by-nvars
    '''
    cdef:
        int nmkts = v.shape[0]
        int nsiminds = v.shape[1]
        int nbrands = X2.shape[1]

        np.ndarray[np.float64_t, ndim=3] mu = np.zeros((nmkts, nsiminds, nbrands))

    _cal_mu(theta2, v, D, X2, mu)

    return mu

cdef double _cal_mu(double[:, :] theta2,
                    double[:, :, :] v,
                    double[:, :, :] D,
                    double[:, :, :] X2,
                    double[:, :, :] mu,
                    ) nogil except -1:

    cdef:
        int mkt, ind, k, d, j  # indices
        double beta_i  # individual params

        int nmkts = v.shape[0]
        int nsiminds = v.shape[1]
        int nbrands = X2.shape[1]
        int nvars = X2.shape[2]

    for mkt in prange(nmkts, nogil=True, schedule='guided'):  # each market
        for ind in range(nsiminds):  # each simulated individual
            for k in range(nvars):  # each betas
                beta_i = theta2[k, 0] * v[mkt, ind, k]

                for d in range(theta2.shape[1] - 1):
                    beta_i += theta2[k, d + 1] * D[mkt, ind, d]

                for j in range(nbrands):
                    mu[mkt, ind, j] += X2[mkt, j, k] * beta_i

def cal_s(double[:, :] delta, double[:, :, :] mu, double[:, :] s):
    ''' Calculate market share by numerical integration

    Parameters
    ----------
    delta : ndarray
        δ, mean utility (nmkts by nbrands)

    mu : ndarray
        μ, individual utility (nmkts by nsiminds by nbrands) 

    s : ndarray
        market share (nmkts by nbrands)

    '''
    cdef:
        int nmkts = mu.shape[0]
        int nsiminds = mu.shape[1]
        int nbrands = mu.shape[2]

        int mkt, ind, brand
        double denom, exp_Xb

    for mkt in prange(nmkts, nogil=True, schedule='guided'):  # each market
        for brand in prange(nbrands):
            s[mkt, brand] = 0

        for ind in range(nsiminds):  # each simulated individual
            denom = 1  # outside good

            for brand in range(nbrands):
                exp_Xb = exp(delta[mkt, brand] + mu[mkt, ind, brand])
                denom += exp_Xb

            for brand in range(nbrands):
                s[mkt, brand] += exp(delta[mkt, brand] + mu[mkt, ind, brand]) / (denom * nsiminds)

def cal_ind_choice_prob(
        double[:, :] delta,
        double[:, :, :] mu,
        double[:, :, :] ind_choice_prob,
        ):
    '''
    calculate individual choice probability

    Parameters
    ----------
    delta : ndarray
        δ, mean utility (nmkts by nbrands)

    mu : ndarray
        μ, individual utility (nmkts by nsiminds by nbrands) 

    ind_choice_prob : ndarray
        Output array of market share (nmkts by nsiminds by nbrands) 
    '''
    _cal_ind_choice_prob(delta, mu, ind_choice_prob)

cdef double _cal_ind_choice_prob(
        double[:, :] delta,  # mean utility (nmkts by nbrands)
        double[:, :, :] mu,  # individual utility (nmkts by nsiminds by nbrands)
        double[:, :, :] ind_choice_prob,
        ) nogil except -1:

    cdef:
        int nmkts = mu.shape[0]
        int nsiminds = mu.shape[1]
        int nbrands = mu.shape[2]

        int mkt, ind, brand

        double denom, exp_Xb

    for mkt in prange(nmkts, nogil=True, schedule='guided'):  # each market
        for ind in range(nsiminds):  # each simulated individual
            denom = 1

            for brand in range(nbrands):
                exp_Xb = exp(delta[mkt, brand] + mu[mkt, ind, brand])
                ind_choice_prob[mkt, ind, brand] = exp_Xb
                denom += exp_Xb

            for brand in range(nbrands):
                ind_choice_prob[mkt, ind, brand] /= denom
