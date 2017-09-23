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
#    along with this program. If not, see <http://www.gnu.org/licenses/>.

import os
import sys

import pytest

import numpy as np
import scipy.io
import xarray as xr

sys.path.append('../')
import pyBLP
import _BLP


class Data(object):
    ''' Synthetic data for Nevo (2000b)

    The file iv.mat contains the variable iv which consists of an id column
    (see the id variable above) and 20 columns of IV's for the price
    variable. The variable is sorted in the same order as the variables in
    ps2.mat.

    '''
    def __init__(self):
        try:
            ps2 = scipy.io.loadmat('examples/ps2.mat')
            Z_org = scipy.io.loadmat('examples/iv.mat')['iv']
        except:
            ps2 = scipy.io.loadmat('../examples/ps2.mat')
            Z_org = scipy.io.loadmat('../examples/iv.mat')['iv']

        nsiminds = self.nsiminds = 20  # number of simulated "indviduals" per market
        nmkts = self.nmkts = 94  # number of markets = (# of cities) * (# of quarters)
        nbrands = self.nbrands = 24  # number of brands per market

        ids = ps2['id'].reshape(-1, )
        self.ids = xr.DataArray(
            ids.reshape((nmkts, nbrands), order='F'),
            coords=[range(nmkts), range(nbrands)],
            dims=['markets', 'brands'],
            attrs={'Desc': 'an id variable in the format bbbbccyyq, where bbbb '
                           'is a unique 4 digit identifier for each brand (the '
                           'first digit is company and last 3 are brand, i.e., '
                           '1006 is K Raisin Bran and 3006 is Post Raisin Bran), '
                           'cc is a city code, yy is year (=88 for all observations '
                           'in this data set) and q is quarter.'}
            )

        X1 = np.array(ps2['x1'].todense())

        self.X1 = xr.DataArray(
            X1.reshape(nmkts, nbrands, -1),
            coords=[range(nmkts), range(nbrands),
                    ['Price'] + ['Brand_{}'.format(brand)
                                 for brand in range(nbrands)]],
            dims=['markets', 'brands', 'vars'],
            attrs={'Desc': 'the variables that enter the linear part of the '
                           'estimation. Here this consists of a price variable '
                           '(first column) and 24 brand dummy variables.'}
            )
            
        X2 = np.array(ps2['x2'].copy())
        self.X2 = xr.DataArray(
            X2.reshape(nmkts, nbrands, -1),
            coords=[range(nmkts), range(nbrands),
                    ['Constant', 'Price', 'Sugar', 'Mushy']],
            dims=['markets', 'brands', 'vars'],
            attrs={'Desc': 'the variables that enter the non-linear part of the '
                           'estimation.'}
            )
            
        self.id_demo = ps2['id_demo'].reshape(-1, )

        D = np.array(ps2['demogr'])
        self.D = xr.DataArray(
            D.reshape((nmkts, nsiminds, -1), order='F'),
            coords=[range(nmkts), range(nsiminds),
                    ['Income', 'Income^2', 'Age', 'Child']],
            dims=['markets', 'nsiminds', 'vars'],
            attrs={'Desc': 'Demeaned draws of demographic variables from the CPS for 20 '
                           'individuals in each market.',
                   'Child': 'Child dummy variable (=1 if age <= 16)'}
            )

        v = np.array(ps2['v'])
        self.v = xr.DataArray(
            v.reshape((nmkts, nsiminds, -1), order='F'),
            coords=[range(nmkts), range(nsiminds),
                    ['Constant', 'Price', 'Sugar', 'Mushy']],
            dims=['markets', 'nsiminds', 'vars'],
            attrs={'Desc': 'random draws given for the estimation.'}
            )

        s_jt = ps2['s_jt'].reshape(-1, )  # s_jt for nmkts * nbransd
        self.s_jt = xr.DataArray(
            s_jt.reshape((nmkts, nbrands)),
            coords=[range(nmkts), range(nbrands),],
            dims=['markets', 'brands'],
            attrs={'Desc': 'Market share of each brand.'}
            )

        self.ans = ps2['ans'].reshape(-1, )

        Z = np.c_[Z_org[:, 1:], X1[:, 1:]]
        self.Z = xr.DataArray(
            Z.reshape((self.nmkts, self.nbrands, -1)),
            coords=[range(nmkts), range(nbrands), range(Z.shape[-1])],
            dims=['markets', 'brands', 'vars'],
            attrs={'Desc': 'Instruments'}
            )


@pytest.fixture(scope="module")
def data():
    return(Data())


def test_cal_mu(data):
    BLP = pyBLP.BLP(data)

    v, D, X2 = BLP.v, BLP.D, BLP.X2

    θ20 = np.array([[ 0.3772,  3.0888,      0,  1.1859,       0],
                   [ 1.8480, 16.5980, -.6590,       0, 11.6245],
                   [-0.0035, -0.1925,      0,  0.0296,       0],
                   [ 0.0810,  1.4684,      0, -1.5143,       0]])

    mu_python = BLP._cal_mu(θ20)
    mu_cython = _BLP.cal_mu(θ20, v.values, D.values, X2.values)

    assert np.allclose(mu_python, mu_cython)

def test_cal_s(data):
    BLP = pyBLP.BLP(data)

    v, D, X2 = BLP.v, BLP.D, BLP.X2

    θ20 = np.array([[ 0.3772,  3.0888,      0,  1.1859,       0],
                   [ 1.8480, 16.5980, -.6590,       0, 11.6245],
                   [-0.0035, -0.1925,      0,  0.0296,       0],
                   [ 0.0810,  1.4684,      0, -1.5143,       0]])

    BLP.GMM(θ20)

    δ, μ = BLP.δ, BLP.μ

    s_python = BLP._cal_s(δ, μ)
    s_cython = BLP.s
    _BLP.cal_s(δ, μ, s_cython)

    assert np.allclose(s_python, s_cython)

def test_GMM(data):
    """ Replicate Nevo (2000b) results
    """
    BLP = pyBLP.BLP(data)

    θ20 = np.array([[ 0.3772,  3.0888,      0,  1.1859,       0],
                    [ 1.8480, 16.5980, -.6590,       0, 11.6245],
                    [-0.0035, -0.1925,      0,  0.0296,       0],
                    [ 0.0810,  1.4684,      0, -1.5143,       0]])

    assert np.allclose(BLP.GMM(θ20), 14.900789417012428)

def test_arguments(data):
    """Pass variables separately
    """
    s_jt, X1, X2, Z, v, D = data.s_jt, data.X1, data.X2, data.Z, data.v, data.D 

    BLP = pyBLP.BLP(s_jt=s_jt, X1=X1, X2=X2, Z=Z, v=v, D=D)

    θ20 = np.array([[ 0.3772,  3.0888,      0,  1.1859,       0],
                    [ 1.8480, 16.5980, -.6590,       0, 11.6245],
                    [-0.0035, -0.1925,      0,  0.0296,       0],
                    [ 0.0810,  1.4684,      0, -1.5143,       0]])

    assert np.allclose(BLP.GMM(θ20), 14.900789417012428)

def test_full_estimation(data):
    BLP = pyBLP.BLP(data)

    θ20 = np.array([[ 0.5580,   3.0888,        0,  1.2844,       0],
                    [ 3.3124, 588.3252, -30.1920,       0, 11.0546],
                    [-0.0057,  -0.3849,        0,  0.0522,       0],
                    [ 0.0934,   0.7483,        0, -1.3533,       0]])

    BLP.estimate(θ20=θ20)

    assert np.allclose(BLP.results['GMM'], 4.561514655033999)


if __name__ == '__main__':
    pass
