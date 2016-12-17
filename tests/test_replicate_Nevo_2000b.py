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

import os
import sys

import pytest

import numpy as np
import scipy.io

sys.path.append('../')
import pyBLP


class Data(object):
    def __init__(self):
        try:
            ps2 = scipy.io.loadmat('tests/ps2.mat')
            self.Z_org = scipy.io.loadmat('tests/iv.mat')['iv']
        except:
            ps2 = scipy.io.loadmat('ps2.mat')
            self.Z_org = scipy.io.loadmat('iv.mat')['iv']

        self.nsimind = 20  # number of simulated "indviduals" per market
        self.nmkt = 94  # number of markets = (# of cities) * (# of quarters)
        self.nbrand = 24  # number of brands per market. if the numebr differs by market this requires some "accounting" vector

        self.x1 = np.array(ps2['x1'].todense())
        self.x2 = np.array(ps2['x2'].copy())
        self.id_demo = ps2['id_demo'].reshape(-1, )
        self.D = np.array(ps2['demogr'])
        self.id = ps2['id'].reshape(-1, )
        self.v = np.array(ps2['v'])
        self.s_jt = ps2['s_jt'].reshape(-1, )  # s_jt for nmkt * nbrand
        self.ans = ps2['ans'].reshape(-1, )

        self.Z = np.c_[self.Z_org[:, 1:], self.x1[:, 1:]]


@pytest.fixture(scope="module")
def data():
    return(Data())


def test_replicate_Nevo(data):
    # the difference is, each v will correspond to each x2, while
    # all 4 Z's will be used for each x2

    BLP = pyBLP.BLP(data)

    θ2 = np.array([[ 0.3772,  3.0888,      0,  1.1859,       0],
                   [ 1.8480, 16.5980, -.6590,       0, 11.6245],
                   [-0.0035, -0.1925,      0,  0.0296,       0],
                   [ 0.0810,  1.4684,      0, -1.5143,       0]])

    assert np.allclose(BLP.GMM(θ2), 14.900789417012428)

if __name__ == '__main__':
    data = Data()

    BLP = pyBLP.BLP(data)

    θ20 = np.array([[ 0.3772,  3.0888,      0,  1.1859,       0],
                    [ 1.8480, 16.5980, -.6590,       0, 11.6245],
                    [-0.0035, -0.1925,      0,  0.0296,       0],
                    [ 0.0810,  1.4684,      0, -1.5143,       0]])

    BLP.estimate(θ20=θ20, method='Nelder-Mead', maxiter=1)

    # Run the line below to get true estimation results
    # BLP.estimate(θ20=θ20, method='Nelder-Mead')
