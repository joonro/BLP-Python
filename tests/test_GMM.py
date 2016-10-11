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

import sys

import numpy as np

sys.path.append('../')
import BLP


class Empty:
    pass


if __name__ == '__main__':
    import scipy.io

    matlab_ps2 = scipy.io.loadmat('ps2.mat')

    data = Empty()

    data.x1 = np.array(matlab_ps2['x1'].todense())
    data.x2 = np.array(matlab_ps2['x2'].copy())
    data.id_demo = matlab_ps2['id_demo'].reshape(-1, )
    data.D = np.array(matlab_ps2['demogr'])
    data.id = matlab_ps2['id'].reshape(-1, )
    data.v = np.array(matlab_ps2['v'])
    data.s_jt = matlab_ps2['s_jt'].reshape(-1, )
    data.ans = matlab_ps2['ans'].reshape(-1, )

    data.Z_org = scipy.io.loadmat('iv.mat')['iv']

    data.Z = np.c_[data.Z_org[:, 1:], data.x1[:, 1:]]

    data.nsimind = 20  # number of simulated "indviduals" per market
    data.nmkt = 94  # number of markets = (# of cities) * (# of quarters)
    data.nbrand = 24  # number of brands per market. if the numebr differs by market this requires some "accounting" vector

    # the difference is, each v will correspond to each x2, while
    # all 4 Z's will be used for each x2

    theta = np.array([[ 0.3772,  3.0888,      0,  1.1859,       0],
                      [ 1.8480, 16.5980, -.6590,       0, 11.6245],
                      [-0.0035, -0.1925,      0,  0.0296,       0],
                      [ 0.0810,  1.4684,      0, -1.5143,       0]])

    BLP = BLP.BLP(data)
    # BLP.init_GMM(theta, cython=True)
    BLP.init_GMM(theta, cython=False)
    assert BLP.GMM(theta) == 14.900789417012428

