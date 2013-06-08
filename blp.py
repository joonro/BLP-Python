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

from __future__ import division

# Standard libraries
import time

# Scientific libraries
from numpy import *
from numpy.linalg import solve, cholesky
from scipy.linalg import cho_solve

import scipy.optimize as optimize
#seterr(all='ignore')
#set_printoptions(precision=10, suppress='None')

# custom libraries
import _blp

import IPython


class BLP:
    '''Random coefficient logit model'''

    def __init__(self, data):
        """Constructor"""

        s_jt = self.s_jt = data.s_jt
        ln_s_jt = self.ln_s_jt = log(self.s_jt)
        v = self.v = data.v
        D = self.D = data.D
        x1 = self.x1 = data.x1
        x2 = self.x2 = data.x2
        Z = self.Z = data.Z

        self.nmkt = data.nmkt
        self.nbrand = data.nbrand
        self.nsimind = data.nsimind

        self.nx2 = self.x2.shape[1]
        self.nD = self.D.shape[1] // self.nsimind

        # choleskey root (lower triangular) of the weighting matrix.
        # do not invert it yet
        LW = self.LW = (cholesky(dot(Z.T, Z)), True)

        # Z'x1
        Z_x1 = self.Z_x1 = Z.T.dot(x1)

        self.etol = 1e-6
        self.iter_limit = 200

        self.GMM_old = 0
        self.GMM_diff = 1

        # calculate market share
        # outside good
        s_out = self.s_out = (1 - self.s_jt.reshape(self.nmkt, -1).sum(axis=1))

        y = self.y = log(s_jt.reshape(self.nmkt, -1))
        y -= log(s_out.reshape(-1, 1))
        y.shape = (-1, )

        # initial delta
        self.delta0 = self.x1.dot(solve(Z_x1.T.dot(cho_solve(LW, Z_x1)),
                                        Z_x1.T.dot(cho_solve(LW, Z.T.dot(y)))))

        self.delta = self.delta0.copy()

        self._blp = _blp

    def init_GMM(self, theta, cython=True):
        """intialize theta"""

        self.cython = cython
        self.ix_theta = nonzero(theta)
        self.theta = theta.copy()

    def GMM(self, theta):
        """wrapper around _GMM objective function"""

        self.theta = theta.copy()

        self._GMM(self.theta[self.ix_theta])

    def _GMM(self, theta_vec):
        """GMM objective function"""

        _blp, theta, delta, v, D, x2, nmkt, nsimind, nbrand = self.set_aliases()

        theta[self.ix_theta] = theta_vec
        theta_v = theta[:, 0]
        theta_D = theta[:, 1:]

        # adaptive etol
        if self.GMM_diff < 1e-6:
            etol = self.etol = 1e-13
        elif self.GMM_diff < 1e-3:
            etol = self.etol = 1e-12
        else:
            etol = self.etol = 1e-9

        if self.cython:
            _blp.cal_delta(delta,
                           theta_v, theta_D,
                           self.ln_s_jt,
                           v, D, x2, nmkt, nsimind, nbrand,
                           etol, self.iter_limit)
        else:
            self.cal_delta(theta)

        if isnan(delta).sum():
            return(1e+10)

        Z_x1 = self.Z_x1
        LW = self.LW

        # Z'delta
        Z_delta = self.Z.T.dot(delta)

        #\[ \theta_1 = (\tilde{X}'ZW^{-1}Z'\tilde{X})^{-1}\tilde{X}'ZW^{-1}Z'\delta \]
        theta1 = solve(Z_x1.T.dot(cho_solve(LW, Z_x1)),
                       Z_x1.T.dot(cho_solve(LW, Z_delta)))

        xi = self.xi = delta - self.x1.dot(theta1)

        # Z'xi
        Z_xi = self.Z.T.dot(xi)

        # \[ (\delta - \tilde{X}\theta_1)'ZW^{-1}Z'(\delta-\tilde{X}\theta_1) \]
        GMM = Z_xi.T.dot(cho_solve(LW, Z_xi))

        self.GMM_diff = abs(self.GMM_old - GMM)
        self.GMM_old = GMM

        print('GMM value: {0}'.format(GMM))
        return(GMM)

    def gradient(self, theta):
        """
        wrapper around gradient of GMM objective function
        you can pass theta as an matrix here
        """

        self.theta = theta.copy()

        return(self._gradient(self.theta[self.ix_theta]))

    def _gradient(self, theta_vec):
        """gradient of GMM objective function"""

        self.theta[self.ix_theta] = theta_vec

        temp = self.cal_jacobian(self.theta).T

        return(2 * temp.dot(self.Z).dot(cho_solve(self.LW, self.Z.T)).dot(self.xi))

    def cal_delta(self, theta):
        """
        calculate delta (mean utility) from contraction mapping
        """
        _blp, theta, delta, v, D, x2, nmkt, nsimind, nbrand = self.set_aliases()

        theta_v = theta[:, 0]
        theta_D = theta[:, 1:]

        niter = 0

        exp_mu = exp(_blp.cal_mu(theta_v, theta_D,
                                 v, D, x2,
                                 nmkt, nsimind, nbrand))

        while True:
            diff = self.ln_s_jt.copy()

            exp_xb = exp(delta.reshape(-1, 1)) * exp_mu

            diff -= log(_blp.cal_mktshr(exp_xb, nmkt, nsimind, nbrand))

            if isnan(diff).sum():
                print('nan in diffs')
                break

            delta += diff

            if (abs(diff).max() < self.etol) and (abs(diff).mean() < 1e-3):
                break

            niter += 1

        print('contraction mapping finished in {} iterations'.format(niter))

    def cal_var_covar_mat(self, theta_vec):
        """calculate variance covariance matrix"""

        self.theta[self.ix_theta] = theta_vec

        LW = self.LW

        jacobian = self.cal_jacobian(self.theta)

        a = c_[self.x1, jacobian].T.dot(self.Z)

        Zres = self.Z * self.xi.reshape(-1, 1)
        b = Zres.T.dot(Zres)

        # inv(a*invW*a')*a*invW*b*invW*a'*inv(a*invW*a');

        tmp = solve(a.dot(cho_solve(LW, a.T)), a.dot(cho_solve(LW, b)).dot(cho_solve(LW, a.T)))

        return(solve(a.dot(cho_solve(LW, a.T)).T, tmp.T).T)

    def cal_jacobian(self, theta):
        """calculate the Jacobian"""

        _blp, theta, delta, v, D, x2, nmkt, nsimind, nbrand = self.set_aliases()

        mu = _blp.cal_mu(theta[:, 0], theta[:, 1:], v, D, x2, nmkt, nsimind, nbrand)

        exp_xb = exp(delta.reshape(-1, 1) + mu)

        ind_choice_prob = _blp.cal_ind_choice_prob(exp_xb, nmkt, nsimind, nbrand)

        nk = self.x2.shape[1]
        nD = theta.shape[1] - 1
        f1 = zeros((delta.shape[0], nk * (nD + 1)))

        # cdid relates each observation to the market it is in
        cdid = arange(nmkt).repeat(nbrand)

        cdindex = arange(nbrand, nbrand * (nmkt + 1), nbrand) - 1

        # computing (partial share) / (partial sigma)
        for k in xrange(nk):
            xv = x2[:, k].reshape(-1, 1).dot(ones((1, nsimind)))
            xv *= v[cdid, nsimind * k:nsimind * (k + 1)]

            temp = (xv * ind_choice_prob).cumsum(axis=0)
            sum1 = temp[cdindex, :]

            sum1[1:, :] = sum1[1:, :] - sum1[0:-1, :]

            f1[:, k] = (ind_choice_prob * (xv - sum1[cdid, :])).mean(axis=1)

        for d in xrange(nD):
            tmpD = D[cdid, nsimind * d:nsimind * (d + 1)]

            temp1 = zeros((cdid.shape[0], nk))

            for k in xrange(nk):
                xd = x2[:, k].reshape(-1, 1).dot(ones((1, nsimind))) * tmpD

                temp = (xd * ind_choice_prob).cumsum(axis=0)
                sum1 = temp[cdindex, :]

                sum1[1:, :] = sum1[1:, :] - sum1[0:-1, :]

                temp1[:, k] = (ind_choice_prob * (xd-sum1[cdid, :])).mean(axis=1)

            f1[:, nk * (d + 1):nk * (d + 2)] = temp1

        rel = nonzero(theta.T.ravel())[0]

        f = zeros((cdid.shape[0], rel.shape[0]))

        n = 0

        for i in xrange(cdindex.shape[0]):
            temp = ind_choice_prob[n:cdindex[i] + 1, :]
            H1 = temp.dot(temp.T)
            H = (diag(temp.sum(axis=1)) - H1) / self.nsimind

            f[n:cdindex[i] + 1, :] = - solve(H, f1[n:cdindex[i] + 1, rel])

            n = cdindex[i] + 1

        return(f)

    def set_aliases(self):
        return(self._blp,
               self.theta,
               self.delta,
               self.v,
               self.D,
               self.x2,
               self.nmkt,
               self.nsimind,
               self.nbrand
               )

    def optimize(self, theta0, algorithm='simplex'):
        """optimize GMM objective function"""

        theta0_vec = theta0[nonzero(theta0)]

        starttime = time.time()

        full_output = True
        disp = True

        self.results = {}

        if algorithm == 'simplex':
            self.results['opt'] = optimize.fmin(func=self._GMM,
                                                x0=theta0_vec,
                                                maxiter=2000000,
                                                maxfun=2000000,
                                                full_output=full_output,
                                                disp=disp)

        elif algorithm == 'powell':
            self.results['opt'] = optimize.fmin_powell(func=self._GMM,
                                                       x0=theta0_vec,
                                                       maxiter=2000000,
                                                       full_output=full_output,
                                                       disp=disp)

        elif algorithm == 'bfgs':
            self.results['opt'] = optimize.fmin_bfgs(f=self._GMM,
                                                     x0=theta0_vec,
                                                     fprime=self._gradient,
                                                     full_output=full_output,
                                                     disp=disp)

        elif algorithm == 'cg':
            self.results['opt'] = optimize.fmin_cg(f=self._GMM,
                                                   x0=theta0_vec,
                                                   fprime=self._gradient,
                                                   full_output=full_output,
                                                   disp=disp)

        elif algorithm == 'ncg':
            self.results['opt'] = optimize.fmin_ncg(f=self._GMM,
                                                    x0=theta0_vec,
                                                    fprime=self._gradient,
                                                    full_output=full_output,
                                                    disp=disp)

        print('optimization: {0} seconds'.format(time.time() - starttime))

        self.results['varcov'] = self.cal_var_covar_mat(self.results['opt'][0])
