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

import time

import numpy as np
from numpy.linalg import solve, cholesky
from scipy.linalg import cho_solve

import scipy.optimize as optimize

import _BLP


class BLP:
    """BLP Class

    Random coefficient logit model

    Parameters
    ----------
    data : object
        Object containing data for estimation. It should contain:

        v :
        D :
        x1 :
        x2 :
        Z :

    Attributes
    ----------
    x : float
        The X coordinate.

    Methods
    -------
    init_GMM(theta, cython=True)
        Initialize GMM.
    GMM(theta)
        GMM objective function.
    """

    def __init__(self, data, cython=False):
        self.s_jt = s_jt = data.s_jt
        self.ln_s_jt = np.log(self.s_jt)
        self.v = data.v
        self.D = data.D
        self.x1 = x1 = data.x1
        self.x2 = data.x2
        self.Z = Z = data.Z

        nmkt = self.nmkt = data.nmkt
        self.nbrand = data.nbrand
        self.nsimind = data.nsimind

        self.nx2 = self.x2.shape[1]
        self.nD = self.D.shape[1] // self.nsimind

        # choleskey root (lower triangular) of the weighting matrix, W = (Z'Z)^{-1}
        # do not invert it yet
        LinvW = self.LinvW = (cholesky(Z.T @ Z), True)

        # Z'x1
        Z_x1 = self.Z_x1 = Z.T @ x1

        self.etol = 1e-6
        self.iter_limit = 200

        self.GMM_old = 0
        self.GMM_diff = 1

        # calculate market share
        # outside good
        s_0t = self.s_0t = (1 - self.s_jt.reshape(nmkt, -1).sum(axis=1))

        y = self.y = np.log(s_jt.reshape(nmkt, -1))
        y -= np.log(s_0t.reshape(-1, 1))
        y.shape = (-1, )

        # initialize δ 
        self.δ = self.x1 @ (solve(Z_x1.T @ cho_solve(LinvW, Z_x1),
                                  Z_x1.T @ cho_solve(LinvW, Z.T @ y)))


        # initialize s
        self.s = np.zeros_like(self.δ)

        self.θ = None
        self.ix_θ = None
        self.cython = cython

    def cal_δ(self, θ):
        """Calculate δ (mean utility) via contraction mapping"""
        v, D, x2, nmkt, nsimind, nbrand = self.set_aliases()

        s, δ = self.s, self.δ

        θ_v = θ[:, 0]
        θ_D = θ[:, 1:]

        niter = 0

        exp_μ = np.exp(_BLP.cal_mu(
                    θ_v, θ_D, v, D, x2, nmkt, nsimind, nbrand))

        while True:
            diff = self.ln_s_jt.copy()

            exp_Xb = np.exp(δ.reshape(-1, 1)) * exp_μ

            _BLP.cal_s(exp_Xb, nmkt, nsimind, nbrand, s)

            diff -= np.log(s)

            if np.isnan(diff).sum():
                print('nan in diffs')
                break

            δ += diff

            if (abs(diff).max() < self.etol) and (abs(diff).mean() < 1e-3):
                break

            niter += 1

        print('contraction mapping finished in {} iterations'.format(niter))

        return δ

    def GMM(self, θ_cand):
        """GMM objective function"""
        v, D, x2, nmkt, nsimind, nbrand = self.set_aliases()

        if self.θ is None:
            if θ_cand.ndim == 1:  # vectorized version
                raise Exception(
                    "Cannot pass θ_vec before θ is initialized!")
            else:
                self.θ = θ_cand

        θ = self.θ

        if self.ix_θ is None:
            self.ix_θ = np.nonzero(θ)

        if θ_cand.ndim == 1:  # vectorized version
            θ[self.ix_θ] = θ_cand
        else:
            θ[:] = θ_cand

        θ_v = θ[:, 0]
        θ_D = θ[:, 1:]

        # adaptive etol
        if self.GMM_diff < 1e-6:
            etol = self.etol = 1e-13
        elif self.GMM_diff < 1e-3:
            etol = self.etol = 1e-12
        else:
            etol = self.etol = 1e-9

        if self.cython:
            _BLP.cal_delta(
                delta,
                θ_v, θ_D,
                self.ln_s_jt,
                v, D, x2, nmkt, nsimind, nbrand,
                etol,
                self.iter_limit
                )
        else:
            δ = self.δ = self.cal_δ(θ)

        if np.isnan(δ).sum():
            return 1e+10

        Z_x1 = self.Z_x1
        LinvW = self.LinvW

        # Z'δ
        Z_δ = self.Z.T @ δ

        #\[ \theta_1 = (\tilde{X}'ZW^{-1}Z'\tilde{X})^{-1}\tilde{X}'ZW^{-1}Z'\delta \]
        θ1 = solve(Z_x1.T @ cho_solve(LinvW, Z_x1),
                   Z_x1.T @ cho_solve(LinvW, Z_δ))

        self.θ1 = θ1

        ξ = self.ξ = δ - self.x1 @ θ1

        # Z'ξ
        Z_ξ = self.Z.T @ ξ

        # \[ (\delta - \tilde{X}\theta_1)'ZW^{-1}Z'(\delta-\tilde{X}\theta_1) \]
        GMM = Z_ξ.T @ cho_solve(LinvW, Z_ξ)

        self.GMM_diff = abs(self.GMM_old - GMM)
        self.GMM_old = GMM

        print('GMM value: {}'.format(GMM))
        return GMM

    def gradient_GMM(self, θ_cand):
        """Return gradient of GMM objective function

        Parameters
        ----------
        θ : array
            Description of parameter `θ`.

        Returns
        -------
        gradient : array
            String representation of the array.

        """
        θ, ix_θ, ξ, Z, LinvW = self.θ, self.ix_θ, self.ξ, self.Z, self.LinvW

        if θ_cand.ndim == 1:  # vectorized version
            θ[ix_θ] = θ_cand
        else:
            θ[:] = θ_cand

        jacob = self.cal_jacobian(θ)

        return 2 * jacob.T @ Z @ cho_solve(LinvW, Z.T) @ ξ

    def cal_varcov(self, θ_vec):
        """calculate variance covariance matrix"""
        θ, ix_θ, ξ, Z, LinvW = self.θ, self.ix_θ, self.ξ, self.Z, self.LinvW

        θ[ix_θ] = θ_vec

        Zres = Z * ξ.reshape(-1, 1)
        Ω = Zres.T @ Zres  # covariance of the momconds

        jacob = self.cal_jacobian(θ)

        G = (np.c_[self.x1, jacob].T @ Z).T  # gradient of the momconds

        WG = cho_solve(LinvW, G)
        WΩ = cho_solve(LinvW, Ω)

        tmp = solve(G.T @ WG, G.T @ WΩ @ WG).T  # G'WΩWG(G'WG)^(-1) part

        varcov = solve((G.T @ WG), tmp)

        return varcov

    def cal_se(self, varcov):
        se_all = np.sqrt(varcov.diagonal())

        se = np.zeros_like(self.θ)
        se[self.ix_θ] = se_all[-self.ix_θ[0].shape[0]:]

        return se

    def cal_jacobian(self, θ):
        """calculate the Jacobian with the current value of δ"""

        v, D, x2, nmkt, nsimind, nbrand = self.set_aliases()

        δ = self.δ

        μ = _BLP.cal_mu(
                 θ[:, 0], θ[:, 1:], v, D, x2, nmkt, nsimind, nbrand)

        exp_Xb = np.exp(δ.reshape(-1, 1) + μ)

        ind_choice_prob = _BLP.cal_ind_choice_prob(
                              exp_Xb, nmkt, nsimind, nbrand)

        nk = self.x2.shape[1]
        nD = θ.shape[1] - 1
        f1 = np.zeros((δ.shape[0], nk * (nD + 1)))

        # cdid relates each observation to the market it is in
        cdid = np.arange(nmkt).repeat(nbrand)

        cdindex = np.arange(nbrand, nbrand * (nmkt + 1), nbrand) - 1

        # compute (partial share) / (partial sigma)
        for k in range(nk):
            xv = x2[:, k].reshape(-1, 1) @ np.ones((1, nsimind))
            xv *= v[cdid, nsimind * k:nsimind * (k + 1)]

            temp = (xv * ind_choice_prob).cumsum(axis=0)
            sum1 = temp[cdindex, :]

            sum1[1:, :] = sum1[1:, :] - sum1[0:-1, :]

            f1[:, k] = (ind_choice_prob * (xv - sum1[cdid, :])).mean(axis=1)

        for d in range(nD):
            tmpD = D[cdid, nsimind * d:nsimind * (d + 1)]

            temp1 = np.zeros((cdid.shape[0], nk))

            for k in range(nk):
                xd = x2[:, k].reshape(-1, 1) @ np.ones((1, nsimind)) * tmpD

                temp = (xd * ind_choice_prob).cumsum(axis=0)
                sum1 = temp[cdindex, :]

                sum1[1:, :] = sum1[1:, :] - sum1[0:-1, :]

                temp1[:, k] = (ind_choice_prob * (xd-sum1[cdid, :])).mean(axis=1)

            f1[:, nk * (d + 1):nk * (d + 2)] = temp1

        rel = np.nonzero(θ.T.ravel())[0]

        f = np.zeros((cdid.shape[0], rel.shape[0]))

        n = 0

        for i in range(cdindex.shape[0]):
            temp = ind_choice_prob[n:cdindex[i] + 1, :]
            H1 = temp @ temp.T
            H = (np.diag(temp.sum(axis=1)) - H1) / self.nsimind

            f[n:cdindex[i] + 1, :] = - solve(H, f1[n:cdindex[i] + 1, rel])

            n = cdindex[i] + 1

        return f

    def set_aliases(self):
        return(
            self.v,
            self.D,
            self.x2,
            self.nmkt,
            self.nsimind,
            self.nbrand
            )

    def optimize(self, theta0, method='Nelder-Mead', disp=True, full_output=True):
        """optimize GMM objective function"""

        self.theta = theta0
        theta0_vec = theta0[np.nonzero(theta0)]

        starttime = time.time()

        self.results = {}

        options = {'maxiter': 2000000,
                   'maxfun': 2000000,
                   'disp': disp,
                   'full_output': full_output}

        self.results['opt'] = optimize.minimize(
            fun=self.GMM, x0=theta0_vec, method=method, options=options)

        print('optimization: {0} seconds'.format(time.time() - starttime))

        varcov = self.cal_varcov(self.results['opt'][0])
        self.results['varcov'] = varcov
        self.results['se'] = self.cal_se(varcov)

