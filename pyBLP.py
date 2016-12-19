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

import time

import numpy as np
from numpy.linalg import cholesky, inv, solve
from scipy.linalg import cho_solve

import scipy.optimize as optimize

import pandas as pd

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
        X1 :
        X2 :
        Z :

    Attributes
    ----------
    x : float
        The X coordinate.

    Methods
    -------
    GMM(theta)
        GMM objective function.
    """

    def __init__(self, data):
        self.id = data.id
        self.s_jt = s_jt = data.s_jt
        self.ln_s_jt = np.log(self.s_jt)
        self.v = data.v
        self.D = data.D
        self.X1 = X1 = data.X1
        self.X2 = data.X2
        self.Z = Z = data.Z

        nmkt = self.nmkt = data.nmkt
        self.nbrand = data.nbrand
        self.nsimind = data.nsimind

        self.nX2 = self.X2.shape[1]
        self.nD = self.D.shape[1] // self.nsimind

        # choleskey root (lower triangular) of the weighting matrix, W = (Z'Z)^{-1}
        # do not invert it yet
        LinvW = self.LinvW = (cholesky(Z.T @ Z), True)

        # Z'x1
        Z_X1 = self.Z_X1 = Z.T @ X1

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

        # initialize δ_old
        self.δ_old = self.X1 @ (solve(Z_X1.T @ cho_solve(LinvW, Z_X1),
                                  Z_X1.T @ cho_solve(LinvW, Z.T @ y)))

        # initialize s
        self.s = np.zeros_like(self.δ_old)

        self.θ2 = None
        self.ix_θ2_T = None  # Transposed to be consistent with MATLAB

    def cal_δ(self, θ2):
        """Calculate δ (mean utility) via contraction mapping"""
        v, D, X2 = self.v, self.D, self.X2
        nmkt, nsimind, nbrand = self.nmkt, self.nsimind, self.nbrand

        s, δ, ln_s_jt = self.s, self.δ_old, self.ln_s_jt

        θ2_v, θ2_D = θ2[:, 0], θ2[:, 1:]

        niter = 0

        μ = _BLP.cal_mu(θ2_v, θ2_D, v, D, X2, nmkt, nsimind, nbrand)

        while True:
            exp_Xb = np.exp(δ.reshape(-1, 1) + μ)

            _BLP.cal_s(exp_Xb, nmkt, nsimind, nbrand, s)  # s gets updated

            diff = ln_s_jt - np.log(s)

            if np.isnan(diff).sum():
                raise Exception('nan in diffs')

            δ += diff

            if (abs(diff).max() < self.etol) and (abs(diff).mean() < 1e-3):
                break

            niter += 1

        print('contraction mapping finished in {} iterations'.format(niter))

        return δ

    def cal_θ1_and_ξ(self, δ):
        """Calculate δ (mean utility) via contraction mapping"""
        X1, Z, Z_X1, LinvW = self.X1, self.Z, self.Z_X1, self.LinvW
        
        # Z'δ
        Z_δ = Z.T @ δ

        #\[ \theta_1 = (\tilde{X}'ZW^{-1}Z'\tilde{X})^{-1}\tilde{X}'ZW^{-1}Z'\delta \]
        # θ1 from FOC
        θ1 = self.θ1_old = solve(Z_X1.T @ cho_solve(LinvW, Z_X1),
                                 Z_X1.T @ cho_solve(LinvW, Z_δ))

        ξ = self.ξ_old = δ - X1 @ θ1

        return θ1, ξ

    def GMM(self, θ2_cand):
        """GMM objective function"""
        if self.θ2 is None:
            if θ2_cand.ndim == 1:  # vectorized version
                raise Exception(
                    "Cannot pass θ2_vec before θ2 is initialized!")
            else:
                self.θ2 = θ2_cand.copy()

        if self.ix_θ2_T is None:
            self.ix_θ2_T = np.nonzero(self.θ2.T)

        if θ2_cand.ndim == 1:  # vectorized version
            self.θ2.T[self.ix_θ2_T] = θ2_cand
        else:
            self.θ2[:] = θ2_cand

        θ2, Z, X1, Z_X1, LinvW = self.θ2, self.Z, self.X1, self.Z_X1, self.LinvW

        θ2_v, θ2_D = θ2[:, 0], θ2[:, 1:]

        # adaptive etol
        if self.GMM_diff < 1e-6:
            etol = self.etol = 1e-13
        elif self.GMM_diff < 1e-3:
            etol = self.etol = 1e-12
        else:
            etol = self.etol = 1e-9

        # update δ
        δ = self.cal_δ(θ2)

        if np.isnan(δ).sum():
            return 1e+10

        θ1, ξ = self.cal_θ1_and_ξ(δ)

        # Z'ξ
        Z_ξ = Z.T @ ξ

        # \[ (\delta - \tilde{X}\theta_1)'ZW^{-1}Z'(\delta-\tilde{X}\theta_1) \]
        GMM = Z_ξ.T @ cho_solve(LinvW, Z_ξ)

        self.GMM_diff = abs(self.GMM_old - GMM)
        self.GMM_old = GMM

        print('GMM value: {}'.format(GMM))
        return GMM

    def gradient_GMM(self, θ2_cand):
        """Return gradient of GMM objective function

        Parameters
        ----------
        θ2_cand : array
            Description of parameter `θ2`.

        Returns
        -------
        gradient : array
            String representation of the array.

        """
        θ2, ix_θ2_T, Z, LinvW = self.θ2, self.ix_θ2_T, self.Z, self.LinvW

        if θ2_cand.ndim == 1:  # vectorized version
            θ2.T[ix_θ2_T] = θ2_cand
        else:
            θ2[:] = θ2_cand

        # update δ
        δ = self.cal_δ(θ2)

        jacob = self.cal_jacobian(θ2, δ)

        θ1, ξ = self.cal_θ1_and_ξ(δ)

        return 2 * jacob.T @ Z @ cho_solve(LinvW, Z.T) @ ξ

    def cal_varcov(self, θ2_vec):
        """calculate variance covariance matrix"""
        θ2, ix_θ2_T, Z, LinvW, X1 = self.θ2, self.ix_θ2_T, self.Z, self.LinvW, self.X1

        θ2.T[ix_θ2_T] = θ2_vec

        # update δ
        δ = self.cal_δ(θ2)

        jacob = self.cal_jacobian(θ2, δ)

        θ1, ξ = self.cal_θ1_and_ξ(δ)

        Zres = Z * ξ.reshape(-1, 1)
        Ω = Zres.T @ Zres  # covariance of the momconds

        G = (np.c_[X1, jacob].T @ Z).T  # gradient of the momconds

        WG = cho_solve(LinvW, G)
        WΩ = cho_solve(LinvW, Ω)

        tmp = solve(G.T @ WG, G.T @ WΩ @ WG).T  # G'WΩWG(G'WG)^(-1) part

        varcov = solve((G.T @ WG), tmp)

        return varcov

    def cal_se(self, varcov):
        se_all = np.sqrt(varcov.diagonal())

        se = np.zeros_like(self.θ2)
        se.T[self.ix_θ2_T] = se_all[-self.ix_θ2_T[0].shape[0]:]  # to be consistent with MATLAB

        return se

    def cal_jacobian(self, θ2, δ):
        """calculate the Jacobian with the current value of δ"""
        v, D, X2 = self.v, self.D, self.X2
        nmkt, nsimind, nbrand = self.nmkt, self.nsimind, self.nbrand

        μ = _BLP.cal_mu(
                 θ2[:, 0], θ2[:, 1:], v, D, X2, nmkt, nsimind, nbrand)

        exp_Xb = np.exp(δ.reshape(-1, 1) + μ)

        ind_choice_prob = _BLP.cal_ind_choice_prob(
                              exp_Xb, nmkt, nsimind, nbrand)

        nk = X2.shape[1]
        nD = θ2.shape[1] - 1
        f1 = np.zeros((δ.shape[0], nk * (nD + 1)))

        # cdid relates each observation to the market it is in
        cdid = np.arange(nmkt).repeat(nbrand)

        cdindex = np.arange(nbrand, nbrand * (nmkt + 1), nbrand) - 1

        # compute (partial share) / (partial sigma)
        for k in range(nk):
            xv = X2[:, k].reshape(-1, 1) @ np.ones((1, nsimind))
            xv *= v[cdid, nsimind * k:nsimind * (k + 1)]

            temp = (xv * ind_choice_prob).cumsum(axis=0)
            sum1 = temp[cdindex, :]

            sum1[1:, :] = sum1[1:, :] - sum1[:-1, :]

            f1[:, k] = (ind_choice_prob * (xv - sum1[cdid, :])).mean(axis=1)

        # If no demogr comment out the next part
        # computing (partial share)/(partial pi)
        for d in range(nD):
            tmpD = D[cdid, nsimind * d:nsimind * (d + 1)]

            temp1 = np.zeros((cdid.shape[0], nk))

            for k in range(nk):
                xd = X2[:, k].reshape(-1, 1) @ np.ones((1, nsimind)) * tmpD

                temp = (xd * ind_choice_prob).cumsum(axis=0)
                sum1 = temp[cdindex, :]

                sum1[1:, :] = sum1[1:, :] - sum1[0:-1, :]

                temp1[:, k] = (ind_choice_prob * (xd-sum1[cdid, :])).mean(axis=1)

            f1[:, nk * (d + 1):nk * (d + 2)] = temp1

        # computing (partial delta)/(partial theta2)
        rel = np.nonzero(θ2.T.ravel())[0]
        jacob = np.zeros((cdid.shape[0], rel.shape[0]))
        n = 0

        for i in range(cdindex.shape[0]):
            temp = ind_choice_prob[n:cdindex[i] + 1, :]
            H1 = temp @ temp.T
            H = (np.diag(temp.sum(axis=1)) - H1) / self.nsimind

            jacob[n:cdindex[i] + 1, :] = - solve(H, f1[n:cdindex[i] + 1, rel])

            n = cdindex[i] + 1

        return jacob

    def minimize_GMM(
            self, results, θ20, method='Nelder-Mead', maxiter=2000000, disp=True):
        """minimize GMM objective function"""

        self.θ2 = θ20.copy()
        θ20_vec = θ20.T[np.nonzero(θ20.T)]

        options = {'maxiter': maxiter,
                   'disp': disp,
                   }

        results['θ2'] = optimize.minimize(
            fun=self.GMM, x0=θ20_vec, jac=self.gradient_GMM,
            method=method, options=options)

        varcov = self.cal_varcov(results['θ2']['x'])
        results['varcov'] = varcov
        results['θ2']['se'] = self.cal_se(varcov)

    def estimate_param_means(self, results):
        """Estimate mean of the parameters with minimum-distance procedure

        In the current example (Nevo 2000), skip the first variable (price)
        which is included in the both X1 and X2
        """
        X2 = self.X2
        nbrand = self.nbrand

        self.θ2.T[self.ix_θ2_T] = results['θ2']['x']
        θ2 = self.θ2
        varcov = results['varcov']

        δ = self.cal_δ(θ2)
        θ1, ξ = self.cal_θ1_and_ξ(δ)

        V = varcov[1:θ1.shape[0], 1:θ1.shape[0]]
        y = θ1[1:]  # estimated brand (product) dummies. Skip the first element (price)
        X = X2[:nbrand, [0, 2, 3]]

        L = X.T @ solve(V, X)  # X'V^{-1}X
        R = X.T @ solve(V, y)  # X'V^{-1}y

        results['β'] = {}
        β = results['β']['β'] = solve(L, R)  # (X'V^{-1}X)^{-1} X'V^{-1}y
        β_se = results['β']['se'] = np.sqrt(inv(L).diagonal())

        r = y - X @ β
        y_demeaned = y - y.mean()
        r_demeaned = r - r.mean()
        
        Rsq = 1 - (r_demeaned @ r_demeaned) / (y_demeaned @ y_demeaned)
        results['β']['Rsq'] = Rsq

        Rsq_G = 1 - (r @ solve(V, r)) / (y_demeaned @ solve(V, y_demeaned))
        results['β']['Rsq_G'] = Rsq_G

        Chisq = results['β']['Chisq'] = len(self.id) * r @ solve(V, r)

    def estimate(
            self, θ20, method='BFGS', maxiter=2000000, disp=True):

        self.GMM(θ20)

        results = self.results = {}

        starttime = time.time()

        self.minimize_GMM(
            results, θ20=θ20, method=method, maxiter=maxiter, disp=disp)

        results['GMM'] = results['θ2']['fun']

        self.estimate_param_means(results)

        X_names = ['Constant', 'Price', 'Sugar', 'Mushy']

        index = []
        for var in X_names:
            index.append(var)
            index.append('')
            
        D_names = ['Income', 'Income^2', 'Age', 'Child']

        table_results = pd.DataFrame(
                            data=np.zeros((self.X2.shape[1] * 2, 2 + self.nD)),
                            index=index,
                            columns=['Mean', 'SD'] + D_names,
        )

        self.table_results = table_results

        θ2 = np.zeros_like(self.θ2)
        θ2.T[self.ix_θ2_T] = results['θ2']['x']

        table_results.values[::2, 1:] = θ2
        table_results.values[1::2, 1:] = results['θ2']['se']

        δ = self.cal_δ(θ2)
        θ1, ξ = self.cal_θ1_and_ξ(δ)

        β = np.zeros((θ2.shape[0], ))
        se_β = np.zeros((θ2.shape[0], ))

        β[0] = results['β']['β'][0]
        β[1] = θ1[0]
        β[2:] = results['β']['β'][1:]

        se_β[0] = results['β']['se'][0]
        se_β[1] = np.sqrt(results['varcov'][0, 0])
        se_β[2:] = results['β']['se'][1:]

        table_results.values[::2, 0] = β
        table_results.values[1::2, 0] = se_β

        print(table_results)

        print('GMM objective: {}'.format(results['GMM']))
        print('Min-Dist R-squared: {}'.format(results['β']['Rsq']))
        print('Min-Dist weighted R-squared: {}'.format(results['β']['Rsq_G']))
        print('run time: {} (minutes)'.format((time.time() - starttime) / 60))

