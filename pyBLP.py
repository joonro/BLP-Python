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
    """Random coefficient logit model 

    Parameters
    ----------
    data : object (optional)
        Object containing data for estimation. It should contain all other variables 
        below. If not None, other variables should not be used.

    s_jt: xarray.DataArray
        Market share of each brand in each market. (=nmkts= by =nbrands=) dimension.
        Market share of the outside good will be automatically calculated.

    X1 : xarray.DataArray
        The variables that enter the linear part of the estimation with 
        (nmkts by nbrands by nvars) dimension.

    X2 : xarray.DataArray 
        The variables that enter the nonlinear part of the estimation with 
        (nmkts by nbrands by nvars) dimension.

    Z : xarray.DataArray
        Instruments with (nmkts by nbrands by nvars) dimension.

    v : xarray.DataArray
        Random draws given for the estimation with (nmkts by nsiminds by nvars) dimension.

    D : xarray.DataArray
        Demeaned draws of demographic variables with (nmkts by nsiminds by nvars) dimension.

    Attributes
    ----------
    results : dictionary
        Results of GMM estimation

    Methods
    -------
    GMM(θ2_cand)
        GMM objective function.

    minimize_GMM(results, θ20, method='BFGS', maxiter=2000000, disp=True)
        Minimize GMM objective function.

    estimate(θ20, method='BFGS', maxiter=2000000, disp=True)
        Run full estimation.
    """

    def __init__(self, data=None, s_jt=None, X1=None, X2=None, Z=None, v=None, D=None):
        if data is not None:
            for var in [s_jt, X1, X2, Z, v, D]:
                assert var is None, "When data is passed, individual variables should not be used"
            
            s_jt = data.s_jt
            v = data.v
            D = data.D
            X1_nd = data.X1
            X2 = data.X2
            Z_nd = data.Z

        else:
            assert data is None, "When individual variables are passed, data should not be used"
            
            X1_nd = X1.copy()
            Z_nd = Z.copy()
            
        self.s_jt = s_jt
        self.ln_s_jt = np.log(self.s_jt.values)

        self.X1_nd = X1_nd
        # vectorized version
        self.X1 = X1 = X1_nd.values.reshape(-1, X1_nd.shape[-1])

        self.X2 = X2

        self.Z_nd = Z_nd
        # vectorized version
        self.Z = Z = Z_nd.values.reshape(-1, Z_nd.shape[-1])

        self.v = v
        self.D = D

        nmkts = self.nmkts = len(X1_nd.coords['markets'])
        nbrands = self.nbrands = len(X1_nd.coords['brands'])
        nsiminds = self.nsiminds = len(v.coords['nsiminds'])

        self.nX2 = len(X2.coords['vars'])
        self.nD = len(D.coords['vars'])

        # LinvW: choleskey root (lower triangular) of the inverse of the
        # weighting matrix, W. (W = (Z'Z)^{-1}).
        LinvW = self.LinvW = (cholesky(Z.T @ Z), True)

        # Z'X1
        Z_X1 = self.Z_X1 = Z.T @ X1

        # calculate market share
        # outside good
        s_0t = self.s_0t = (1 - self.s_jt.sum(dim='brands'))

        y = self.y = np.log(s_jt)
        y -= np.log(s_0t)
        y = y.values.reshape(-1, )

        # initialize δ
        self.δ = X1 @ (solve(Z_X1.T @ cho_solve(LinvW, Z_X1),
                             Z_X1.T @ cho_solve(LinvW, Z.T @ y)))

        self.δ.shape = (nmkts, nbrands)

        # initialize s
        self.s = np.zeros_like(self.δ)
        self.ind_choice_prob = np.zeros((nmkts, nsiminds, nbrands))

        self.θ2 = None
        self.ix_θ2_T = None  # Transposed to be consistent with MATLAB

    def _cal_mu(self, θ2):
        """Calculate individual-specific utility

        Same speed as the single-thread Cython function (_BLP.cal_mu()),
        but slower than parallelized Cython module  

        Mainly used for unit testing
        """
        v, D, X2 = self.v, self.D, self.X2

        Π = θ2[:, 1:]
        Σ = np.diag(θ2[:, 0])  # off-diagonals of Σ are zero

        # these are nmkts by nsiminds by nvars arrays 
        ΠD = (Π @ D.values.transpose([0, 2, 1])).transpose([0, 2, 1])
        Σv = (Σ @ v.values.transpose([0, 2, 1])).transpose([0, 2, 1])   

        # nmkts by nsiminds by nbrands
        μ = (X2.values @ (ΠD + Σv).transpose(0, 2, 1)).transpose([0, 2, 1])

        return μ

    def _cal_δ(self, θ2):
        """Calculate δ (mean utility) via contraction mapping"""
        v, D, X2 = self.v, self.D, self.X2
        nmkts, nsiminds, nbrands = self.nmkts, self.nsiminds, self.nbrands

        δ, ln_s_jt = self.δ, self.ln_s_jt  # initial values

        niter = 0

        ε = 1e-13  # tight tolerance

        μ = self.μ = _BLP.cal_mu(θ2, v.values, D.values, X2.values)

        while True:
            s = self._cal_s(δ, μ)
            #_BLP.cal_s(δ, μ, s)  # s gets updated

            diff = ln_s_jt - np.log(s)

            if np.isnan(diff).sum():
                raise Exception('nan in diffs')

            δ += diff

            if (abs(diff).max() < ε) and (abs(diff).mean() < 1e-3):
                break

            niter += 1

        print('contraction mapping finished in {} iterations'.format(niter))

        return δ

    def _cal_s(self, δ, μ):
        """Calculate market share

        Calculates individual choice probability first, then take the weighted
        sum

        """
        nsiminds = self.nsiminds
        ind_choice_prob = self.ind_choice_prob 

        _BLP.cal_ind_choice_prob(δ, μ, ind_choice_prob)
        s = ind_choice_prob.sum(axis=1) / nsiminds

        return s

    def _cal_θ1_and_ξ(self, δ):
        """Calculate θ1 and ξ with F.O.C"""
        X1, Z, Z_X1, LinvW = self.X1, self.Z, self.Z_X1, self.LinvW
        
        # Z'δ
        Z_δ = Z.T @ δ.flatten()

        #\[ \theta_1 = (\tilde{X}'ZW^{-1}Z'\tilde{X})^{-1}\tilde{X}'ZW^{-1}Z'\delta \]
        # θ1 from FOC
        θ1 = self.θ1 = solve(Z_X1.T @ cho_solve(LinvW, Z_X1),
                             Z_X1.T @ cho_solve(LinvW, Z_δ))

        ξ = self.ξ = δ.flatten() - X1 @ θ1

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

        # update δ
        δ = self._cal_δ(θ2)

        if np.isnan(δ).sum():
            return 1e+10

        θ1, ξ = self._cal_θ1_and_ξ(δ)

        # Z'ξ = (\delta - \tilde{X}\theta_1)
        Z_ξ = Z.T @ ξ

        # \[ (\delta - \tilde{X}\theta_1)'ZW^{-1}Z'(\delta-\tilde{X}\theta_1) \]
        GMM = Z_ξ.T @ cho_solve(LinvW, Z_ξ)

        print('GMM value: {}'.format(GMM))
        return GMM

    def _gradient_GMM(self, θ2_cand):
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
        δ = self._cal_δ(θ2)

        θ1, ξ = self._cal_θ1_and_ξ(δ)

        jacob = self._cal_jacobian(θ2, δ)

        return 2 * jacob.T @ Z @ cho_solve(LinvW, Z.T) @ ξ

    def _cal_varcov(self, θ2_vec):
        """calculate variance covariance matrix"""
        θ2, ix_θ2_T, Z, LinvW, X1 = self.θ2, self.ix_θ2_T, self.Z, self.LinvW, self.X1

        θ2.T[ix_θ2_T] = θ2_vec

        # update δ
        δ = self._cal_δ(θ2)

        jacob = self._cal_jacobian(θ2, δ)

        θ1, ξ = self._cal_θ1_and_ξ(δ)

        Zres = Z * ξ.reshape(-1, 1)
        Ω = Zres.T @ Zres  # covariance of the momconds

        G = (np.c_[X1, jacob].T @ Z).T  # gradient of the momconds

        WG = cho_solve(LinvW, G)
        WΩ = cho_solve(LinvW, Ω)

        tmp = solve(G.T @ WG, G.T @ WΩ @ WG).T  # G'WΩWG(G'WG)^(-1) part

        varcov = solve((G.T @ WG), tmp)

        return varcov

    def _cal_se(self, varcov):
        se_all = np.sqrt(varcov.diagonal())

        se = np.zeros_like(self.θ2)
        se.T[self.ix_θ2_T] = se_all[-self.ix_θ2_T[0].shape[0]:]  # to be consistent with MATLAB

        return se

    def _cal_jacobian(self, θ2, δ):
        """calculate the Jacobian with the current value of δ"""
        v, D, X2 = self.v, self.D, self.X2
        nmkts, nsiminds, nbrands = self.nmkts, self.nsiminds, self.nbrands

        ind_choice_prob = self.ind_choice_prob 

        μ = _BLP.cal_mu(θ2, v.values, D.values, X2.values)

        _BLP.cal_ind_choice_prob(δ, μ, ind_choice_prob)
        ind_choice_prob_vec = ind_choice_prob.transpose(0, 2, 1).reshape(-1, nsiminds)

        nk = len(X2.coords['vars'])
        nD = len(D.coords['vars'])
        f1 = np.zeros((δ.flatten().shape[0], nk * (nD + 1)))

        # cdid relates each observation to the market it is in
        cdid = np.arange(nmkts).repeat(nbrands)

        cdindex = np.arange(nbrands, nbrands * (nmkts + 1), nbrands) - 1

        # compute ∂share/∂σ
        for k in range(nk):
            X2v = X2[..., k].values.reshape(-1, 1) @ np.ones((1, nsiminds))
            X2v *= v[cdid, :, k].values

            temp = (X2v * ind_choice_prob_vec).cumsum(axis=0)
            sum1 = temp[cdindex, :]

            sum1[1:, :] = sum1[1:, :] - sum1[:-1, :]

            f1[:, k] = (ind_choice_prob_vec * (X2v - sum1[cdid, :])).mean(axis=1)

        # compute ∂share/∂pi
        for d in range(nD):
            tmpD = D[cdid, :, d].values

            temp1 = np.zeros((cdid.shape[0], nk))

            for k in range(nk):
                X2d = X2[..., k].values.reshape(-1, 1) @ np.ones((1, nsiminds)) * tmpD

                temp = (X2d * ind_choice_prob_vec).cumsum(axis=0)
                sum1 = temp[cdindex, :]

                sum1[1:, :] = sum1[1:, :] - sum1[:-1, :]

                temp1[:, k] = (ind_choice_prob_vec * (X2d - sum1[cdid, :])).mean(axis=1)

            f1[:, nk * (d + 1):nk * (d + 2)] = temp1

        # compute ∂δ/∂θ2
        rel = np.nonzero(θ2.T.ravel())[0]
        jacob = np.zeros((cdid.shape[0], rel.shape[0]))
        n = 0

        for i in range(cdindex.shape[0]):
            temp = ind_choice_prob_vec[n:cdindex[i] + 1, :]
            H1 = temp @ temp.T
            H = (np.diag(temp.sum(axis=1)) - H1) / nsiminds

            jacob[n:cdindex[i] + 1, :] = - solve(H, f1[n:cdindex[i] + 1, rel])

            n = cdindex[i] + 1

        return jacob

    def minimize_GMM(
            self, results, θ20, method='BFGS', maxiter=2000000, disp=True):
        """minimize GMM objective function"""

        self.θ2 = θ20.copy()
        θ20_vec = θ20.T[np.nonzero(θ20.T)]

        options = {'maxiter': maxiter,
                   'disp': disp,
                   }

        results['θ2'] = optimize.minimize(
            fun=self.GMM, x0=θ20_vec, jac=self._gradient_GMM,
            method=method, options=options)

        varcov = self._cal_varcov(results['θ2']['x'])
        results['varcov'] = varcov
        results['θ2']['se'] = self._cal_se(varcov)

    def _estimate_param_means(self, results):
        """Estimate mean of the parameters with minimum-distance procedure

        In the current example (Nevo 2000), skip the first variable (price)
        which is included in the both X1 and X2
        """
        X1_nd, X2 = self.X1_nd, self.X2
        nbrands = self.nbrands

        kX1 = len(X1_nd.coords['vars'])

        self.θ2.T[self.ix_θ2_T] = results['θ2']['x']
        θ2 = self.θ2
        varcov = results['varcov']

        δ = self._cal_δ(θ2)
        θ1, ξ = self._cal_θ1_and_ξ(δ)

        """Exclude variables present in both X1 and X2"""
        bool_ix_varcov = np.ones_like(varcov, dtype=bool)

        bool_ix_varcov[kX1:, :] = False
        bool_ix_varcov[:, kX1:] = False

        count = 0
        iix_include = []
        for iix, var in enumerate(X1_nd.coords['vars'].values):
            if var in X2.coords['vars'].values:
                bool_ix_varcov[iix, :] = False
                bool_ix_varcov[:, iix] = False
            else:
                iix_include.append(iix)
                count += 1

        V = varcov[bool_ix_varcov].reshape(count, count)
        y = θ1[iix_include]  # estimated brand (product) dummies

        iix_exclude_X2 = []
        iix_include_X2 = []
        for iix, var in enumerate(X2.coords['vars'].values):
            if var in X1_nd.coords['vars'].values:
                iix_exclude_X2.append(iix)
            else:
                iix_include_X2.append(iix)

        # these are the same across markets
        X = X2[0, :, iix_include_X2].values

        L = X.T @ solve(V, X)  # X'V^{-1}X
        R = X.T @ solve(V, y)  # X'V^{-1}y

        β = solve(L, R)  # (X'V^{-1}X)^{-1} X'V^{-1}y
        β_se = np.sqrt(inv(L).diagonal())

        results['β'] = {}
        results['β']['β'] = np.zeros((len(X2.coords['vars']), ))
        results['β']['se'] = np.zeros((len(X2.coords['vars']), ))

        kX2 = len(X2.coords['vars'])

        iix_θ1 = 0
        for iix in range(kX2):
            if iix in iix_include_X2:
                results['β']['β'][iix] = β[iix - iix_θ1]
                results['β']['se'][iix] = β_se[iix - iix_θ1]
            else:
                results['β']['β'][iix] = θ1[iix_θ1]
                results['β']['se'][iix] = np.sqrt(varcov[iix_θ1, iix_θ1])
                iix_θ1 += 1

        r = y - X @ β
        y_demeaned = y - y.mean()
        r_demeaned = r - r.mean()
        
        Rsq = 1 - (r_demeaned @ r_demeaned) / (y_demeaned @ y_demeaned)
        results['β']['Rsq'] = Rsq

        Rsq_G = 1 - (r @ solve(V, r)) / (y_demeaned @ solve(V, y_demeaned))
        results['β']['Rsq_G'] = Rsq_G

        Chisq = results['β']['Chisq'] = self.nmkts * r @ solve(V, r)

    def estimate(
            self, θ20, method='BFGS', maxiter=2000000, disp=True):
        """ Run the full estimation
        """

        self.GMM(θ20)

        results = self.results = {}

        starttime = time.time()

        self.minimize_GMM(
            results, θ20=θ20, method=method, maxiter=maxiter, disp=disp)

        results['GMM'] = results['θ2']['fun']

        self._estimate_param_means(results)

        X2, D = self.X2, self.D

        index = []
        for var in X2.coords['vars'].values:
            index.append(var)
            index.append('')
            
        table_results = pd.DataFrame(
            data=np.zeros((len(X2.coords['vars']) * 2, 2 + self.nD)),
            index=index,
            columns=['Mean', 'SD'] + list(D.coords['vars'].values),
        )

        self.table_results = table_results

        θ2 = np.zeros_like(self.θ2)
        θ2.T[self.ix_θ2_T] = results['θ2']['x']
        δ = self._cal_δ(θ2)
        θ1, ξ = self._cal_θ1_and_ξ(δ)

        table_results.values[::2, 1:] = θ2
        table_results.values[1::2, 1:] = results['θ2']['se']

        β = results['β']['β']
        se_β = results['β']['se']

        table_results.values[::2, 0] = β
        table_results.values[1::2, 0] = se_β

        print(table_results)

        print('GMM objective: {}'.format(results['GMM']))
        print('Min-Dist R-squared: {}'.format(results['β']['Rsq']))
        print('Min-Dist weighted R-squared: {}'.format(results['β']['Rsq_G']))
        print('run time: {} (minutes)'.format((time.time() - starttime) / 60))
