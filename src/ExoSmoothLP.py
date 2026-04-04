import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.interpolate as interp
import statsmodels.api as sm
from pyGAM import LinearGAM, f
import matplotlib.pyplot as plt
import seaborn as sns

class ExoSmoothLP:

    def __init__(self, Y: np.ndarray, x: np.ndarray, p: int, H: int):
        self.Y = Y
        self.x = x
        self.p = p
        self.H = H
        self.T, self.k = Y.shape
        self.beta = None
        self.irf = None

    def _build_lag_matrix(self):
        # helper function to build lag matrix W
        T, k = self.Y.shape
        out = np.full((T, k * self.p), np.nan)
        for lag in range(1, self.p + 1):
            out[lag:, (lag - 1) * k : lag * k] = self.Y[:-lag]
        return out[p:]

    def estimate(self):
        W = self._build_lag_matrix()
        self.irf = np.empty((self.H + 1, self.k))

        for j in range(self.k):
            y = self.Y[:, j]
            for h in range(self.H + 1):
                T_h = self.T - self.p - h
                X = sm.add_constant(np.append(self.x[:T_h].reshape(-1, 1), W[:T_h, :], axis = 1))
                lp_mod = sm.OLS(endog = y[self.p + h:], exog = X)
                lp_fit = lp_mod.fit(cov_type = 'HAC', cov_kwds = {'maxlags': h})
                self.beta[h, j] = lp_fit.params[1]
        return self.beta

    def smooth_irf_bspline(self, lam: float, n_points = 1000):
        self.irf = np.empty((n_points, self.k))
        h_grid = np.linspace(0, self.H + 1, n_points)
        for j in range(self.k):
            beta_j = self.beta[:, j]
            bs = interp.make_smoothing_spline(x = np.arange(0, self.H + 1), y = beta_j, lam = lam)
            self.irf[:, j] = bs(h_grid)
        return self.irf

    def smooth_irf_gam(self):
        self.irf = np.empty((n_poins, self.k))
        h_grid = np.linspace(0, self.H + 1, n_points)
        for j in range(self.k):
            beta_j = self.beta[:, j]
            gam = LinearGAM(s(0), fit_intercept = True).fit(X = np.arange(0, self.H + 1), y = beta_j)
            self.irf[:, j] = gam.predict(h_grid)
        return self.irf

    def smooth_irf_kernel(self, kernel: str = "gaussian", bandwidth: float = 1.0):
        h_seq = np.arange(self.H + 1).reshape(-1, 1)
        dist = np.abs(h_seq - h_seq.T).astype(float)
        match kernel:
            case "gaussian":
                S = stats.norm.pdf(dist, loc = 0, scale = bandwidth * 0.3706506)
            case "uniform":
                S = (dist <= bandwidth * 0.5).astype(float)
            case "epanechnikov"
                u = dist / bandwidth
                S = np.where(np.abs(u) <= 1, 0.75 * (1 - u ** 2), 0.0)
        S = S / S.sum(axis = 1, keepdims = True)
        return S @ self.irf
    
if __name__ == "__main__":
    # quick smoke test with simulated data
    T, n, p, H = 200, 3, 2, 12
    Y = np.random.randn(T, n)
    x = np.random.randn(T)

    model = ExoSmoothLP(Y = Y, p = p, H = H)
    model.estimate()
    model.irf
