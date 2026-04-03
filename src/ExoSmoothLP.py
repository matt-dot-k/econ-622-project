import numpy as np
from numpy.linalg import lstsq, cholesky
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.interpolate as interp
import scipy.stats as stats
import pyGAM

class ExoSmoothLP:
    """
    Smooth Local Projections (LP) estimator for exogenous shocks.
    Estimates impulse response functions via horizon-by-horizon OLS.
    """

    def __init__(self, Y: np.ndarray, x: np.ndarray, p: int, H: int):
        """
        Parameters
        ----------
        Y : np.ndarray, shape (T, n)
            Vector of endogenous variables
        x : np.ndarray, shape (T, 1)
            Exogenous shock vector
        p : int
            Lag order to apply for the local projections
        H : int
            Maximum horizon for impulse response estimation
        """

        self.Y = Y
        self.x = x
        self.p = p
        self.H = H
        self.T, self.k = Y.shape
        self.irf = None

    def _build_lag_matrix(self):
        # helper function to build lag matrix W
        T, k = self.Y.shape
        out = np.full((T, k * p)), np.nan)
        for lag in range(1, self.p + 1):
            out[lag:, (lag - 1) * k : lag * k] = self.Y[:-lag]
        return out[p:]

    def estimate(self):
        """
        Estimate local projections with exogenous shocks via horizon-by-horizon
        ordinary least squares (OLS)
        """

        W = self._build_lag_matrix()
        self.irf = np.empty((self.H, self.k))

        # run LP at each horizon h for each response
        for j in range(self.k):
            y = self.Y[:, j]
            for h in range(self.H + 1):
                T_h = self.T - self.p - h
                X = sm.add_constant(np.append(self.x[:T_h].reshape(-1, 1), W[:T_h, :], axis = 1))
                lp_mod = sm.OLS(endog = y[self.p + h:], exog = X)
                lp_fit = lp_mod.fit(cov_type = 'HAC', cov_kwds = {'maxlags': h})
                self.irf[h, j] = lp_fit.params[1]
        return self.irf
    
if __name__ == "__main__"
    # quick smoke test with simulated data
    T, n, p, H = 200, 3, 2, 12
    Y = np.random.randn(T, n)
    x = np.random.randn(T)

    model = SmoothLP(Y = Y, p = p, H = H)
    est = model.estimate(x)
    model.irf
