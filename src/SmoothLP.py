import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.interpolate as interp
import pyGAM

class SmoothLP:

    """
    Smooth Local Projections (LP) estimator for fully endogenous data.
    Estimates impulse responses functions via horizon-by-horizon OLS.
    """

    def __init__(self, Y: np.ndarray, p_lag: int, H: int):
        """
        Parameters
        ----------
        Y : np.ndarray, shape (T, n)
            Matrix of endogenous variables
        p : int
            Number of lags to include as controls
        H : int
            Maximum horizon for IRF estimation
        """

        self.Y = Y 
        self.p_lag = p_lag
        self.H = H 
        self.T, self.n = Y.shape
        self.gamma = None # list of H+1 coefficient matrices
        self.irf   = None # array of shape (n, n, H+1)

    def _build_lag_matrix(self):
        # helper function to build and return W matrix
        pass

    def estimate(self):
        """
        Run OLS at each horizon h = 0, ..., H and identify shocks 
        via Cholesky decomposition of h=0 residual covariance. Stores
        results in self.gamma and self.irf.
        """
        W = self._build_lag_matrix()
        pass
    
if __name__ == "__main__"
    model = SmoothLP(Y = Y, p = p, H = H)
    model.fit()
