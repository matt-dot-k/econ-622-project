import numpy as np
import scipy.stats as stats
import scipy.interpolate as interp
from pygam import LinearGAM, s
from .results import LPResults

class LoessSmoother:
    """
    A simple LOESS smoother. Smooths the resulting sequence of IRF coefficients
    by weighted linear least squares on subsets of coefficients.
    """
    
    def __init__(self, frac: float = 0.5):
        """
        Parameters
        ----------
        frac : The fraction of observations to use in each local regression.
        """
        self.frac = frac

    def smooth(self, results: LPResults) -> np.ndarray:
        """
        Returns
        -------
        An (H+1) * k matrix containing the smoothed IRF coefficients.

        """
        irf = np.empty((results.H + 1, results.k))
        h_grid = np.arange(results.H + 1)
        for j in range(results.k):
            beta_j = results.beta[:, j]
            loess = sm.nonparametric.lowess(endog = beta_j, exog = h_grid, frac = self.frac)
            irf[:, j] = loess[:, 1]
        return irf

class KernelSmoother:
    """
    A simple kernel smoother. Smooths the resulting sequence of IRF coefficients
    via a weighted average, using a kernel function to determine the weights.
    """

    def __init__(self, kernel: str = "gaussian", band: float = 1.0):
        """
        Parameters
        ----------
        kernel : A string specifying the kernel function to use.
        band   : The bandwidth for the kernel function. Larger values create a 
                 smoother curve, smaller values track the LP IRF more closely.
        """
        self.kernel = kernel
        self.band = band

    def smooth(self, results: LPResults) -> np.ndarray:
        """
        Returns
        -------
        An (H+1) * k matrix containing the smoothed IRF coefficients.
        """
        h_seq = np.arange(results.H + 1).reshape(-1, 1)
        dist = np.abs(h_seq - h_seq.T).astype(float)
        match self.kernel:
            case "gaussian":
                S = stats.norm.pdf(dist, loc = 0, scale = self.band * 0.3706506)
            case "uniform":
                S = (dist <= self.band * 0.5).astype(float)
            case "epanechnikov":
                u = dist / self.band
                S = np.where(np.abs(u) <= 1, 0.75 * (1 - u ** 2), 0.0)
        S = S / S.sum(axis = 1, keepdims = True)
        irf = S @ results.beta
        return irf
