import numpy as np
import scipy.stats as stats
import scipy.interpolate as interp
from pygam import LinearGAM, s
from .results import LPResults

class BSplineSmoother:

    def __init__(self, lam: float, n_points: int = 1000):
        self.lam = lam
        self.n_points = n_points

    def smooth(self, results: LPResults) -> np.ndarray:
        irf = np.empty((self.n_points, results.k))
        h_grid = np.linspace(0, results.H + 1, self.n_points)
        for j in range(results.k):
            beta_j = results.beta[:, j]
            bs = interp.make_smoothing_spline(x = np.arange(0, results.H + 1), y = beta_j, lam = self.lam)
            irf[:, j] = bs(h_grid)
            
        return irf

class GAMSmoother:
    
    def __init__(self, n_points: int = 1000):

        self.n_points = n_points

    def smooth(self, results: LPResults) -> np.ndarray:
        irf = np.empty((self.n_points, results.k))
        h_grid = np.linspace(0, results.H + 1, self.n_points)
        for j in range(self.k):
            beta_j = results.beta[:, j]
            gam = LinearGAM(s(0), fit_intercept = True).fit(X = np.arange(0, results.H + 1), y = beta_j)
            irf[:, j] = gam.predict(h_grid)
        
        return irf

class KernelSmoother:

    def __init__(self, kernel: str = "gaussian", band: float = 1.0):
        
        self.kernel = kernel
        self.band = band

    def smooth(self, results: LPResults) -> np.ndarray:
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
        return S @ results.beta
