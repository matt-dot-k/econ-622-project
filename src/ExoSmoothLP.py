import numpy as np
import pandas as pd
import statsmodels.api as sm

class LinearProjections:

    def __init__(self, data: pd.DataFrame, shock: str, endog: list[str] = None, p: int, H: int):

        Y_df = data.drop(columns = shock) if endog is None else data[endog_cols]
        Y = Y_df.to_numpy()
        contemp = data.columns[:data.columns.get_loc(shock)].tolist()
        Z = data[contemp].to_numpy()
        x = data[shock].to_numpy()
        var_names = Y_df.columns.tolist()

        self.Y = Y
        self.x = x
        self.Z = Z
        self.p = p
        self.H = H
        self.T, self.k = Y.shape
        self.beta = None
        self.irf = None
        self.var_names = var_names

    def _build_lag_matrix(self):
        # helper function to build lag matrix W
        T, k = self.Y.shape
        out = np.full((T, k * self.p), np.nan)
        for lag in range(1, self.p + 1):
            out[lag:, (lag - 1) * k : lag * k] = self.Y[:-lag]
        return out[p:]

    def ExoLP(self):
        W = self._build_lag_matrix()
        self.beta = np.empty((self.H + 1, self.k))

        for j in range(self.k):
            y = self.Y[:, j]
            for h in range(self.H + 1):
                T_h = self.T - self.p - h
                X_h = np.append(self.x[self.p:self.p + T_h].reshape(-1, 1), W[:T_h, :], axis = 1)
                X_h = sm.add_constant(X_h)
                y_h = y[self.p + h: ]
                lp_mod = sm.OLS(endog = y_h, exog = X_h)
                lp_fit = lp_mod.fit(cov_type = 'HAC', cov_kwds = {'maxlags': h})
                self.beta[h, j] = lp_fit.params[1]
        return self.beta
    
    def EndoLP(self):
        W = self._build_lag_matrix()
        self.beta = np.empty((self.H + 1, self.k))

        for j in range(self.k):
            y = self.Y[:, j]
            for h in range(self.H + 1):
                T_h = self.T - self.p - h
                X_h = np.hstack([self.x[self.p:self.p + T_h], self.Z[self.p:self.p + T_h], self.W[:T_h]])
                X_h = sm.add_constant(X_h)
                y_h = y[self.p + h: ]
                lp_mod = sm.OLS(endog = y_h, exog = X_h)
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

    def smooth_irf_gam(self, n_points: int = 1000):
        self.irf = np.empty((n_points, self.k))
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

    model = LinearProjections(Y = Y, p = p, H = H)
    model.estimate()
    model.irf
