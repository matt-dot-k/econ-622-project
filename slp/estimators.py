import numpy as np
import pandas as pd
import statsmodels.api as sm
from .results import LPResults

class LocalProjections:

    def __init__(self, data: pd.DataFrame, shock: str, endog: list[str], p: int, H: int):

        Y_df = data.drop(columns = shock) if endog is None else data[endog]
        Y = Y_df.to_numpy()
        contemp = data.columns[:data.columns.get_loc(shock)].tolist()
        Z = data[contemp].to_numpy()
        x = data[shock].to_numpy()
        var_names = Y_df.columns.tolist()

        self.Y = Y
        self.Z = Z
        self.x = x
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
        return out[self.p:]

    def ExoLP(self) -> LPResults:
        W = self._build_lag_matrix()
        beta = np.empty((self.H + 1, self.k))

        for j in range(self.k):
            y = self.Y[:, j]
            for h in range(self.H + 1):
                T_h = self.T - self.p - h
                X_h = np.append(self.x[self.p:self.p + T_h].reshape(-1, 1), W[:T_h, :], axis = 1)
                X_h = sm.add_constant(X_h)
                y_h = y[self.p + h: ]
                lp_mod = sm.OLS(endog = y_h, exog = X_h)
                lp_fit = lp_mod.fit(cov_type = 'HAC', cov_kwds = {'maxlags': h})
                beta[h, j] = lp_fit.params[1]
                
        return LPResults(beta = beta, H = self.H, k = self.k)
    
    def EndoLP(self):
        W = self._build_lag_matrix()
        beta = np.empty((self.H + 1, self.k))

        for j in range(self.k):
            y = self.Y[:, j]
            for h in range(self.H + 1):
                T_h = self.T - self.p - h
                X_h = np.column_stack([self.x[self.p:self.p + T_h], self.Z[self.p:self.p + T_h, :], W[:T_h]])
                X_h = sm.add_constant(X_h)
                y_h = y[self.p + h: ]
                lp_mod = sm.OLS(endog = y_h, exog = X_h)
                lp_fit = lp_mod.fit(cov_type = 'HAC', cov_kwds = {'maxlags': h})
                beta[h, j] = lp_fit.params[1]
        return LPResults(beta = beta, H = self.H, k = self.k)
