import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.interpolate as interp
import scipy.linalg as linalg
from .results import LPResults, SLPResults

class SmoothLocalProjections:

    def __init__(self, data: pd.DataFrame, shock: str, endog: list[str], shock_exo: bool,
                 p: int, H: int):

        Y_df = data.drop(columns = shock) if endog is None else data[endog]
        contemp = data.columns[:data.columns.get_loc(shock)].tolist()
        Z = data[contemp].to_numpy() if shock_exo is True else None
        Y = Y_df.to_numpy()
        x = data[shock].to_numpy()

        self.Y = Y
        self.T, self.k = Y.shape
        self.Z = Z
        self.x = x
        self.p = p
        self.H = H

    # ------------------------------------------
    # Helper functions for LP and SLP estimators
    # ------------------------------------------

    # ----- Build lag matrix W -----
    def _build_lag_matrix(self):
        T, k = self.Y.shape
        out = np.full((T, k * self.p), np.nan)
        for lag in range(1, self.p + 1):
            out[lag:, (lag - 1) * k : lag * k] = self.Y[:-lag]
        return out[self.p:]

    # ----- Build B-spline basis -----
    def _build_bspline_basis(self, n_knots, degree) -> np.ndarray:
        horizons = np.arange(self.H + 1, dtype = float)
        interior = np.quantile(horizons, np.linspace(0, 1, n_knots + 2)[1:-1])
        knots = np.concatenate([
            np.repeat(horizons[0], degree + 1),
            interior,
            np.repeat(horizons[-1], degree + 1),
        ])
        B = interp.BSpline.design_matrix(horizons, knots, degree).toarray()
        return B

    # ----- Difference penalty matrix -----
    @staticmethod
    def _diff_penalty(K: int, r: int = 2) -> np.ndarray:
        D = np.diff(np.eye(K), n = r, axis = 0)
        return D.T @ D

    # -----------------------------------------------------------------------
    # Classic local projections estimator for endogenous and exogenous shocks
    # -----------------------------------------------------------------------
    def LP(self) -> LPResults:
        W = self._build_lag_matrix()
        beta = np.empty((self.H + 1, self.k))

        for j in range(self.k):
            y = self.Y[:, j]
            for h in range(self.H + 1):
                T_h = self.T - self.p - h
                W_h = W[:T_h, :]
                if self.Z is not None:
                    W_h = np.hstack([self.Z[self.p:self.p  + T_h], W_h])
                X_h = np.append(self.x[self.p:self.p + T_h].reshape(-1, 1), W[:T_h, :], axis = 1)
                X_h = sm.add_constant(X_h)
                y_h = y[self.p + h: ]
                lp_mod = sm.OLS(endog = y_h, exog = X_h)
                lp_fit = lp_mod.fit(cov_type = 'HAC', cov_kwds = {'maxlags': h})
                beta[h, j] = lp_fit.params[1]

        return LPResults(beta = beta, H = self.H, k = self.k)

    # ----------------------------------------------------------------------
    # Smooth local projections estimator for endogenous and exogenous shocks
    # ---------------------------------------------------------------------- 
    def SLP(self, n_knots: int = 5, degree: int = 3, lam: float = 1.0, r: int = 2) -> SLPResults:
        W = self._build_lag_matrix()
        B = self._build_bspline_basis(n_knots, degree)
        beta = np.empty((self.H + 1, self.k))
        K = B.shape[1]
        n_w = W.shape[1]
        if self.Z is not None:
            n_w += self.Z.shape[1]

        # ----- Build penalty matrix -----
        P_block = self._diff_penalty(K, r)
        P_full = linalg.block_diag(np.zeros((K, K)), P_block, *[P_block] * n_w)

        for j in range(self.k):
            # ----- Stack across horizons -----
            Y_blocks = []
            A_blocks = []
            W_blocks = []
            X_blocks = []

            for h in range(self.H + 1):
                T_h = self.T - self.p - h
                b_h = B[h, :]
                A_basis_h = np.tile(B[h, :], (T_h, 1))
                X_basis_h = np.outer(self.x[self.p:self.p + T_h], b_h)
                W_h = W[:T_h, :]
                if self.Z is not None:
                    W_h = np.hstack([self.Z[self.p:self.p + T_h], W_h])
                W_basis_h = np.hstack([
                    np.outer(W_h[:, c], B[h, :]) for c in range(W_h.shape[1])
                ])
                Y_h = self.Y[self.p + h:self.p + h + T_h, j]

                Y_blocks.append(Y_h)
                A_blocks.append(A_basis_h)
                W_blocks.append(W_basis_h)
                X_blocks.append(X_basis_h)

            # ----- Assemble stacked system -----
            Y_cal = np.concatenate(Y_blocks)
            A_tilde = np.vstack(A_blocks)
            W_tilde = np.vstack(W_blocks)
            X_tilde = np.vstack(X_blocks)
            X_cal = np.hstack([A_tilde, X_tilde, W_tilde])

            # ----- Ridge estimation of B-spline coefficients -----
            XtX = X_cal.T @ X_cal
            XtY = X_cal.T @ Y_cal
            theta = np.linalg.solve(XtX + lam * P_full, XtY)

            # ----- Recover impulse response: beta(h) = B @ delta
            delta = theta[K:2*K]
            beta[:, j] = B @ delta

        return SLPResults(beta = beta, H = self.H, k = self.k)
