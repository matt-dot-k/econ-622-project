import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.linalg as linalg
from scipy.interpolate import BSpline
from .results import LPResults, SLPResults

class SmoothLocalProjections:
    """
    Implements the smooth local projections (SLP) methodology of Barnichon and Brownlees (2019).

    Estimates impulse responses by stacking LP regressions across horizons, approximating IRF
    coefficients beta(h) with a B-spline basis expansion, and solving via penalised least squares.
    """

    def __init__(self, data: pd.DataFrame, shock: str, endog: list[str], shock_exo: bool,
                 p: int, H: int):
        """
        Parameters
        ----------
        data      : pd.Dataframe
                    A pandas DataFrame with relevant variables. If data contains endogenous
                    shocks, the endogenous variables must be contemporaneously ordered.
        shock     : str
                    Name of the shock variable (x_t)
        endog     : list[str]
                    List of names of endogenous variables (Y_t). If None, all variables
                    are assumed to be endogenous.
        shock_exo : bool
                    Boolean indicator for whether shock variable (x_t) is exogenous
        p         : int
                    Lag order for the LP specification
        H         : int
                    Maximum IRF forecast horizon
        """
        # ----- Input checks -----
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"expected pd.DataFrame, got {type(data)}")
        if endog is not None and (not isinstance(endog, list) or not all(isinstance(e, str) for e in endog)):
            raise TypeError(f"endog must be a list of strings or None, got {type(endog)}")
        if not isinstance(shock_exo, bool):
            raise TypeError(f"expected bool, got {type(shock_exo)}")
        if not shock_exo and data.columns.get_loc(shock) == 0:
            raise ValueError(f"shock is endogenous but no columns precede shock in data - no instruments available.")
        T = len(data)
        if H >= T:
            raise ValueError(f"H must be less than the number of observations, got H = {H}, T = {T}")
        if p >= T:
            raise ValueError(f"p must be less than the number of observations, got p = {p}, T = {T}")

        Y = data.to_numpy()
        x = data[shock].to_numpy()
        if not shock_exo:
            contemp = data.columns[:data.columns.get_loc(shock)].tolist()
            Z = data[contemp].to_numpy()
        else:
            Z = None

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
        """
        Build a control matrix containing lags of endogenous variables.
        Returns a (T-p) * (k * p) matrix.
        """
        T, k = self.Y.shape
        out = np.full((T, k * self.p), np.nan)
        for lag in range(1, self.p + 1):
            out[lag:, (lag - 1) * k : lag * k] = self.Y[:-lag]
        W = out[self.p:]
        assert W.shape == (T - self.p, k * self.p), (
            f"lag matrix shape mismatch: expected ({T - self.p}, {k * self.p}), got {W.shape}"
        )
        return W

    # ----- Build B-spline basis -----
    def _build_bspline_basis(self, n_knots, degree) -> np.ndarray:
        """
        Build a design matrix for a B-spline basis expansion evaluated at
        integer horizons h = 0, ..., H. Returns a (H+1) * K matrix where
        K = n_knots + degree + 1
        """
        horizons = np.arange(self.H + 1, dtype = float)
        interior = np.quantile(horizons, np.linspace(0, 1, n_knots + 2)[1:-1])
        knots = np.concatenate([
            np.repeat(horizons[0], degree + 1), interior, np.repeat(horizons[-1], degree + 1),
        ])
        B = BSpline.design_matrix(horizons, knots, degree).toarray()

        assert B.shape == (self.H + 1, n_knots + degree + 1), (
            f"B-spline basis shape mismatch: expected ({self.H + 1}, {n_knots + degree + 1}, got {B.shape})"
        )
        assert np.allclose(B.sum(axis = 1), 1.0), (
            "B-spline rows do not sum to 1 - partition of unity violated"
        )

        return B

    # ----- Difference penalty matrix -----
    @staticmethod
    def _diff_penalty(K: int, r: int = 2) -> np.ndarray:
        """
        Build the penalty matrix D_r'D_r, where D_r is the r-th order
        difference operator of size (K-r) * K. Returns a K * K matrix.
        """
        D = np.diff(np.eye(K), n = r, axis = 0)
        P = D.T @ D
        assert P.shape == (K, K), (
            f"penalty matrix shape mismatch: expected({K},{K}), got {P.shape}"
        )
        return P

    # -----------------------------------------------------------------------
    # Classic local projections estimator for endogenous and exogenous shocks
    # -----------------------------------------------------------------------
    def LP(self) -> LPResults:
        """
        Estimate impulse responses to an exogenous or endogenous shock via the standard
        local LP method of Jorda (2005).

        Returns
        ------
        An LPResults object with an (H+1) * k matrix of estimated IRF coefficients.
        """
        W = self._build_lag_matrix()          # (T-p, k*p) 
        beta = np.empty((self.H + 1, self.k)) # (H+1, k)

        # ----- Loop over response variables -----
        for j in range(self.k):
            y = self.Y[:, j]
            # ----- OLS at each horizon -----
            for h in range(self.H + 1):
                T_h = self.T - self.p - h
                W_h = W[:T_h, :]
                if self.Z is not None:
                    W_h = np.hstack([self.Z[self.p:self.p  + T_h], W_h])
                X_h = np.append(self.x[self.p:self.p + T_h].reshape(-1, 1), W_h, axis = 1)
                X_h = sm.add_constant(X_h)
                y_h = y[self.p + h: ]
                assert y_h.shape[0] == X_h.shape[0], (
                    f"row count mismatch at h = {h}: y_h has {y_h.shape[0]} rows, X_h has {X_h.shape[0]}"
                )
                lp_mod = sm.OLS(endog = y_h, exog = X_h)
                lp_fit = lp_mod.fit(cov_type = 'HAC', cov_kwds = {'maxlags': h})
                beta[h, j] = lp_fit.params[1]
 
        # ----- Check finiteness of estimates -----
        if not np.isfinite(beta).all():
            raise ArithmeticError(
                "non-finite values in LP IRF estimates; check data for outliers or multicollinearity"
            )
        return LPResults(beta = beta, H = self.H, k = self.k)

    # ----------------------------------------------------------------------
    # Smooth local projections estimator for endogenous and exogenous shocks
    # ---------------------------------------------------------------------- 
    def SLP(self, n_knots: int = 5, degree: int = 3, lam: float = 1.0, r: int = 2) -> SLPResults:
        """
        Estimate impulse responses to an endogenous or exogenous shock
        via penalised B-splines (P-splines)

        Parameters
        ----------
        n_knots : int
                  Number of knots (q) for the B-spline basis. Basis is constructed using
                  q + 2 inner knots.
        degree  : int
                  Polynomial degree (p) for the B-spline basis expansion. Defaults to
                  cubic splines (p = 3).
        lam     : float
                  Shrinkage parameter lambda. (0 = OLS, large = polynomial of order r - 1).
                  Can be externally selected via cross-validation.
        r       : int
                  Order of the difference penalty (2 = shrinks toward a linear function,
                  3 = shrinks toward a quadratic function)

        Returns
        -------
        An SLPResults object with an (H+1) * k matrix of estimated IRF coefficients.
        """
        # ----- Input checks -----
        if not isinstance(n_knots, int):
            raise TypeError(f"expected int, got {type(n_knots)}")
        if not isinstance(degree, int):
            raise TypeError(f"expected int, got {type(degree)}")
        if not isinstance(lam, (int, float)):
            raise TypeError(f"expected float, got {type(lam)}")
        if not isinstance(r, int):
            raise TypeError(f"expected int, got {type(r)}")
        if not (n_knots > 0):
            raise ValueError(
                f"at least 1 knot required for B-spline basis, got n_knots = {n_knots}")
        if not (degree > 0):
            raise ValueError(f"minimum degree 1 required to form B-spline basis, got degree = {degree}")
        if not (lam >= 0):
            raise ValueError(f"penalty term must be at least 0, got lam = {lam}")
        if not (r < n_knots + degree + 1):
            raise ValueError(
                    f"r cannot exceed n_knots + degree + 1, otherwise penalty matrix will be "
                    "degenerate, got n_knots = {n_knots}, degree = {degree} and r = {r}"
            )

        W = self._build_lag_matrix()                   # (T-p, k*p)
        B = self._build_bspline_basis(n_knots, degree) # (H+1, K)
        beta = np.empty((self.H + 1, self.k))          # (H+1, k)
        K = B.shape[1]
        n_w = W.shape[1]                               # no. of control columns
        if self.Z is not None:
            n_w += self.Z.shape[1]

        # ----- Build penalty matrix -----
        P_block = self._diff_penalty(K, r)
        P_full = linalg.block_diag(np.zeros((K, K)), P_block, *[P_block] * n_w)
        
        # ----- Loop over each response variable -----
        for j in range(self.k):
            Y_blocks = []
            A_blocks = []
            W_blocks = []
            X_blocks = []

            # ----- Stack across horizons -----
            for h in range(self.H + 1):
                T_h = self.T - self.p - h
                b_h = B[h, :]
                A_basis_h = np.tile(b_h, (T_h, 1))
                X_basis_h = np.outer(self.x[self.p:self.p + T_h], b_h)
                W_h = W[:T_h, :]
                if self.Z is not None:
                    W_h = np.hstack([self.Z[self.p:self.p + T_h], W_h])
                W_basis_h = np.hstack([np.outer(W_h[:, c], b_h) for c in range(W_h.shape[1])])
                Y_h = self.Y[self.p + h:self.p + h + T_h, j]

                # ----- Check matrix dimensions -----
                assert X_basis_h.shape == (T_h, K), (
                    f"X_basis_h shape mismatch at h = {h}: expected ({T_h}, {K}), got {X_basis_h.shape}"    
                )
                assert W_basis_h.shape == (T_h, n_w * K), (
                    f"W_basis_h shape mismatch at h = {h}: expected ({T_h}, {n_w * K}), got {W_basis_h.shape}"
                )
                assert Y_h.shape[0] == T_h, (
                    f"Y_h length mismatch at h = {h}: expected {T_h}, got {Y_h.shape[0]}"
                )

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
            # theta = (X'X + lam * P)^{-1} X'Y
            XtX = X_cal.T @ X_cal
            XtY = X_cal.T @ Y_cal
            assert XtX.shape == P_full.shape, (
                f"XtX shape {XtX.shape} != P_full shape {P_full.shape} - cannot add penalty"
            ) 
            # ----- Check condition number of least squares problem -----
            mat = XtX + lam * P_full
            cond = np.linalg.cond(mat)
            if cond > 1e12:
                warnings.warn(
                    f"near-singular system for variable j = {j} (condition number {cond:.2e}); "
                    "solution may be inaccurate — try increasing lam or reducing n_knots",
                    RuntimeWarning,
                    stacklevel = 2,
                )
            theta = np.linalg.solve(mat, XtY)
            if not np.isfinite(theta).all():
                raise ArithmeticError(
                    f"non-finite values in theta for variable j = {j}; system may be near-singular"
                    "try increasing lam or reducing n_knots"
                )
            # ----- Recover impulse response: beta(h) = B @ delta -----
            delta = theta[K:2*K]
            beta[:, j] = B @ delta
            if not np.isfinite(beta[:, j]).all():
                raise ArithmeticError(
                    f"non-finite values in SLP IRF estimates for variable j = {j}"
                )
        return SLPResults(beta = beta, H = self.H, k = self.k)
