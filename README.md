# ECON 622 Final Project
---
## Impulse Response Estimation by Smooth Local Projections

This repository consists of a collection of python methods which implement the smooth local projections (SLP) methodology presented in Barnichon and Brownlees (2019) for estimation of impulse response functions. The objective is to provide a compact yet straightforward implementation, as well as some demonstrations of the method in practice.

Local projections estimate the impulse response of a variable $y_t$ to a shock $x_t$ at horizon $h$ by the following OLS regression:

$$
\begin{equation}
    y_{t+h} = \mu_h + \beta_h x_t + \gamma_h'r_t + \sum_{k=1}^p \delta'_{h,k}w_{t-k} + \varepsilon_{h,t}
\end{equation}
$$

where $x_t$ is a shock vector, $r_t$ are contemporaneous controls, $w_{t-k}$ is a matrix with lags of the endogenous variables, and $\varepsilon_{h,t}$ is the projection residual at horizon $h$. Barnichon and Brownlees (2019) propose approximating the LP in equation (1) above with a linear B-splines basis function expansion of the form:

$$
\begin{equation}
    y_{t+h} \approx \sum_{k=1}^K a_kB_k(h) + \sum_{k=1}^K b_kB_k(h)x_t + \sum_{i=1}^p\sum_{k=1}^K c_{ik}B_k(h)w_{it} + u_{(h)t + h}
\end{equation}
$$

Where $B_k:\mathbb{R}\to\mathbb{R}$ for $k=1,...K$ is a set of B-spline basis functions and $a_k,b_k,c_{ik}$ for $k=1,...,K$ is a set of scalar coefficients. This library contains a set of generic routines for estimating impulse responses of endogenous and exogenous shocks using the SLP approximation above. In addition, it contains routines for performing classic local projections estimation.

### Example Usage

The following code shows a basic expository example of how to utilise the methods available in this repository:

```{python}
from slp.estimators import SmoothLocalProjections
from slp.results import LPResults, SLPResults

# Smoke test with simulated data
T, n, p, H = 200, 3, 2, 24
Y = np.random.randn(T, n)
x = np.random.randn(T)

data = pd.DataFrame(
    np.column_stack([x, Y]),
    columns = ['y1', 'y2', 'y3', 'shock_var']
)

# Initiate the LP object
model = SmoothLocalProjections(data = data, shock = 'shock_var', endog = ['y1', 'y2', 'y3'],
                               shock_exo = False, p = p, H = H)

# Run LP/SLP estimation
slp_mod = model.SLP(n_knots = 5, degree = 3, lam = 1.0, r = 2)
lp_mod = model.LP()

# Extract estimated IRFs
slp_irf = slp_mod.beta
lp_irf = lp_mod.beta
```

`smoothers.py` also contains some extra smoothing methods which can be applied to the raw IRF coefficients estimated by `LP()` for flexibility or illustrative purposes. These can be used like so:

```{python}
from slp.smoothers import LoessSmoother, KernelSmoother

lp_irf_kernel = KernelSmoother(kernel = 'gaussian', band = 5).smooth(lp_irf)
lp_irf_loess = LoessSmoother(frac = 0.66).smooth(lp_irf)

print(lp_irf_kernel)
print(lp_irf_loess)
```

### References

Barnichon, R. and Brownlees, C. (2019). Impulse response estimation by smooth local projections. *The Review of Economics and Statistics*, 101(3), 522–530.

Plagborg‐Møller, M., & Wolf, C. K. (2021). Local projections and VARs estimate the same impulse responses. *Econometrica*, 89(2), 955-980.

Olea, J. L. M., Plagborg-Møller, M., Qian, E., & Wolf, C. K. (2025). Local projections or VARs? a primer for macroeconomists (No. w33871). *National Bureau of Economic Research.*
