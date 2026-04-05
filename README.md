# ECON 622 Final Project
---
## Impulse Response Estimation by Smooth Local Projections

This repository consists of a collection of python methods which implement the smooth local projections (SLP) methodology presented in Barnichon and Brownlees (2019) for estimation of impulse response functions. The objective is to provide a compact yet straightforward implementation 

Local projections estimate the impulse response of a variable $y_t$ at horizon $h$ by the following OLS regression:

$$y_{t+h} = \mu_h + \beta_h x_t + \gamma_h'r_t + \sum_{k=1}^p \delta'_{h,k}w_{t-k} + \varepsilon_{h,t}$$

where $x_t$ is a shock vector, $r_t$ are contemporaneous controls, $w_{t-k}$ is a matrix with lags of endogenous variables, and $\varepsilon_{h,t}$ is the projection residual at horizon $h$. Barnichon and Brownlees (2019) propose utilising penalised B-splines to smooth the resulting sequence of impulse response coefficients $\{\beta_1, ..., \beta_h\}$.

This library contains generic routines for LP estimation of impulse response functions for endogenous and exogenous shocks and simple B-spline smoothing. In addition, it also includes some additional smoothing methods

### Example Usage

```{python}
from slp.estimator import LocalProjections
from slp.smoothers import BSplineSmoother
from slp.results import LPResults

# Smoke test with simulated data
T, n, p, H = 200, 3, 2, 24
Y = np.random.randn(T, n)
x = np.random.randn(T)

data = pd.DataFrame(
    np.column_stack([x, Y]),
    columns = ['shock_var', 'y1', 'y2', 'y3']
)

model = LocalProjections(data = data, shock = 'shock_var', p = p, H = H)
results = model.ExoLP()
irf = BSplineSmoother(lam = 0.8).smooth(results)

print(results.beta) # get sequence of LP estimates
print(irf)
```

### References

Barnichon, R. and Brownlees, C. (2019). Impulse response estimation by smooth local projections. *The Review of Economics and Statistics*, 101(3), 522–530.

Plagborg‐Møller, M., & Wolf, C. K. (2021). Local projections and VARs estimate the same impulse responses. *Econometrica*, 89(2), 955-980.

Olea, J. L. M., Plagborg-Møller, M., Qian, E., & Wolf, C. K. (2025). Local projections or VARs? a primer for macroeconomists (No. w33871). *National Bureau of Economic Research.*
