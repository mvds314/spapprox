# Vector-valued estimating equations

Status: implemented (branch `feature/vector-estimating-equations`). Follow-up of
`docs/estimating_equations_plan.md`, which covers a scalar parameter.

## Idea

The estimator `tau_hat` in `R^p` is the root of `S(tau) = sum_i psi(x_i; tau) W_i = 0`, with
`psi` in `R^p` and i.i.d. random weights `W_i`, optionally conditional on `T = sum_i W_i = t`.
For a scalar parameter, `P(tau_hat <= tau) = P(S(tau) <= 0)` gives the cdf; for `p > 1`
there is no such identity, so the starting point is the density (Skovgaard, 1990; Strawderman,
Casella and Wells, 1996; Davison and Hinkley, 1997, Section 9.5.2):

    f(tau) ~ |det E_s[dS/dtau']| f_{S(tau) | T}(0 | t),

with the expectation under the exponential tilt `s` that puts the mean of `(T, S(tau))` at
`(t, 0)`. With `a_i = s_T + s_S' psi_i(tau)` and `K_W` the cgf of the weights:

- `K(s; tau) = sum_i K_W(a_i)`, gradient `sum_i K_W'(a_i) (1, psi_i)`, Hessian
  `sum_i K_W''(a_i) (1, psi_i)(1, psi_i)'`; without conditioning drop the leading `1`.
- `f_{T, S}(t, 0) ~ exp(K(s) - s_T t) / ((2 pi)^((p+1)/2) sqrt(det K''(s)))`, divided by the
  saddle point density of `T`, whose cgf is `n K_W`.
- `E_s[dS/dtau'] = sum_i K_W'(a_i) dpsi_i/dtau'`, since the tilted `W_i` are independent with
  means `K_W'(a_i)`.

The scalar class gets the tilted Jacobian as `d_tau K(s; tau) / s_S`, which needs only the cgf
of `S(tau)`. That trick has no matrix analogue, so the vector class needs the weighted
structure: it takes `psi`, `x` and the cgf of the weights, rather than a cgf per `tau`. It
evaluates `K_W`, `K_W'`, `K_W''` vectorized over `i` and solves the saddle point equation by
damped Newton, which is about 30 times faster than building the cgf of `(T, S(tau))` with
`from_univariate(...).ldot(...)` for every `tau`. That matters, as normalization and marginals
need a grid of `tau` values.

## API

`MultivariateEstimatingEquationSaddlePointApprox(psi, x, weights, given=None, dpsi=None,
bracket=None)`, with `psi(x, tau)` returning an `(n, p)` array and `dpsi(x, tau)` an
`(n, p, p)` array (central differences if not given).

- `pdf(tau, normalize_pdf=True)`: `tau` of shape `(..., p)`. Normalized with the trapezoidal
  rule on a tensor grid over `bracket` (one `(lo, hi)` pair per component).
- `fit_pdf(bracket=None, num=41)`: tabulate the density on the grid; used for the
  normalization, `marginal_pdf`, `marginal_cdf`, `marginal_ppf`, `mean` and `cov`. The grid
  has `num**p` points, so this is meant for small `p`.

## Checks

- `p = 1` reproduces the scalar `EstimatingEquationSaddlePointApprox` density.
- The vector mean, `psi = x - tau`, with Poisson weights given `n` reproduces the saddle point
  density of the mean of `n` draws from the bivariate empirical distribution (the cut).
- Least squares regression under the Bayesian (Dirichlet) bootstrap: marginal quantiles agree
  with Monte Carlo, see `examples/example_bootstrap_regression.py`.

## Not done

- Tail probabilities for a component of `tau_hat` without integrating the density, e.g.
  Jensen and Wood (1998) or the Skovgaard (1990) marginal approximation.
- Lattice corrections.
