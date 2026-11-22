# spapprox

Saddle point approximations of probability distributions from their cumulant generating
functions, in Python.

If you know the cumulant generating function (cgf) of a random variable or random vector,
`spapprox` approximates its pdf, cdf and quantiles. This is especially useful for sums and
means of independent random variables, and for resampling distributions such as the
bootstrap, where the cgf is easy to write down but the distribution is not.

The implementation follows Ronald W. Butler (2007), *Saddlepoint Approximations with
Applications*. As he puts it:

> Among the various tools that have been developed for use in statistics and probability
> over the years, perhaps the least understood and most remarkable tool is the saddlepoint
> approximation. It is remarkable because it usually provides probability approximations
> whose accuracy is much greater than the current supporting theory would suggest.

## Installation

```bash
pip install git+https://github.com/mvds314/spapprox.git
```

`spapprox` requires Python 3.11 or later. Optional dependencies, installed as extras:

- `numdiff`: [numdifftools](https://pypi.org/project/numdifftools/), the default numerical
  differentiation backend, used when a cgf has no analytic derivatives.
- `findiff`: [findiff](https://pypi.org/project/findiff/) (>= 0.11), an alternative finite
  difference backend.
- `fastnorm`: [fastnorm](https://pypi.org/project/fastnorm/), for faster bivariate normal
  cdf evaluation.
- `all`: all of the above.

```bash
pip install "spapprox[all] @ git+https://github.com/mvds314/spapprox.git"
```

## Why cumulant generating functions?

The cgf of a random variable `X` is `K(t) = log E[exp(t X)]`. It determines the
distribution, its derivatives at zero are the cumulants (`K'(0)` is the mean and `K''(0)`
the variance), and it is **additive over independent random variables**: the cgf of
`X + Y` is `K_X(t) + K_Y(t)`. You cannot simply add distribution functions, but you can
add cgfs.

The saddle point approximation then recovers the density from the cgf:

```
f(x) ≈ exp(K(t) - t x) / sqrt(2 π K''(t)),   where t solves K'(t) = x.
```

`K'(t) = x` is the *saddle point equation*. The cdf follows from the
Lugannani-Rice or Barndorff-Nielsen formulas. The
[Wikipedia page](https://en.wikipedia.org/wiki/Saddlepoint_approximation_method) is a good
introduction.

## Quick start

Build a cgf, combine cgfs if needed, and pass the result to an approximation class:

```python
import numpy as np
import scipy.stats as sps
import spapprox as spa

# The sum of a standard normal and an independent exponential with scale 2
cgf = spa.norm(loc=0, scale=1) + spa.exponential(scale=2)
approx = spa.UnivariateSaddlePointApprox(cgf)

approx.pdf(x=[0.0, 2.0, 5.0])
approx.cdf(x=[0.0, 2.0, 5.0])  # Lugannani-Rice, or backend="BN" for Barndorff-Nielsen
approx.ppf([0.05, 0.5, 0.95])

# Check against the exact distribution in a simple case
cgf_normal = spa.norm(loc=0, scale=1)
spa_normal = spa.UnivariateSaddlePointApprox(cgf_normal)
t = np.linspace(-3, 3)
x = cgf_normal.dK(t)
np.allclose(spa_normal.pdf(t=t), sps.norm.pdf(x))
```

Every `pdf` and `cdf` takes either the point `x` or the saddle point `t`. Passing `x` solves
the saddle point equation for every value. Passing `t` (and getting `x = cgf.dK(t)`) avoids
that, which is much faster for plotting a whole curve. To evaluate at many `x` values
repeatedly, call `approx.fit_saddle_point_eqn()` first to tabulate the solution.

## Features

- **Algebra on cgfs:** `cgf1 + cgf2` is the sum of independent random variables, `a * cgf + b`
  an affine transformation, `mcgf[i]` a marginal, and `mcgf.ldot(A)` the linear transformation
  `A X` of a random vector.
- **Univariate approximations** (`UnivariateSaddlePointApprox`): pdf, cdf (Lugannani-Rice or
  Barndorff-Nielsen) and ppf. `UnivariateSaddlePointApproxMean` approximates the
  distribution of a sample mean.
- **Discrete (lattice) variables:** pass the span of the lattice, e.g. `cdf(x, delta=1)` or
  `pdf(x, delta=1)`, for the continuity-corrected cdf or the probability mass function.
- **Multivariate approximations** (`MultivariateSaddlePointApprox`,
  `BivariateSaddlePointApprox`): joint pdf, marginals, and the joint cdf following Kolassa
  (2003) or Kolassa and Li (2010), in any dimension. The bivariate class adds Wang (1990).
- **Conditional distributions:** `P(X_i ≤ x | X_J = x_J)` with the double saddle point
  approximation of Skovgaard (1987), or Kolassa (1996) with `backend="Kolassa"`. Two
  variables of interest are supported too (Kolassa and Li, 2010).
- **Transform-domain bootstrap:** the bootstrap distribution of a sample mean without
  resampling, by combining `univariate_empirical` and `univariate_sample_mean`.
- **Estimating equations and resampling** (`EstimatingEquationSaddlePointApprox`,
  `MultivariateEstimatingEquationSaddlePointApprox`): the distribution of the root of a
  randomly weighted estimating equation (Daniels, 1983; Booth and Butler, 1990). This covers
  the multinomial and Dirichlet bootstrap, the delete-m jackknife and permutation tests.

### Available cgfs

| Univariate | Multivariate | Transformations |
|---|---|---|
| `norm`, `exponential`, `gamma`, `chi2`, `laplace`, `asymmetric_laplace`, `generalized_normal`, `asymmetric_generalized_normal`, `poisson`, `bernoulli`, `binomial`, `negative_binomial` | `multivariate_norm`, `bivariate_gamma`, `jensen_multivariate_gamma`, `multinomial`, `bivariate_poisson` | `univariate_empirical` (the empirical distribution of a sample), `univariate_sample_mean` |

Any other cgf can be supplied directly with `UnivariateCumulantGeneratingFunction` or
`MultivariateCumulantGeneratingFunction`. Derivatives you do not supply are computed
numerically. A `Domain` describes where the cgf is finite.

## Examples

### Conditional distributions

```python
cgf = spa.multivariate_norm(loc=[0, 0], cov=[[1, 0.5], [0.5, 1]])
spa_mvn = spa.MultivariateSaddlePointApprox(cgf)
spa_mvn.conditional_cdf([-1, 0, 1], given=[0.3], i=0)
spa_mvn.conditional_pdf([-1, 0, 1], given=[0.3], i=0)  # normalized, normalize_pdf=False to skip
```

### Estimating equations and resampling

The root `τ̂` of the randomly weighted estimating equation `Σ_i ψ(x_i; τ) W_i = 0`,
optionally conditional on `Σ_i W_i`, depends on the resampling scheme through the weights:

- Poisson weights given their sum `n` give the multinomial bootstrap.
- Exponential weights give the Dirichlet bootstrap.
- Bernoulli weights given `n - m` give the delete-m jackknife.

```python
x = np.array([9.6, 10.4, 13.0, 15.0, 16.6, 17.2, 17.3, 21.8, 24.0, 33.8])
ee = spa.EstimatingEquationSaddlePointApprox.from_resampling(
    lambda x, tau: x - tau, x, spa.exponential(), given=len(x), bracket=(x.min(), x.max())
)
ee.cdf([15, 18, 21])
ee.ppf([0.05, 0.95])
```

`cdf` and `ppf` solve saddle point equations for every value. After `ee.fit_cdf()`, pass
`interpolate=True` to interpolate the tabulated cdf instead. This is much faster, and
accurate up to the resolution of the table. For a discrete `S(τ)`, e.g. a mean of integer
data under the multinomial bootstrap, pass `delta=1` to `from_resampling` for the
continuity-corrected cdf.

For a vector `τ`, e.g. the coefficients of a regression,
`MultivariateEstimatingEquationSaddlePointApprox(psi, x, weights, given)` approximates the
joint density, and its marginals come from a tabulated grid (`fit_pdf`, `marginal_cdf`,
`marginal_ppf`).

### Example scripts

The [`examples`](https://github.com/mvds314/spapprox/tree/main/examples) folder contains
standalone scripts. They need `matplotlib`, and `example_nonparametric_bootstrap.py` also
needs `statsmodels`.

| Script | Shows |
|---|---|
| [`basic_example.py`](https://github.com/mvds314/spapprox/blob/main/examples/basic_example.py) | pdf and cdf of the normal and exponential distributions |
| [`example_sum_random_variables.py`](https://github.com/mvds314/spapprox/blob/main/examples/example_sum_random_variables.py) | sum of independent random variables |
| [`example_parametric_bootstrap.py`](https://github.com/mvds314/spapprox/blob/main/examples/example_parametric_bootstrap.py) | distribution of a sample mean |
| [`example_nonparametric_bootstrap.py`](https://github.com/mvds314/spapprox/blob/main/examples/example_nonparametric_bootstrap.py) | bootstrap of the mean without resampling |
| [`example_bootstrap_multinomial.py`](https://github.com/mvds314/spapprox/blob/main/examples/example_bootstrap_multinomial.py), [`example_bootstrap_dirichlet.py`](https://github.com/mvds314/spapprox/blob/main/examples/example_bootstrap_dirichlet.py) | multinomial and Dirichlet bootstrap (Booth and Butler, 1990) |
| [`example_bootstrap_lattice.py`](https://github.com/mvds314/spapprox/blob/main/examples/example_bootstrap_lattice.py) | continuity-corrected resampling of integer data |
| [`example_bootstrap_robust_estimator.py`](https://github.com/mvds314/spapprox/blob/main/examples/example_bootstrap_robust_estimator.py) | bootstrap of Huber's M-estimator |
| [`example_bootstrap_regression.py`](https://github.com/mvds314/spapprox/blob/main/examples/example_bootstrap_regression.py) | bootstrap of regression coefficients (vector estimating equation) |
| [`example_jackknife_estimating_equation.py`](https://github.com/mvds314/spapprox/blob/main/examples/example_jackknife_estimating_equation.py) | delete-m jackknife |
| [`example_permutation_one_sample.py`](https://github.com/mvds314/spapprox/blob/main/examples/example_permutation_one_sample.py), [`example_permutation_two_sample.py`](https://github.com/mvds314/spapprox/blob/main/examples/example_permutation_two_sample.py) | permutation tests |
| [`example_numdiff_backends.py`](https://github.com/mvds314/spapprox/blob/main/examples/example_numdiff_backends.py) | numerical derivatives of a cgf with `numdifftools` and `findiff` |
| [`example_Wang1990.py`](https://github.com/mvds314/spapprox/blob/main/examples/example_Wang1990.py), [`example_Kolassa2003.py`](https://github.com/mvds314/spapprox/blob/main/examples/example_Kolassa2003.py) | bivariate and multivariate cdf, reproducing published tables |

## Limitations and future work

- Multivariate support is not complete. For example, the Kolassa and Li cdf does not support
  mixed lattice and continuous components.
- `generalized_normal` and `asymmetric_generalized_normal` evaluate the cgf by quadrature
  unless the shape is 1 or 2, which is slow.
- The Kolassa and Li bivariate cdf (the default `backend="KL"`) does not factorize exactly
  for independent components (error up to about 5e-3 in tests). Its extension to more
  than two dimensions has not been checked against the literature yet.
- Without analytic derivatives, cgfs are differentiated numerically, which is slower and
  less accurate, in particular close to the boundary of the domain. The multivariate third
  derivative with `numdifftools` differentiates numerically twice and is slow.
- The pdf is normalized by numerical integration by default, which can be expensive. Pass
  `normalize_pdf=False`, or a known `pdf_normalization` to the constructor, to avoid it.
- The distribution of a quotient of random variables is not implemented yet.

## Related software

- [saddlepoint-expansions](https://github.com/s-broda/saddlepoint-expansions) (Python, also
  on PyPI) by Broda and Kan, a port of the Julia package
  [SaddlepointExpansions.jl](https://github.com/s-broda/SaddlepointExpansions.jl). It
  implements saddle point expansions of arbitrary order for multivariate tail probabilities,
  densities, conditional distributions, ratios and Studentized means, with derivatives from
  automatic differentiation.
- [saddlepoint](https://github.com/godrick/saddlepoint) (R) by Oketch, Fewster and Goodman, a
  framework for saddle point approximations built on cgf operations, including approximate
  likelihood inference.
- [esaddle](https://github.com/mfasiolo/esaddle) (R, on CRAN) by Fasiolo and Wood, the
  extended empirical saddle point density approximation (Fasiolo et al., 2018), which
  estimates a density from a sample of (possibly multivariate) simulated statistics.

## References

- Barndorff-Nielsen, O. E. (1986). Inference on full or partial parameters based on the
  standardized signed log likelihood ratio. *Biometrika*, 73, 307-322.
- Booth, J. G. and Butler, R. W. (1990). Randomization distributions and saddlepoint
  approximations in generalized linear models. *Biometrika*, 77, 787-796.
- Butler, R. W. (2007). *Saddlepoint Approximations with Applications*. Cambridge University
  Press.
- Daniels, H. E. (1983). Saddlepoint approximations for estimating equations. *Biometrika*,
  70, 89-96.
- Fasiolo, M., Wood, S. N., Hartig, F. and Bravington, M. V. (2018). An extended empirical
  saddlepoint approximation for intractable likelihoods. *Electronic Journal of Statistics*,
  12(1), 1544-1578.
- Kolassa, J. E. (1996). Higher-order approximations to conditional distribution functions.
  *Annals of Statistics*, 24, 353-364.
- Kolassa, J. E. (2003). Multivariate saddlepoint tail probability approximations. *Annals
  of Statistics*, 31, 274-286.
- Kolassa, J. E. and Li, J. (2010). Multivariate saddlepoint approximations in tail
  probability and conditional inference. *Bernoulli*, 16, 1191-1207.
- Lugannani, R. and Rice, S. (1980). Saddle point approximation for the distribution of the
  sum of independent random variables. *Advances in Applied Probability*, 12, 475-490.
- Skovgaard, I. M. (1987). Saddlepoint expansions for conditional distributions. *Journal of
  Applied Probability*, 24, 875-887.
- Wang, S. (1990). Saddlepoint approximations for bivariate distributions. *Journal of
  Applied Probability*, 27, 586-597.

## Citation

If you use `spapprox`, please cite it as described in
[`CITATION.cff`](https://github.com/mvds314/spapprox/blob/main/CITATION.cff).

## Development

Clone the repository and install it in editable mode with all extras:

```bash
git clone https://github.com/mvds314/spapprox.git
cd spapprox
pip install -e ".[all]"
```

Run the tests with `pytest`. The full suite takes about 16 minutes. For routine iteration,
skip the slow tests (about 4 minutes):

```bash
pytest -m "not slow"
```

Install the optional dependencies to test all functionality. Lint with `ruff check .`.

## License

BSD 3-Clause, see [`LICENSE`](https://github.com/mvds314/spapprox/blob/main/LICENSE).
