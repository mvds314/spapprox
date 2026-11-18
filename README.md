# Saddle point approximation

UNDER CONSTRUCTION

This module includes code for saddle point approximation

Saddle point approximations can be used to approximate distributions when you know their cumulant generating function.
We follow the theory as outlined in: Ronald W Butler (2007): "Saddlepoint Approximations with Applications"
As he puts it:

> Among the various tools that have been developed for use in statistics and probability
> over the years, perhaps the least understood and most remarkable tool is the saddlepoint
> approximation. It is remarkable because it usually provides probability approximations
> whose accuracy is much greater than the current supporting theory would suggest.

For a random variable $X$, cumulant generating function $K(t)$ is given by:
$$K(t) = \log\text{E} \exp(t X).$$
Roughly, it contains all information about the random variable's moments, particularly $K'(0)=E X$.
For common distributions (normal, exponential, gamma etc.), the cumulant generating function is known.
Some are implemented as part of this software, others you can find on, e.g., Wikipedia.

Why is this useful? Now, it's not always straightforward to find the distribution of, e.g., the sum of (independent) random variables, as you can't simply sum
the distribution functions. You can, however, sum their cumulant generating functions!
The saddle point approximation can, subsequently be used to approximate the pdf and cdf given the cumulant generating function.

Altogether, applications include:

- Approximating the distribution of the sum of random variables
- Approximating the distribution of the mean of random variables
- Bootstrapping the transform domain: approximating the (bootstrap) distribution of the mean (or some other statistic) on some sample without actually bootstrapping.
- Resampling distributions of estimating equations: multinomial and Dirichlet bootstrap, delete-m jackknife, and permutation tests (Booth and Butler, 1990).

To be implemented are:

- Approximating conditional distributions
- Approximating the distribution of the quotient of random variables

For a discrete (lattice) variable, pass the span of the lattice, e.g., `cdf(x, delta=1)` or `pdf(x, delta=1)`, to get the continuity-corrected cdf or the probability mass function.

## Basic example

To understand what is going on, it is helpful to read the [Wikipedia page](https://en.wikipedia.org/wiki/Saddlepoint_approximation_method) on the saddle point approximation.
The basic steps to use this package are, define the cumulant generating function in terms of the class in this packages, feed it to the saddle point approximation class, evaluate the pdf or cdf.
It's also useful to understand what the saddle point equation is: it's an equation you have to solve to solve for $t$ given $x$.

```python
import numpy as np
import scipy.stats as sps
import spapprox as spa

cgf_normal = spa.norm(loc=0, scale=1)
spa_normal = spa.UnivariateSaddlePointApprox(cgf_normal)

t = np.linspace(-3, 3)
x = cgf_normal.dK(t)
spa_normal.pdf(t=t)
sps.norm.pdf(x)
```

## Conditional distributions

For a random vector, the distribution of one component given the others,
$P(X_i \leq x \mid X_J = x_J)$, is approximated with the double saddle point approximation of Skovgaard (1987),
or the second order version of Kolassa (1996) with `backend="Kolassa"`.
Use `delta=1` for a discrete variable of interest.
For two variables of interest, pass `i=[a, b]` to get
$P(X_a \leq x_a, X_b \leq x_b \mid X_J = x_J)$, following Kolassa and Li (2010).

```python
cgf = spa.multivariate_norm(loc=[0, 0], cov=[[1, 0.5], [0.5, 1]])
spa_mvn = spa.MultivariateSaddlePointApprox(cgf)
spa_mvn.conditional_cdf([-1, 0, 1], given=[0.3], i=0)
spa_mvn.conditional_pdf([-1, 0, 1], given=[0.3], i=0)  # normalized, normalize_pdf=False to skip
```

## Estimating equations and resampling

The root $\hat\tau$ of a randomly weighted estimating equation $\sum_i \psi(x_i;\tau) W_i = 0$, optionally conditional on $\sum_i W_i$, is approximated with `EstimatingEquationSaddlePointApprox` (Daniels, 1983; Booth and Butler, 1990).
The weights select the resampling scheme: Poisson weights given their sum $n$ give the multinomial bootstrap, exponential weights the Dirichlet bootstrap, and Bernoulli weights given $n-m$ the delete-$m$ jackknife.

```python
x = np.array([9.6, 10.4, 13.0, 15.0, 16.6, 17.2, 17.3, 21.8, 24.0, 33.8])
ee = spa.EstimatingEquationSaddlePointApprox.from_resampling(
    lambda x, tau: x - tau, x, spa.exponential(), given=len(x), bracket=(x.min(), x.max())
)
ee.cdf([15, 18, 21])
ee.ppf([0.05, 0.95])
```

`cdf` and `ppf` solve saddle point equations for every value. After `ee.fit_cdf()`, pass `interpolate=True` to interpolate the tabulated cdf instead: much faster, accurate up to the resolution of the table.

For a vector $\tau$, e.g., the coefficients of a regression, `MultivariateEstimatingEquationSaddlePointApprox(psi, x, weights, given)` approximates the joint density, with marginals from a tabulated grid (`fit_pdf`, `marginal_cdf`, `marginal_ppf`); see `examples/example_bootstrap_regression.py`.

For a discrete $S(\tau)$, e.g., a mean of integer data under the multinomial bootstrap or the jackknife, pass `delta=1` to `from_resampling` for the continuity-corrected cdf.

See the `examples/example_bootstrap_*.py`, `example_permutation_*.py` and `example_jackknife_*.py` scripts.

## Installation

You can install this library directly from GitHub:

```bash
pip install git+https://github.com/mvds314/spapprox.git
```

Optional dependencies are `numdifftools` (the default numerical differentiation backend), `findiff` (an alternative, faster finite difference backend, version >= 0.11), and `fastnorm` for faster bivariate normal cdf evaluation.
Install them as extras, e.g.:

```bash
pip install "spapprox[findiff,numdiff,fastnorm] @ git+https://github.com/mvds314/spapprox.git"
```

## Development

For development purposes, clone the repo:

```bash
git clone https://github.com/mvds314/spapprox.git
```

Then navigate to the folder containing `pyproject.toml` and run

```bash
pip install -e ".[findiff,numdiff,fastnorm]"
```

to install the package in edit mode.

Run unittests with `pytest`.
The full suite takes about 11 minutes; for routine iteration, skip the slow tests:

```bash
pytest -m "not slow"
```

Install the optional dependencies to test all functionality.
