# Estimating equations and resampling distributions

Status: done (branch `feature/estimating-equations`). Source: Booth and Butler (1990),
"Randomization distributions and saddlepoint approximations in generalized linear models",
Biometrika 77, 787-796.

## Idea

Every resampling scheme in the paper is the distribution of
`T = (T1, T2) = (sum_i Y_i, sum_i z_i Y_i)` for i.i.d. `Y_i`, conditional on `T1 = t1`.
The cgf of `T` follows from the existing algebra
(`MultivariateCumulantGeneratingFunction.from_univariate(...).ldot(...)`), and the conditional
cdf from `BivariateSaddlePointApprox.conditional_cdf` (Skovgaard). So no new cgfs are needed.

What the existing classes do not provide is the distribution of an *estimator*: the root of
`S(tau) = sum_i psi(x_i; tau) W_i = 0`. That is `EstimatingEquationSaddlePointApprox`.

| Scheme | `W_i` | Conditioning | Example |
| ------ | ----- | ------------ | ------- |
| multinomial bootstrap | Poisson(1) | `sum W = n` (a cut) | `example_bootstrap_multinomial.py`, Table 4 |
| Dirichlet bootstrap | exponential(1) | `sum W = n` | `example_bootstrap_dirichlet.py`, Table 4 |
| delete-m jackknife | Bernoulli(1/2) | `sum W = n - m` | `example_jackknife_estimating_equation.py`, Table 3 |
| two-sample permutation | Bernoulli(1/2) | `sum Y = m` | `example_permutation_two_sample.py`, Table 1 |
| one-sample sign flip | +-1 | none | `example_permutation_one_sample.py` |

The permutation tests are not estimating equations; their examples use the algebra and
`conditional_cdf` directly.

## API

`EstimatingEquationSaddlePointApprox(cgf, given=None, decreasing=True, bracket=None)`, with
`cgf(tau)` the cgf of `S(tau)` or of `(T1, S(tau))`, and the convenience constructor
`from_resampling(psi, x, weights, given=None)`.

- `cdf(tau)`: `P(S(tau) <= 0 | T1 = t1)`, Lugannani-Rice or Skovgaard.
- `pdf(tau)`: Daniels (1983), density of `S(tau)` at 0 times the tilted Jacobian
  `E_s[d S / d tau] = d K(s; tau) / d tau / s_S`; normalized over `bracket` by default.
- `fit_cdf`, `ppf`, `mean`, `var`: from the cdf tabulated on `bracket`.

The sample mean is the special case `psi = x - tau`; tests check that it reproduces
`UnivariateSaddlePointApproxMean` (i.i.d.) and, with Poisson weights, the empirical cgf (the
cut theorem of the paper).

## Results

- Table 3 (jackknife): saddle point column reproduced to 4 decimals; mean 8.454 (paper 8.451),
  variance 0.295 (paper 0.300, which uses a midpoint rule with a 0.1 bias correction).
- Table 4: both saddle point columns reproduced to 2 decimals, except the extreme Dirichlet
  tails (p <= 0.001 and p = 0.999), where ours are closer to the Monte Carlo values.
- Table 1 (a): significance level 0.097 and interval (-0.31, 3.25), as in the paper. Table 1 (b)
  does not reproduce, also not for the exact enumeration, see `docs/TODO.md`.

## Not done

- Vector-valued estimating equations.

## Continuity correction (follow-up)

`delta=span` (with a `lattice_point(tau)` callable, set by `from_resampling`) applies the lattice
corrected (conditional) cdf at the largest lattice point of `S(tau)` that is `<= 0` (decreasing
`S`) or `< 0` (increasing `S`). For the multinomial bootstrap of an integer mean it equals, by
the cut, the lattice corrected cdf of the sum of `n` empirical draws, within about 1e-3 of the
exact convolution, see `examples/example_bootstrap_lattice.py`.
