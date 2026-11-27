# Monte Carlo conditional bootstrap

Status: done (#57). Deviations from the plan below: `cdf`/`ppf` take `given` (default 0 for
the pivot, `hat theta_2` otherwise) instead of `w`; `default_B` is clamped to at least 1000;
a uniform kernel is offered next to the Gaussian and Epanechnikov ones; the nonpivotal
interval is the basic interval `2 hat theta_1 - x_q`. The coverage example reproduces BHW
Table 2 on fewer samples and sizes, `examples/example_bootstrap_conditional_coverage.py`. Source: Booth, Hall and Wood (1992), "Bootstrap
estimation of conditional distributions", Annals of Statistics 20, 1594-1610 (BHW).

## Idea

#56 covers BHW Section 4: for a vector mean, the saddle point approximation of the
conditional bootstrap, without resampling. That approximates the *nonpivotal* conditional
bootstrap, error `O_p(n^{-1/2})`. The rest of the paper is about Monte Carlo: the pivotal
(Studentized) conditional bootstrap with error `O_p(n^{-1})`, importance resampling to reduce
its variance, a plug-in bandwidth, and conditional confidence intervals. None of this needs a
cgf, so it goes in a new module, `spapprox/bootstrap.py`, next to the saddle point tools it is
compared with. It depends only on numpy and scipy.

## Setting: smooth functions of means

Data `x` of shape `(n, d)`, `theta_k = g_k(mu)`, `hat theta_k = g_k(xbar)`, `k = 1, 2`. The
gradients `grad g_k` are user supplied or numerical (central differences, no optional
dependency). Influence values `D_kj = n^{-1} (x_j - xbar) . grad g_k(xbar)`, then
`s_k^2 = sum_j D_kj^2`, `eps_kj = D_kj / s_k`, `rho = sum_j eps_1j eps_2j`.

Per resample with counts `M` (shape `(B, n)`), everything is vectorized:
`xbar* = M x / n`, `S* = M (x x^T) / n - xbar* xbar*^T`, and the delta-method
`sigma_k*^2 = grad g_k(xbar*)^T S* grad g_k(xbar*) / n`, `rho*` likewise. That gives

- `V* = (g_1(xbar*) - hat theta_1) / sigma_1*`, `W* = (g_2(xbar*) - hat theta_2) / sigma_2*`
- pivot `U* = (1 - rho*^2)^{-1/2} (V* - rho* W*)` (BHW (2.5)-(2.6))
- nonpivotal: `theta_1*` itself, conditioned on `theta_2*` near `y` (BHW (2.3))

## API

```python
cb = spa.ConditionalBootstrap(x, g1, g2, grad1=None, grad2=None, kernel="gaussian", rng=None)
cb.cdf(u, w=0.0, B=None, h=None, importance=True, pivotal=True)   # p-hat-dagger(u, w)
cb.ppf(alpha, ...)                                                 # u-hat_alpha
cb.ci(level=0.9, sided="two", ...)                                 # BHW Section 2.7
spa.optimal_tilt(u)                                                # t(u), BHW Table 1
spa.conditional_bootstrap_bandwidth(u, rho, B, beta, kernel, w=0.0)  # BHW Section 2.6
```

- `cdf`: the kernel estimate `lambda_1 / lambda_2` of (2.7), weights
  `K((w - W*_b) / h) omega_b`. With `pivotal=False`, `u` and `w` are on the `theta` scale
  (`x` and `y` of (2.3)).
- `importance=True`: draw with `pi_j = exp{-(A_1 eps_1j + A_2 eps_2j) + C}` (2.8),
  `A_1 = (1 - rho^2)^{-1/2} t(u)`, `A_2 = -A_1 rho - w` for general `w`, and
  `omega_b = prod_j (n pi_j)^{-M_bj}`, computed in log space. The tilt depends on `u`, so each
  `u` gets its own resamples; with `importance=False` one set of uniform resamples is shared.
- `optimal_tilt(u)`: minimizes `b(t, u) = [{1 - 2 Phi(u)} Phi(u - t) + Phi(u)^2] e^{t^2}`
  (2.11), with `t(-u) = -t(u)`; also returns the efficiency `e(u)` of (2.12).
- Bandwidth, on the `W` scale (multiply by `sigma_2` on the `theta_2` scale):
  `h = [kappa_2 beta / {4 B phi(w)}]^{1/5} (kappa_1 |psi(u, w)|)^{-2/5}`, with
  `psi(u, w) = {2 rho (1 - rho^2)^{-1/2} w - rho^2 (1 - rho^2)^{-1} u} phi(u)` (5.5) and
  `beta` the minimized (2.9)/(2.10), or `Phi(u)(1 - Phi(u))` for uniform resampling.
  `psi` vanishes at `u = 0` (and for `rho = 0`), where the rule gives `h = inf`: cap `h` at
  the value for `|u| = 0.5`, and at 1. Default `B = ceil(n^{5/2})`, capped at `10^6`: that
  gives `h^4 + (Bh)^{-1} = O(n^{-2})`. Note that the printed condition
  `h^4 + (Bh)^{-1} = O(n^{-1})` only gives `O_p(n^{-1/2})`.
- `ci`: one-sided `(theta_1 - (1 - rho^2)^{1/2} sigma_1 u_alpha, inf)`, two-sided with
  `u_{(1+level)/2}` and `u_{(1-level)/2}`. The quantiles are found by solving `cdf(u) = alpha`
  with the tilt and bandwidth set at `u = Phi^{-1}(alpha)`, as BHW do.

## Steps

1. Influence values, vectorized resample statistics, uniform kernel estimate (pivotal and
   nonpivotal). Test: nonpivotal `q*` against `conditional_cdf` of #56 on a vector mean.
   The exact reference of BHW Section 3: for `X2 ~ Exp(1)`, `X1 = log X2`,
   `theta_1-hat | theta_2-hat =d n^{-1} sum_i log D_i + log theta_2-hat` with `D` uniform on the
   simplex `sum D_i = n` (`n * Dirichlet(1)`), independent of `theta_2-hat`; simulated, it
   is the target of the conditional confidence intervals.
2. `optimal_tilt` and efficiency. Test: BHW Table 1 (all entries to 2 d.p.),
   `t(1.645) = -0.93`, `e(1.645) = 2.85`, `e(1.96) = 4.34`.
3. Importance resampling. Test: same estimate as uniform resampling within Monte Carlo
   error, and smaller variance over repeated runs in the tails (ratio close to `e(u)`).
4. Bandwidth plug-in and default `B`. Test: `h ~ B^{-1/5}`, cap near `u = 0`, kernel
   constants (Gaussian `kappa_1 = 1/2`, `kappa_2 = 1/(2 sqrt(pi))`; Epanechnikov `1/10`, `3/5`).
5. `ppf` and `ci`. Test: one- and two-sided intervals bracket `theta_1` and invert `cdf`.
6. Example `examples/example_bootstrap_conditional_coverage.py`: BHW Table 2, coverage of
   two-sided 90% intervals for `theta_1 = E log X2 = -0.5772` given `theta_2-hat`, `n = 10..50`:
   pivotal bootstrap (`B = 200, 500, 1000`), SP (nonpivotal, #56), normal theory (`t_{n-2}`),
   and here also the Monte Carlo nonpivotal bootstrap. BHW use 1000 samples; the example
   defaults to fewer and is only smoke tested (`slow`, `examples`).
7. Exports in `__init__.py`, README feature list, skill files, `docs/TODO.md`, version bump.

## Out of scope

- Statistics that are not smooth functions of means (would need jackknife influence values).
- Conditioning on more than one statistic (`theta_2` a vector): the kernel and the bandwidth
  rule are scalar in BHW.
- The nonpivotal bandwidth: BHW give no rule (end of Section 5.1); reuse the pivotal one with
  `rho` and warn in the docstring.
