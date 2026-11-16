# `cgf_base.py` -- CGF API reference

The largest module (1919 loc) and the one you will spend most time in.
Class hierarchy:

```
CumulantGeneratingFunction (ABC)
+-- UnivariateCumulantGeneratingFunction
`-- MultivariateCumulantGeneratingFunction
```

## Contents

- [Construction](#construction)
- [The loc/scale contract](#the-locscale-contract)
- [Evaluation methods and the domain-masking pattern](#evaluation-methods-and-the-domain-masking-pattern)
- [Cumulants at zero: raw vs transformed](#cumulants-at-zero-raw-vs-transformed)
- [Numerical differentiation backends](#numerical-differentiation-backends)
- [dK_inv: solving the saddle point equation](#dk_inv-solving-the-saddle-point-equation)
- [Algebra](#algebra)
- [Multivariate-only API](#multivariate-only-api)

## Construction

```python
CumulantGeneratingFunction(
    K,                      # callable, the CGF of the *standardized* variable
    loc=0, scale=1,
    dK=None, dK_inv=None, d2K=None, d3K=None,   # optional analytic derivatives
    dK0=None, d2K0=None, d3K0=None,             # optional cumulants at zero (raw)
    domain=None,            # Domain; defaults to all of R^dim
    numdiff_backend="numdifftools",             # or "findiff"
)
```

`MultivariateCumulantGeneratingFunction` adds `dim=2` as its second positional
parameter and, when `domain is None`, infers the domain dimension from
`len(loc)` or `len(scale)` before falling back to `dim`.

The callables are stored privately as `_K`, `_dK`, `_dK_inv`, `_d2K`, `_d3K`;
the public `K`, `dK`, `dK_inv`, `d2K`, `d3K` are methods that add domain masking
and the loc/scale transform on top. Do not confuse the two -- assigning to
`self._dK` is how lazily-built numerical derivatives get memoized.

Convenience properties: `kappa1`/`mean` (= `dK(0)`), `kappa2` (= `d2K(0)`),
`variance` (univariate: `kappa2`; multivariate: `np.diag(cov)`), `std`, and --
multivariate only -- `cov` (= `d2K0`) and `cor` (via
`statsmodels`' `cov2corr`).

## The loc/scale contract

The object represents $X = \text{scale}\cdot Z + \text{loc}$, where the `K` you
supplied is $K_Z$. Univariate:

$$K_X(t) = K_Z(\sigma t) + \mu t,\quad
K_X'(t) = \sigma K_Z'(\sigma t) + \mu,\quad
K_X''(t) = \sigma^2 K_Z''(\sigma t),\quad
K_X'''(t) = \sigma^3 K_Z'''(\sigma t).$$

Multivariate: `loc` is a vector, `scale` a matrix. A 1-D `scale` is interpreted
as a vector of standard deviations (equivalently a diagonal matrix); a scalar
broadcasts. `K(t) = K_Z(scale^T t) + <loc, t>`. Helper properties `loc_vect`,
`scale_mat`, `scale_inv`, `scale_mat_inv` and `scale_is_invertible` materialize
the general forms and cache them; `_scale_t` / `_inv_scale_t` apply and undo the
scaling to a `t` argument.

The `loc` and `scale` setters **invalidate the transformed cumulant caches**
(`loc` clears `_dK0_cache`; `scale` clears all three) while leaving the raw
caches intact. That asymmetry is the whole point of the raw/transformed split.

Both public `K`/`dK`/`d2K`/`d3K` accept `loc=` and `scale=` overrides. This is
used internally to evaluate the *standardized* function (`loc=0, scale=1`) when
building numerical derivatives, so the transform is applied exactly once.

## Evaluation methods and the domain-masking pattern

Every evaluation method follows this shape:

```python
@type_wrapper(xloc=1)
def K(self, t, fillna=np.nan, loc=None, scale=None):
    loc = self.loc if loc is None else loc
    scale = self.scale if scale is None else scale
    st = scale * t
    cond = self.domain.is_in_domain(st)
    st = np.where(cond, st, 0)          # never let user code see out-of-domain input
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")
        val = self._K(st) + loc * t
        return np.where(cond, val, fillna)
```

Three things are load-bearing:

- `@type_wrapper(xloc=1)` -- pandas containers survive the round trip and 0-d
  results come back as Python scalars.
- The substitution `st = np.where(cond, st, 0)` -- the supplied `K` is often
  `exp`-based and would overflow or emit warnings outside its domain. Evaluating
  at a dummy 0 and discarding the result afterwards is cheaper and safer than
  trying to index around the mask.
- `fillna` -- out-of-domain entries take this value. Callers doing numerical
  integration pass `fillna=0` so the integrand vanishes outside the domain
  instead of poisoning the result with NaN.

## Cumulants at zero: raw vs transformed

| Name | Meaning |
|---|---|
| `_dK0_raw_cache` | user-supplied or computed $K_Z^{(n)}(0)$, set in `__init__` from `dK0=` |
| `_dK0_raw` | property; computes and memoizes the raw value on first access |
| `dK0` | property; applies loc/scale to the raw value, memoized in `_dK0_cache` |

Same pattern for `d2K0` and `d3K0`. In the multivariate case the transform is
the tensor one: `dK0 = scale @ _dK0_raw + loc`,
`d2K0 = scale @ _d2K0_raw @ scale.T`, and `d3K0` goes through
`transform_rank3_tensor` (or elementwise broadcasting when `scale` is 1-D).

Why this matters in practice: `d3K0` is needed by the Lugannani-Rice and
Barndorff-Nielsen cdf approximations at $t=0$. If a CGF cannot produce it --
analytically or numerically -- `cdf` fails at the origin.

The algebra methods check `hasattr(self, "_dK0_cache") or self._dK0_raw_cache is
not None` before propagating cumulants to a derived object. That guard exists so
combining two CGFs never *triggers* an expensive numerical evaluation; it only
reuses values that already happen to be known.

## Numerical differentiation backends

Chosen per object with `numdiff_backend`, validated in `__init__` to be
`"numdifftools"` (default) or `"findiff"`. Construction is lazy -- on the first
call to `dK`, if `self._dK is None`:

```python
if self._numdiff_backend == "findiff":
    self._dK = PartialDerivative(lambda tt: self.K(tt, loc=0, scale=1), 1)
elif self._numdiff_backend == "numdifftools":
    if not has_numdifftools:
        raise ImportError("Numdifftools is required if derivatives are not provided")
    self._dK = nd.Derivative(lambda tt: self.K(tt, loc=0, scale=1), n=1)
```

Multivariate uses `nd.Gradient` / `nd.Hessian` or `diff.Gradient` /
`diff.Hessian` correspondingly. **`MultivariateCumulantGeneratingFunction.d3K`
only has a findiff path** (`diff.Tressian`); the numdifftools branch is an
open TODO. Both multivariate `dK` and `d2K` also zero out any `t` row containing
NaN before calling the backend, with the comment that numdifftools breaks if any
entry evaluates to NaN.

## `dK_inv`: solving the saddle point equation

`dK_inv(x)` solves $K'(t) = x$. If an analytic `dK_inv` was supplied it is used
directly (after undoing loc/scale: `x = (x - loc) / scale`). Otherwise it falls
back to `scipy.optimize` root finding with a **cascade of methods**, trying in
order `halley`, `newton`, `secant`, then the bracketing methods `bisect`,
`brentq`, `brenth`, `ridder`, `toms748`. Derivative-based methods get
`fprime=d2K` and `fprime2=d3K` automatically.

For bracketing methods a valid bracket must be found first, which is done by
walking `-0.9**i` / `0.9**i` until `dK` stops returning NaN and then widening by
Fibonacci-based scaling factors (`1 - 1/fib(i)`, hence `util.fib`). The same
trick appears in `spa.infer_t_range` and `spa.ppf`.

This cascade is why supplying an analytic `dK_inv` in a `cgfs.py` factory is
worth real effort -- it removes per-point root finding from every `pdf(x=...)`
call.

## Algebra

| Method | Operators | Behavior |
|---|---|---|
| `add(other, inplace=False)` | `+`, `-`, `radd`, `rsub` | scalar -> shifts `loc`; CGF -> sum of independent variables, building lazily-composed `K`/`dK`/`d2K`/`d3K` lambdas and intersecting domains |
| `mul(other, inplace=False)` | `*`, `/`, `rmul` | scalar only; scales both `loc` and `scale` |

`__sub__` is `add(-other)`; `__rsub__` is `mul(-1).add(other)`.

Note the asymmetry in `inplace` support: shifting/scaling by a scalar can be
done in place, but combining two CGFs cannot (there is no single `_K` to mutate),
and `UnivariateCumulantGeneratingFunction.add` asserts this. Also note
`MultivariateCumulantGeneratingFunction.add` defaults to `inplace=True` while the
univariate one defaults to `False` -- check before assuming.

The lambdas built by `add` capture `self.scale`/`other.scale`/`self.loc`/
`other.loc` as **default arguments** (`lambda t, ss=self.scale, ...`). That is
deliberate late-binding protection: the parents' `loc`/`scale` may later be
mutated in place, and the derived CGF must keep the values it was built with.

## Multivariate-only API

- `__getitem__(item)` -- marginals. An `int` returns a
  `UnivariateCumulantGeneratingFunction`; a list/slice/tuple returns a
  multivariate one. Implemented by zeroing the other components (Kolassa 2006,
  ch. 6.8), effectively selecting rows of the scale matrix.
- `ldot(A, inplace=False)` -- the linear map $X \mapsto AX$, using
  $K_{AX}(t) = K_X(A^T t)$. A **2-D** `A` gives a multivariate CGF; a **1-D** `A`
  gives a univariate one (inner product $\langle A, X\rangle$) and forbids
  `inplace`. In-place is only allowed when the dimension is preserved.
- `stack(other)` -> `from_cgfs(self, other)` -- concatenate into a longer random
  vector.
- `from_univariate(*cgfs)` (classmethod) -- stack **independent** univariate CGFs,
  using $K_X(t) = \sum_i K_{X_i}(t_i)$.
- `from_cgfs(*cgfs)` (classmethod) -- the general version accepting a mix.

All of these propagate the domain through the corresponding `Domain` method
(`ldot`, `intersect`, `stack`, `from_domains`), which is why `domain.py` carries
an algebra that mirrors this one.
