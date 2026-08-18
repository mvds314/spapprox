# `diff.py` and `domain.py` -- reference

Two leaf modules with no knowledge of CGFs. `cgf_base.py` depends on both;
neither depends on anything else in the package except `util`.

---

# `diff.py` -- numerical differentiation

Requires `findiff` **>= 0.11**, imported as `fd` behind a `try/except ImportError`
that sets the module-level `_has_findiff` flag. The package uses findiff's
current `Diff` API; `FinDiff` is deprecated upstream and must not be
reintroduced. Composition rules:

| Derivative | Expression |
|---|---|
| $\partial_0$ | `fd.Diff(0, h, acc=acc)` |
| $\partial_0^2$ | `fd.Diff(0, h, acc=acc) ** 2` |
| $\partial_0 \partial_1$ | `fd.Diff(0, h0, acc=acc) * fd.Diff(1, h1, acc=acc)` |

## `PartialDerivative(f, *orders, dim=None, h=None, acc=2)`

Computes $\partial^{\sum_k \alpha_k}_\alpha f(t)$ where `orders` is the
multi-index $\alpha$ (`PartialDerivative(f, 2, 1)` = twice in the first argument,
once in the second).

The interesting problem it solves: **findiff operates on grids, but CGF
derivatives are needed at arbitrary points.** So `_build_grid(t)` constructs a
small local stencil around `t`, applies the findiff operator, and reads off the
center value. `acc` (default 2, must be >= 2) sets the finite-difference accuracy
order and therefore the stencil width; `h` is the step size (scalar, or one per
component).

Domains are *not* declared to this class -- they are inferred from `f` returning
NaN. That is the contract: **`f` must map out-of-domain points to NaN**, which is
exactly what `cgf_base`'s domain masking guarantees. `_build_grid` exploits this:
it starts with a centered stencil and, if any point evaluates to NaN, **shifts
the stencil left or right** to a one-sided scheme that stays inside the domain.
It raises if the evaluation point itself is NaN, or if neither the left nor the
right half-stencil is fully in-domain. This is what lets derivatives be taken
right up against a domain boundary -- common for CGFs like `gamma`, which is
finite only for $t < 1/\text{scale}$.

`dim=0` means a scalar-valued domain (a univariate CGF); `dim=d` a
$\mathbb{R}^d$ domain. A single-element `orders` defaults to scalar/`dim=0`
unless `dim=1` is passed explicitly -- the class docstring calls this out because
it is the one genuinely surprising bit of the interface.

## `TensorDerivative` and `Gradient` / `Hessian` / `Tressian`

`TensorDerivative(f, dim, order, h=None, acc=2)` assembles the full order-`n`
derivative tensor of shape `(dim,) * order`. Subclasses fix the order:
`Gradient` (1), `Hessian` (2), `Tressian` (3).

The tensor is symmetric, and the implementation leans on that hard. It stores an
object array `_partials` of `PartialDerivative` instances, but **instantiates
only one per sorted index tuple** and aliases all permutations to it, so a
Hessian builds `d(d+1)/2` partials rather than `d^2`. `__call__` likewise
evaluates only `keys(unique=True)`
(`itertools.combinations_with_replacement`) and then broadcasts each value to
every permutation of its index. `__call__` also handles vectorized `t` by
recursing over the leading axis.

Both `h` and `acc` may be given per component: pass a scalar, or an array of
shape `(dim,) * order`.

## `transform_rank3_tensor(T, *A)` and `block_diag_3d(*tensors)`

`transform_rank3_tensor` applies $T_{ijk} \mapsto \sum_{lmn} A_{il}A_{jm}A_{kn}T_{lmn}$
(one matrix reused on all three axes, or three distinct ones). Vectors are
interpreted as diagonal matrices. This exists because a `loc`/`scale` transform
must be pushed through the rank-3 cumulant tensor `d3K0`, and numpy has no
one-liner for it.

`block_diag_3d` is the rank-3 analogue of `scipy.linalg.block_diag`, used when
stacking independent random vectors -- their third cumulant tensor is block
diagonal because cross-terms between independent blocks vanish.

---

# `domain.py` -- where `K` is finite

```python
Domain(dim=1, l=None, g=None, le=None, ge=None, A=None, a=None, B=None, b=None)
```

A point is in the domain when it satisfies **all** of:

- strict/inclusive bounds `l` (less), `le` (less-equal), `ge` (greater-equal),
  `g` (greater) -- required to satisfy the ordering `g > ge > le > l`, asserted in
  the constructor;
- the linear inequalities $Ax \le a$ and $Bx < b$.

For `dim == 1` the bounds must be scalars; for `dim > 1` they may be scalars
(broadcast) or length-`dim` vectors, and the ordering constraint is checked
row-wise via a pandas frame. Properties `l_vect`, `g_vect`, `le_vect`, `ge_vect`
materialize the vector forms; `has_bounds`, `has_lower_bounds`,
`has_upper_bounds`, `has_strict_bounds`, `has_inclusive_bounds` and
`has_ineq_constraints` are cheap predicates used to skip work.

`is_in_domain(t)` (decorated with `type_wrapper`) returns the boolean mask that
`cgf_base.py` uses everywhere; `__contains__` gives the `t in domain` form.

## The algebra mirrors the CGF algebra

This is the key structural point: every operation you can perform on a CGF has a
counterpart here, so domains propagate automatically instead of being recomputed.

| `Domain` method | Corresponding CGF operation |
|---|---|
| `add(other)` / `+`, `-` | shifting a CGF by a constant |
| `mul(other)` / `*` | scaling a CGF |
| `intersect(other)` | `cgf1 + cgf2` -- the sum is finite only where both are |
| `ldot(A)` | `mcgf.ldot(A)` |
| `ldotinv(A)` | the inverse map, used when undoing a scale transform |
| `stack(other)` / `from_domains(*domains)` | stacking CGFs into a random vector |

Transforming a box under a general linear map does not stay a box, which is why
`Domain` carries the `A x <= a` / `B x < b` machinery at all. Be aware that
`ldot` is the weakest link here -- its own docstring notes that `A` need not be
invertible (so the transform is sometimes impossible) and that combining several
bounds into one **can enlarge the domain**. Treat a `ldot`-derived domain as a
superset, not an exact answer.

When adding a distribution to `cgfs.py`, supply a `Domain` whenever the CGF is
not finite on all of $\mathbb{R}$ -- otherwise `PartialDerivative` will happily
build stencils across the singularity and the failure will surface far away as a
NaN or an overflow warning.
