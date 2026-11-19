r"""
The multivariate tail approximation of Kolassa and Li (2010) in any dimension.

:func:`_kl_cdf` approximates :math:`F(\mathbf{x})=P(\mathbf{X}\leq\mathbf{x})` from the
cumulant generating function :math:`K` of :math:`\mathbf{X}`, by reflection and
inclusion-exclusion over orthants in which the saddlepoint is non-negative, where the
orthant probabilities are approximated by :class:`_KLOrder`, blended over the orders of the
variables. See ``docs/multivariate_cdf_plan.md`` (phase B) for the derivation, and
:meth:`spapprox.spa.MultivariateSaddlePointApprox._kl_cdf` for a summary.

References
----------
[1] Kolassa and Li (2010) - Multivariate saddlepoint approximations in tail probability and
conditional inference

[2] Kolassa (2003) - Multivariate saddlepoint tail probability approximations
"""

import itertools

import numpy as np
import scipy.stats as sps

from .mvn import _kolassa_orthant, _mvn_logtail
from .util import _lagrange_weights

# Below this weight, an order of the variables is not evaluated
_min_order_weight = 1e-12
# Relative distance to the minimizer below which dtau/dw is obtained by expansion
_expansion_width = 1e-4
_snap_width = 1e-7
_nodes = np.array([-2.0, -1.0, 1.0, 2.0])


def _subsets(s):
    s = tuple(s)
    return itertools.chain.from_iterable(itertools.combinations(s, k) for k in range(len(s) + 1))


def _order_weights(z2):
    r"""
    Weights of the orders of the variables: the first variable is :math:`j` with weight
    :math:`z_j^{-2}/\sum_k z_k^{-2}`, and the others are ordered recursively in the same way.
    """
    m = len(z2)
    out = {}

    def first(rem):
        zz = z2[list(rem)]
        if (zz > 0).all():
            p = (1 / zz) / np.sum(1 / zz)
        else:
            p = (zz == 0) / np.sum(zz == 0)
        return p

    def rec(prefix, rem, w):
        if w < _min_order_weight:
            return
        if not rem:
            out[prefix] = w
            return
        for k, p in zip(rem, first(rem)):
            if p > 0:
                rec(prefix + (k,), tuple(r for r in rem if r != k), w * p)

    rec((), tuple(range(m)), 1.0)
    total = sum(out.values())
    return {k: v / total for k, v in out.items()}


class _KLSolver:
    r"""
    Minimizers of :math:`L(\tau)=K(\tau)-\tau\cdot\mathbf{x}` over the components ``A`` of
    :math:`\tau` that are not fixed, with the components outside ``A`` equal to zero,
    memoized. These are the saddlepoints of the subvectors and the nested minimizers of
    :class:`_KLOrder`.
    """

    def __init__(self, cgf, x, max_iter=100):
        self.cgf, self.x = cgf, np.asarray(x, dtype=float)
        self.d = len(self.x)
        self.max_iter = max_iter
        self._cache = {}
        self._d2K = {}
        self._dK = {}
        self._K = {}

    def L(self, A, te):
        key = tuple(te)
        if key not in self._K:
            self._K[key] = float(self.cgf.K(te))
        return self._K[key] - te[list(A)] @ self.x[list(A)]

    def dK(self, te):
        key = tuple(te)
        if key not in self._dK:
            self._dK[key] = np.asarray(self.cgf.dK(te), dtype=float)
        return self._dK[key]

    def d2K(self, te):
        key = tuple(te)
        if key not in self._d2K:
            self._d2K[key] = np.asarray(self.cgf.d2K(te), dtype=float)
        return self._d2K[key]

    def register(self, A, fixed, te, ell):
        key = (tuple(A), tuple(sorted(fixed.items())))
        self._cache.setdefault(key, (te, ell))

    def solve(self, A, fixed, start=None):
        """The minimizer (embedded in the full vector) and the minimum, NaN if not found."""
        key = (tuple(A), tuple(sorted(fixed.items())))
        if key not in self._cache:
            self._cache[key] = self._solve(A, fixed, start)
        return self._cache[key]

    def _solve(self, A, fixed, start):
        free = [i for i in A if i not in fixed]
        te = np.zeros(self.d)
        for i, v in fixed.items():
            te[i] = v
        if not free:
            ell = self.L(A, te)
            return (te, ell) if np.isfinite(ell) else (np.full(self.d, np.nan), np.nan)
        starts = [] if start is None else [np.asarray(start, dtype=float)[free]]
        starts.append(np.zeros(len(free)))
        for s in starts:
            te[free] = s
            res = self._newton(A, free, te.copy())
            if res is not None:
                return res
        # Continuation from the fixed components equal to zero
        cur = self._newton(A, free, np.zeros(self.d))
        if cur is None:
            return np.full(self.d, np.nan), np.nan
        s, step = 0.0, 0.25
        while s < 1:
            sn = min(s + step, 1.0)
            tn = cur[0].copy()
            for i, v in fixed.items():
                tn[i] = sn * v
            new = self._newton(A, free, tn)
            if new is None:
                step /= 2
                if step < 1e-4:
                    return np.full(self.d, np.nan), np.nan
                continue
            s, cur, step = sn, new, 2 * step
        return cur

    def _newton(self, A, free, te):
        f = self.L(A, te)
        tol = 1e-12 * (1 + np.max(np.abs(self.x[free])))
        for _ in range(self.max_iter):
            if not np.isfinite(f):
                return None
            g = self.dK(te)[free] - self.x[free]
            if not np.isfinite(g).all():
                return None
            if np.max(np.abs(g)) < tol:
                return te, f
            H = self.d2K(te)[np.ix_(free, free)]
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                return None
            s = 1.0
            while s > 1e-12:
                new = te.copy()
                new[free] -= s * step
                fn = self.L(A, new)
                if np.isfinite(fn) and fn <= f - 1e-4 * s * (g @ step) + 1e-14 * (1 + abs(f)):
                    break
                s /= 2
            else:
                # No more progress possible, accept if nearly stationary
                return (te, f) if np.max(np.abs(g)) < 1e3 * tol else None
            te, f = new, fn
        return None


class _KLOrder:
    r"""
    The approximation of Kolassa and Li (2010) of the orthant probability
    :math:`P(\sigma_jX_j\geq\sigma_jx_j,\ j\in A)` with the variables of ``A`` in the order
    ``order``, for a saddlepoint ``tau`` of :math:`\mathbf{X}_A` with
    :math:`\sigma\odot\hat\tau\geq0` (except at interpolation nodes). All quantities are in
    the reflected coordinates :math:`\tau_p = \sigma_p t_{\text{order}[p]}`, indexed by the
    position ``p`` in the order. See ``docs/multivariate_cdf_plan.md``, phase B.
    """

    def __init__(self, solver, A, order, sg, tau, weight, rho, h):
        self.solver, self.cgf, self.x = solver, solver.cgf, solver.x
        self.A = tuple(A)
        self.order = list(order)
        self.s = np.asarray(sg, dtype=float)
        self.m = len(self.order)
        self.weight, self.rho, self.width = weight, rho, h
        self.tau_hat = self.s * np.asarray(tau, dtype=float)
        self._G, self._w = {}, {}
        te = self.full(self.tau_hat)
        self.ell_hat = solver.L(self.A, te)
        self._seed(te, self.ell_hat, 0)
        self.w_hat = np.empty(self.m)
        for j in range(self.m):
            p0 = self.prof((0.0,) * j)
            p1 = self.prof((0.0,) * (j + 1))
            self.w_hat[j] = np.sign(p0[2][j]) * np.sqrt(max(2 * (p1[1] - p0[1]), 0.0))

    def full(self, tau):
        te = np.zeros(self.solver.d)
        te[self.order[: len(tau)]] = self.s[: len(tau)] * np.asarray(tau, dtype=float)
        return te

    def _seed(self, te, ell, k):
        """Register ``te`` as the minimizer for all prefixes of its own components from ``k``."""
        for kk in range(k + 1, self.m + 1):
            fixed = {i: te[i] for i in self.order[:kk]}
            self.solver.register(self.A, fixed, te, ell)

    def prof(self, prefix):
        """Nested minimizer for the fixed ``prefix``: (full vector, minimum, reflected tau)."""
        prefix = tuple(prefix)
        k = len(prefix)
        fixed = {self.order[p]: self.s[p] * prefix[p] for p in range(k)}
        start = self.full(self.tau_hat)
        te, ell = self.solver.solve(self.A, fixed, start=start)
        if k < self.m and not np.isnan(ell):
            self._seed(te, ell, k)
        return te, ell, self.s * te[self.order]

    def schur(self, te, k):
        r"""Schur complement of the reflected :math:`K''(te)` onto the first ``k`` positions."""
        H = self.solver.d2K(te)[np.ix_(self.order, self.order)] * np.outer(self.s, self.s)
        if k == self.m:
            return H
        return H[:k, :k] - H[:k, k:] @ np.linalg.solve(H[k:, k:], H[k:, :k])

    def w(self, j, prefix, v):
        r"""Signed root :math:`w_j(\tau^{j-1}, v)`."""
        key = (j, tuple(prefix), v)
        if key not in self._w:
            te0, ell0, tt = self.prof(prefix)
            delta = v - tt[j]
            S0 = self.schur(te0, j + 1)[j, j] if delta != 0 else 0.0
            if abs(delta) * np.sqrt(S0) < _snap_width:
                # Rounding level differences, e.g., between minimizers of the same problem
                # shared by two orders, for which the square root below is inaccurate
                val = self.w_hat[j] + delta * np.sqrt(S0)
            else:
                ell1 = self.prof(tuple(prefix) + (v,))[1]
                val = self.w_hat[j] + np.sign(v - tt[j]) * np.sqrt(max(2 * (ell1 - ell0), 0.0))
            self._w[key] = val
        return self._w[key]

    def u(self, j, prefix, v):
        return self.w(j, prefix, v) - self.w(j, prefix, 0.0)

    def dtau_dw(self, j, prefix, v):
        prefix = tuple(prefix)
        te0, ell0, tt = self.prof(prefix)
        te1, ell1, _ = self.prof(prefix + (v,))
        S = self.schur(te1, j + 1)[j, j]
        if v == tt[j]:
            return 1 / np.sqrt(S)
        S0 = self.schur(te0, j + 1)[j, j]
        delta = v - tt[j]
        if abs(delta) * np.sqrt(S0) < _expansion_width:
            return (1 - (S - S0) / (3 * S0)) / np.sqrt(S0)
        oj = self.order[j]
        dl = self.s[j] * (self.solver.dK(te1)[oj] - self.x[oj])
        return np.sign(delta) * np.sqrt(max(2 * (ell1 - ell0), 0.0)) / dl

    def phi(self, j, prefix, v):
        if v == 0:
            return 1.0
        rho = self.rho(v, self.order[j])
        return self.u(j, prefix, v) / rho * self.dtau_dw(j, prefix, v)

    def G(self, tau):
        r""":math:`G(\tau)=r(\tau)\prod_j\phi_j(\tau^j)`, with :math:`G(0)=r(0)`."""
        key = tuple(tau)
        if key not in self._G:
            val = self.weight(self.full(tau))
            for j in range(self.m):
                if tau[j] != 0:
                    val *= self.phi(j, tau[:j], tau[j])
            self._G[key] = val
        return self._G[key]

    def h(self, A, tau):
        r""":math:`h_A(\tau)=\sum_{S\subseteq A}(-1)^{|A-S|}G(\tau^S)/\prod_{j\in A}u_j(\tau)`."""
        num = 0.0
        for S in _subsets(A):
            ts = tuple(tau[j] if j in S else 0.0 for j in range(self.m))
            num += (-1) ** (len(A) - len(S)) * self.G(ts)
        den = np.prod([self.u(j, tau[:j], tau[j]) for j in A])
        return num / den

    def coef(self, A, R):
        r"""
        :math:`c_{A,R}`: :math:`h_A` with the poles in ``R`` removed by nested differences
        (largest position innermost), the other poles at :math:`\tau_j=0`. A removable
        singularity at a nested minimizer :math:`\tilde\tau_j\approx0` is interpolated.
        """
        memo = {}

        def rec(prefix):
            if prefix in memo:
                return memo[prefix]
            j = len(prefix)
            if j == self.m:
                val = self.h(A, prefix)
            elif j not in A and j not in R:
                val = rec(prefix + (0.0,))
            else:
                tt = self.prof(prefix)[2][j]
                if j in R:

                    def f(v):
                        return (rec(prefix + (v,)) - rec(prefix + (0.0,))) / self.u(j, prefix, v)

                else:

                    def f(v):
                        return rec(prefix + (v,))

                hj = self.width[j]
                if abs(tt) < hj / 2:
                    nodes = _nodes * hj
                    val = _lagrange_weights(nodes, tt) @ [f(v) for v in nodes]
                else:
                    val = f(tt)
            memo[prefix] = val
            return val

        return rec(())

    def jets(self):
        r"""
        :math:`\hat u`, and the gradients ``c[j]`` and Hessians ``Hc[j]`` of
        :math:`\tilde w_j(w^{j-1})` at :math:`\hat w^{j-1}`.
        """
        m, s = self.m, self.s
        te = self.full(self.tau_hat)
        K2 = self.solver.d2K(te)[np.ix_(self.order, self.order)] * np.outer(s, s)
        K3 = np.asarray(self.cgf.d3K(te), dtype=float)[np.ix_(self.order, self.order, self.order)]
        K3 = K3 * np.einsum("i,j,k->ijk", s, s, s)
        Hs, l3 = [], []
        for k in range(m + 1):
            V = np.zeros((m, k))
            V[:k] = np.eye(k)
            if k < m:
                V[k:] = -np.linalg.solve(K2[k:, k:], K2[k:, :k])
            Hs.append(V.T @ K2 @ V)
            l3.append(np.einsum("pqr,pa,qb,rc->abc", K3, V, V, V))
        # 2-jets of s_k = w_k - w_hat_k as functions of tau^k
        g1, g2 = [], []
        for k in range(m - 1):
            H = Hs[k + 1]
            g = H[:, k] / np.sqrt(H[k, k])
            D3 = l3[k + 1].copy()
            D3[:k, :k, :k] -= l3[k]
            s2 = np.zeros((k + 1, k + 1))
            s2[k, k] = D3[k, k, k] / (3 * g[k])
            for c in range(k):
                s2[k, c] = s2[c, k] = (D3[k, k, c] - g[c] * s2[k, k]) / (2 * g[k])
            for b in range(k):
                for c in range(b, k):
                    s2[b, c] = s2[c, b] = (D3[k, b, c] - g[b] * s2[k, c] - g[c] * s2[k, b]) / g[k]
            g1.append(g)
            g2.append(s2)
        u_hat = np.empty(m)
        u_hat[0] = self.w_hat[0]
        c, Hc = [np.zeros(0)], [np.zeros((0, 0))]
        for j in range(1, m):
            Mt = np.zeros((j, j))
            for k in range(j):
                Mt[k, : k + 1] = g1[k]
            B = np.linalg.inv(Mt)
            T2 = np.zeros((j, j, j))
            for k in range(j):
                s2 = np.zeros((j, j))
                s2[: k + 1, : k + 1] = g2[k]
                T2 -= np.einsum("i,pq->ipq", B[:, k], B.T @ s2 @ B)
            tep, ellp, _ = self.prof(tuple(self.tau_hat[:j]) + (0.0,))
            oj = self.order[:j]
            E1 = s[:j] * (self.solver.dK(tep)[oj] - self.x[oj])
            E2 = self.schur(tep, j + 1)[:j, :j] - Hs[j]
            uj = np.sign(self.tau_hat[j]) * np.sqrt(max(2 * (ellp - self.ell_hat), 0.0))
            u_hat[j] = uj
            F1 = -E1 / uj
            F2 = -E2 / uj + np.outer(E1, E1) / uj**3
            c.append(F1 @ B)
            Hc.append(B.T @ F2 @ B + np.einsum("a,apq->pq", F1, T2))
        return u_hat, c, Hc

    def gaussian(self, P, u_hat, c, Hc):
        """The Gaussian integral :math:`N_P` with the poles ``P``."""
        m = self.m
        Bs = [j for j in range(m) if j not in P]
        Ps = list(P)
        if not Ps:
            return np.exp(np.sum(sps.norm.logpdf(self.w_hat)))
        M = np.zeros((m, m))
        for j in range(m):
            M[j, j] = 1.0
            if j in P:
                M[j] += c[j] @ M[:j]
        A = M.T @ M
        v = np.where([j in P for j in range(m)], u_hat, self.w_hat)
        logC = -self.w_hat @ self.w_hat / 2 + v @ A @ v / 2
        if not Bs:
            # I^emptyset with the cubic correction
            W = []
            for j in range(m):
                Wj = np.einsum("lm,la,mb->ab", Hc[j], M[:j], M[:j])
                for ll in range(j):
                    Wj = Wj + c[j][ll] * W[ll]
                W.append(Wj)
            W = np.array(W)
            g3 = np.einsum("ja,jbc->abc", M, W)
            g3 = g3 + np.einsum("jb,jac->abc", M, W) + np.einsum("jc,jab->abc", M, W)
            return _kolassa_orthant(-self.w_hat @ self.w_hat / 2, A, g3, u_hat, 0.0)
        b = A @ v
        Abb = A[np.ix_(Bs, Bs)]
        Abp = A[np.ix_(Bs, Ps)]
        Sch = A[np.ix_(Ps, Ps)] - Abp.T @ np.linalg.solve(Abb, Abp)
        bb = b[Bs]
        logphi = -bb @ np.linalg.solve(Abb, bb) / 2 - np.linalg.slogdet(2 * np.pi * Abb)[1] / 2
        return np.exp(logC + logphi + _mvn_logtail(Sch @ u_hat[Ps], Sch))

    def tail(self):
        """The orthant probability, NaN if a nested minimizer does not exist."""
        if not np.isfinite(self.w_hat).all():
            return np.nan
        try:
            return self._tail()
        except np.linalg.LinAlgError:
            return np.nan

    def _tail(self):
        u_hat, c, Hc = self.jets()
        if not (np.isfinite(u_hat).all() and all(np.isfinite(cj).all() for cj in c)):
            return np.nan
        U = tuple(range(self.m))
        total = 0.0
        for P in _subsets(U):
            Bset = tuple(j for j in U if j not in P)
            a = 0.0
            for A in _subsets(Bset):
                if not A:
                    if len(Bset) == 0:
                        a += self.G((0.0,) * self.m)
                    continue
                R = tuple(j for j in Bset if j not in A)
                a += self.coef(set(A), set(R))
            if a != 0:
                total += self.gaussian(set(P), u_hat, c, Hc) * a
        return total


def _kl_orthant(solver, A, sg, tau, weight, rho, h, sd):
    r"""
    The approximation of :math:`P(\sigma_jX_j\geq\sigma_jx_j,\ j\in A)`, blended over the
    orders of the variables, and interpolated in the components of the saddlepoint ``tau``
    that are close to zero.
    """
    A = tuple(A)
    tau = np.asarray(tau, dtype=float)
    weights = _order_weights(np.square(tau * sd[list(A)]))
    hA = h[list(A)]
    near = np.abs(tau) < hA

    def ordered(perm, tt, slv):
        order = [A[p] for p in perm]
        hw = hA[list(perm)]
        return _KLOrder(slv, A, order, sg[list(perm)], tt[list(perm)], weight, rho, hw).tail()

    def blend(tt, slv):
        # Orders for which a nested minimizer is outside the domain are dropped
        vals = {perm: ordered(perm, tt, slv) for perm in weights}
        ok = [perm for perm in weights if np.isfinite(vals[perm])]
        if not ok:
            return np.nan
        return sum(weights[p] * vals[p] for p in ok) / sum(weights[p] for p in ok)

    if not near.any():
        return blend(tau, solver)
    nodes, lw = [], []
    for tj, hj, nj in zip(tau, hA, near):
        if nj:
            nodes.append(_nodes * hj)
            lw.append(_lagrange_weights(nodes[-1], tj))
        else:
            nodes.append(np.array([tj]))
            lw.append(np.ones(1))
    total = 0.0
    for combo in itertools.product(*[range(len(n)) for n in nodes]):
        coef = np.prod([lw[k][i] for k, i in enumerate(combo)])
        tn = np.array([nodes[k][i] for k, i in enumerate(combo)])
        te = np.zeros(solver.d)
        te[list(A)] = tn
        xn = solver.x.copy()
        xn[list(A)] = np.asarray(solver.cgf.dK(te), dtype=float)[list(A)]
        slv = _KLSolver(solver.cgf, xn)
        slv.register(A, {}, te, slv.L(A, te))
        total += coef * blend(tn, slv)
    return total


def _kl_cdf(cgf, x, t, marginal_cdf, weight=None, rho=None, width=1e-2):
    r"""
    The approximation of :math:`P(\mathbf{X}\leq\mathbf{x})` of Kolassa and Li (2010) in any
    dimension, see :meth:`spapprox.spa.MultivariateSaddlePointApprox._kl_cdf`.

    Parameters
    ----------
    cgf : MultivariateCumulantGeneratingFunction
        The cumulant generating function, with ``K``, ``dK``, ``d2K``, ``d3K`` and ``d2K0``.
    x, t : array_like
        The point, and the saddlepoint at ``x``.
    marginal_cdf : callable
        ``marginal_cdf(i, x_i, t_i)``, the one-dimensional cdf of component ``i``.
    weight : callable, optional
        Weight :math:`r(\tau)` of the inversion integrand, at the full unreflected
        :math:`\tau`; 1 if not given.
    rho : callable, optional
        ``rho(tau, i)`` replacing :math:`\tau_i` in the factors :math:`1/\tau_i` (lattice);
        :math:`\tau` if not given.
    width : float, optional
        Half width of the interpolation near singularities, in standard deviations.
    """
    x, t = np.asarray(x, dtype=float), np.asarray(t, dtype=float)
    d = len(x)
    if not (np.isfinite(x).all() and np.isfinite(t).all()):
        return np.nan
    weight = (lambda tt: 1.0) if weight is None else weight
    rho = (lambda tau, i: tau) if rho is None else rho
    sd = np.sqrt(np.diag(np.asarray(cgf.d2K0, dtype=float).reshape(d, d)))
    h = width / sd
    solver = _KLSolver(cgf, x)
    full = tuple(range(d))
    # The nested differences amplify any inconsistency between t and x, so polish t
    res = solver._newton(full, list(full), t.copy())
    if res is not None:
        t = res[0]
    solver.register(full, {}, t, solver.L(full, t))
    cache = {}

    def saddle(A):
        if len(A) == d:
            return t
        te_, ell = solver.solve(A, {}, start=t)
        return te_[list(A)]

    def tail(A, sg):
        if not A:
            return 1.0
        key = (A, tuple(sg))
        if key in cache:
            return cache[key]
        tau = saddle(A)
        if np.isnan(tau).any():
            val = np.nan
        elif len(A) == 1:
            F = marginal_cdf(A[0], x[A[0]], tau[0])
            val = F if sg[0] < 0 else 1 - F
        else:
            neg = np.flatnonzero(sg * tau < 0)
            if len(neg) == 0:
                val = _kl_orthant(solver, A, sg, tau, weight, rho, h, sd)
            else:
                j = neg[0]
                flipped = sg.copy()
                flipped[j] *= -1
                val = tail(A[:j] + A[j + 1 :], np.delete(sg, j)) - tail(A, flipped)
        cache[key] = val
        return val

    return tail(tuple(range(d)), -np.ones(d))
