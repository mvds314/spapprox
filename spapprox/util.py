"""
Small utilities used throughout the package.

:class:`Timer` times a block of code, and :func:`type_wrapper` is the decorator that makes the
evaluation methods accept array-like input and return output of the same container type
(pandas in, pandas out; a 0-d result becomes a Python scalar).
"""

import functools as ft
import inspect
import time
from functools import reduce

import numpy as np
import pandas as pd


class Timer:
    """
    Context manager that prints how long a block of code took.

    On entry it prints ``"<msg>..."`` and on exit ``" finished in <seconds> seconds"``.

    Parameters
    ----------
    msg : str, optional
        Message printed on entry. Default is "Running".
    decimals : int, optional
        Number of decimals of the elapsed time. Default is 2.

    Examples
    --------
    >>> import spapprox as spa
    >>> with spa.Timer("Computing", decimals=0):
    ...     x = sum(range(10))
    Computing... finished in 0 seconds

    See ``examples/example_bootstrap_dirichlet.py``.
    """

    def __init__(self, msg="Running", decimals=2):
        self.msg = f"{msg}..."
        self.t0 = 0
        assert isinstance(decimals, int) and decimals >= 0
        self.decimals = decimals

    def __enter__(self):
        print(self.msg, end="")
        self.t0 = time.time()
        return self.t0

    def __exit__(self, ex_type, ex_value, ex_traceback):
        print(f" finished in {time.time() - self.t0:.{self.decimals}f} seconds")
        return False


def fib(n):
    """
    The :math:`n`-th Fibonacci number, with ``fib(0) == 0`` and ``fib(1) == 1``.
    """
    return reduce(lambda x, n: [x[1], x[0] + x[1]], range(n), [0, 1])[0]


class PandasWrapper:
    """
    Wrap a result in the pandas type of an original input.

    Parameters
    ----------
    orig : array_like
        The original input. If it is a :class:`pandas.Series` or :class:`pandas.DataFrame`,
        :meth:`wrap` reuses its index, and its name or columns.
    """

    def __init__(self, orig):
        self._orig = orig
        self._is_pandas = isinstance(orig, (pd.Series, pd.DataFrame))

    def wrap(self, obj):
        """
        Wrap ``obj`` like the original input.

        Parameters
        ----------
        obj : array_like
            The result, of dimension 1 or 2 with as many rows as the original input if that
            was pandas.

        Returns
        -------
        pandas.Series, pandas.DataFrame or numpy.ndarray
            A Series for a 1-d and a DataFrame for a 2-d ``obj`` if the original input was
            pandas, otherwise ``obj`` as an array.
        """
        obj = np.asarray(obj)
        if not self._is_pandas:
            return obj
        if obj.shape[0] != self._orig.shape[0]:
            raise ValueError("obj must have the same number of elements in axis 0 as orig")
        if obj.ndim == 1:
            return pd.Series(obj, name=getattr(self._orig, "name", None), index=self._orig.index)
        elif obj.ndim == 2:
            columns = getattr(self._orig, "columns", None)
            return pd.DataFrame(obj, columns=columns, index=self._orig.index)
        else:
            raise ValueError("Can only wrap 1 or 2-d array_like")


def cov2corr(cov):
    r"""
    Correlation matrix of a covariance matrix.

    Parameters
    ----------
    cov : array_like
        Covariance matrix.

    Returns
    -------
    numpy.ndarray
        The correlation matrix :math:`\Sigma_{ij}/\sqrt{\Sigma_{ii}\Sigma_{jj}}`.
    """
    cov = np.asanyarray(cov)
    std = np.sqrt(np.diag(cov))
    return cov / np.outer(std, std)


def type_wrapper(xloc=0):
    """
    Decorator converting an array-like argument to a numpy array and the result back.

    The argument at position ``xloc`` (or passed by its name) is converted with
    :func:`numpy.asanyarray` before calling the function. The result is wrapped with
    :class:`PandasWrapper`, so pandas input gives pandas output,
    and a 0-d result is returned as a Python scalar. If the argument is not passed, the
    function is called unchanged.

    Parameters
    ----------
    xloc : int, optional
        Position of the argument in the signature, counting ``self`` for methods.

    Returns
    -------
    callable
        The decorator.
    """
    def decorator(f):
        xname = list(inspect.signature(f).parameters)[xloc]

        @ft.wraps(f)
        def wrapped(*args, **kwargs):
            if len(args) > xloc:
                x = args[xloc]
                args = tuple(np.asanyarray(x) if i == xloc else a for i, a in enumerate(args))
            elif xname in kwargs:
                x = kwargs[xname]
                kwargs[xname] = np.asanyarray(x)
            else:
                return f(*args, **kwargs)
            wrapper = PandasWrapper(x)
            y = np.asanyarray(f(*args, **kwargs))
            return y.tolist() if y.ndim == 0 else wrapper.wrap(y)

        return wrapped

    return decorator


def _lagrange_weights(nodes, x):
    """Weights of the Lagrange interpolating polynomial through ``nodes`` at ``x``."""
    nodes = np.asarray(nodes, dtype=float)
    return np.array(
        [
            np.prod([(x - nk) / (nj - nk) for k, nk in enumerate(nodes) if k != j])
            for j, nj in enumerate(nodes)
        ]
    )
