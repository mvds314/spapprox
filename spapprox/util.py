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
from statsmodels.tools.validation import PandasWrapper


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


def type_wrapper(xloc=0):
    """
    Decorator converting an array-like argument to a numpy array and the result back.

    The argument at position ``xloc`` (or passed by its name) is converted with
    :func:`numpy.asanyarray` before calling the function. The result is wrapped with
    :class:`statsmodels.tools.validation.PandasWrapper`, so pandas input gives pandas output,
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
