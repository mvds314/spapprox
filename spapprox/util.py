# -*- coding: utf-8 -*-

import time
import functools as ft
import numpy as np
from statsmodels.tools.validation import PandasWrapper
from functools import reduce


class Timer:
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
        print(f" finished in {time.time()-self.t0:.{self.decimals}f} seconds")
        return False


# Fibonacci sequence
def fib(n):
    return reduce(lambda x, n: [x[1], x[0] + x[1]], range(n), [0, 1])[0]


def type_wrapper(xloc=0):
    def decorator(f):
        @ft.wraps(f)
        def wrapped(*args, **kwargs):
            x = args[xloc]
            wrapper = PandasWrapper(x)
            args = tuple(np.asanyarray(x) if i == xloc else a for i, a in enumerate(args))
            y = np.asanyarray(f(*args, **kwargs))
            return y.tolist() if y.ndim == 0 else wrapper.wrap(y)

        return wrapped

    return decorator
