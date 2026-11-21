import numpy as np
import pandas as pd
import pytest

from spapprox import multivariate_norm
from spapprox.util import PandasWrapper, cov2corr


def test_pandas_wrapper():
    s = pd.Series([1.0, 2.0], index=["a", "b"], name="x")
    y = PandasWrapper(s).wrap(np.array([3.0, 4.0]))
    pd.testing.assert_series_equal(y, pd.Series([3.0, 4.0], index=["a", "b"], name="x"))
    df = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=["a", "b"], columns=["u", "v"])
    y = PandasWrapper(df).wrap(np.eye(2))
    pd.testing.assert_frame_equal(y, pd.DataFrame(np.eye(2), index=df.index, columns=df.columns))
    y = PandasWrapper(df).wrap(np.array([5.0, 6.0]))
    pd.testing.assert_series_equal(y, pd.Series([5.0, 6.0], index=["a", "b"]))
    assert isinstance(PandasWrapper([1, 2]).wrap([3, 4]), np.ndarray)
    with pytest.raises(ValueError):
        PandasWrapper(s).wrap(np.zeros(3))
    with pytest.raises(ValueError):
        PandasWrapper(s).wrap(np.zeros((2, 2, 2)))


def test_cov2corr():
    cov = np.array([[4.0, 1.0], [1.0, 9.0]])
    assert np.allclose(cov2corr(cov), [[1.0, 1 / 6], [1 / 6, 1.0]])
    assert np.allclose(multivariate_norm(loc=[0, 0], cov=cov).cor, cov2corr(cov))
