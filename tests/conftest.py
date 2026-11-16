import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def returns():
    """Deterministic synthetic daily returns on business days, 2016-2021 (spans a 'holdout')."""
    idx = pd.bdate_range("2016-01-01", "2021-06-30")
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0, 0.01, len(idx)), index=idx, name="r")
