from __future__ import annotations

import pandas as pd

from src.portfolio.weighting import equal_weight


def test_equal_weight_sums_to_one():
    selection = pd.DataFrame({"A": [1, 0], "B": [1, 1], "C": [0, 1]}, dtype=int)
    weights = equal_weight(selection)
    assert (weights.sum(axis=1).iloc[1] == 1.0)


def test_equal_weight_no_negative():
    selection = pd.DataFrame({"A": [1, 0], "B": [1, 1], "C": [0, 1]}, dtype=int)
    weights = equal_weight(selection)
    assert (weights >= 0).all().all()
