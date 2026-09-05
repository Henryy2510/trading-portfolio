from __future__ import annotations

import pandas as pd

from src.signals.ranking import cross_sectional_rank


def test_cross_sectional_rank():
    feature = pd.DataFrame(
        {
            "A": [1.0, 5.0],
            "B": [2.0, 3.0],
            "C": [3.0, 1.0],
        }
    )
    result = cross_sectional_rank(feature)
    expected = feature.rank(axis=1, pct=True)
    assert result.equals(expected)
