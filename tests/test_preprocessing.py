"""Test dataset preprocessing utilities."""

import pandas as pd

from src.preprocessing.gnn.build_features import compute_load_proxy


def test_compute_load_proxy():
    df = pd.DataFrame({"time": [0, 0, 0, 1, 1], "node_id": [1, 2, 3, 1, 2], "degree": [2, 1, 1, 1, 1]})
    res = compute_load_proxy(df)
    assert "load" in res.columns
    # for time 0, nunique=3, denom = 2. degree for node 1 is 2. 2 / 2 = 1.0
    assert res.loc[0, "load"] == 1.0
    # for time 1, nunique=2, denom = 1. degree for node 1 is 1. 1 / 1 = 1.0
    assert res.loc[3, "load"] == 1.0
