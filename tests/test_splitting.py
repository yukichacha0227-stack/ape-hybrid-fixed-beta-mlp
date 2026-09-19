import pandas as pd

from ape_hybrid.splitting import chronological_roles


def test_roles_are_ordered_and_disjoint():
    times = pd.date_range("2020-01-01", periods=1000, freq="h", tz="UTC")
    roles = chronological_roles(times)
    assert set(roles) == {
        "train_oof",
        "validation_early",
        "validation_calibration",
        "test",
    }
    assert roles["train_oof"].max() < roles["validation_early"].min()
    assert roles["validation_early"].max() < roles["validation_calibration"].min()
    assert roles["validation_calibration"].max() < roles["test"].min()
    all_values = [set(values) for values in roles.values()]
    assert all(not left.intersection(right) for i, left in enumerate(all_values) for right in all_values[i + 1 :])
