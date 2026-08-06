import numpy as np
import pandas as pd


def test_compute_feature_importance_uses_requested_lightgbm_definition():
    import lightgbm as lgb

    from src.attribution import compute_feature_importance

    rng = np.random.default_rng(23)
    feature_names = ["strong", "weak", "noise"]
    x = pd.DataFrame(rng.normal(size=(500, 3)), columns=feature_names)
    y = 2.0 * x["strong"] + 0.2 * x["weak"] + rng.normal(0.0, 0.1, len(x))
    model = lgb.LGBMRegressor(
        n_estimators=35,
        num_leaves=9,
        min_child_samples=5,
        verbose=-1,
        random_state=23,
    ).fit(x, y)

    gain = compute_feature_importance(model, feature_names, importance_type="gain")
    split = compute_feature_importance(model, feature_names, importance_type="split")

    expected_gain = pd.Series(
        model.booster_.feature_importance(importance_type="gain"),
        index=feature_names,
        dtype=float,
    ).sort_values(ascending=False)
    expected_split = pd.Series(
        model.booster_.feature_importance(importance_type="split"),
        index=feature_names,
        dtype=float,
    ).sort_values(ascending=False)
    pd.testing.assert_series_equal(
        gain, expected_gain.rename("gain_importance"), check_exact=True
    )
    pd.testing.assert_series_equal(
        split, expected_split.rename("split_importance"), check_exact=True
    )

