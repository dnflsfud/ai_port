"""Probe: LightGBM 4.6 LGBMRanker early-stopping semantics as used by
src.model_trainer.train_model (production params: rank_xendcg, eval_at [20]).
Checks: which metric ES monitors, n_estimators_ == best_iteration_, predict()
uses best iteration, subsample without subsample_freq is inert."""
import numpy as np, pandas as pd
from src.config import PipelineConfig
from src.model_trainer import train_model

rng = np.random.default_rng(0)
dates = pd.bdate_range("2020-01-01", periods=120)
tickers = [f"T{i:03d}" for i in range(60)]
idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
X = pd.DataFrame(rng.normal(size=(len(idx), 5)), index=idx, columns=[f"f{i}" for i in range(5)])
y = (0.05 * X["f0"] + rng.normal(size=len(idx))).unstack("ticker")

cfg = PipelineConfig(model_objective="cross_sectional_rank", rank_eval_at=[20],
                     lgbm_params={"objective": "rank_xendcg", "metric": "ndcg", "learning_rate": 0.02,
                                  "num_leaves": 31, "max_depth": 5, "min_child_samples": 60,
                                  "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.3,
                                  "reg_lambda": 2.0, "n_estimators": 800, "verbose": -1,
                                  "random_state": 42},
                     early_stopping_rounds=100)
m = train_model(X, y, list(X.columns), dates[:80], dates[80:], config=cfg)
print("evals_result keys:", {k: list(v.keys()) for k, v in m.evals_result_.items()})
print("best_iteration_:", m.best_iteration_, " n_estimators_:", m.n_estimators_,
      " booster trees:", m.booster_.num_trees(), " current_iteration:", m.booster_.current_iteration())
Xp = X.loc[X.index.get_level_values("date") == dates[-1]].values
p_default = m.predict(Xp)
p_best = m.booster_.predict(Xp, num_iteration=m.best_iteration_)
print("predict()==best_iteration predict:", np.allclose(p_default, p_best))
print("bagging_freq in booster params:", m.booster_.params.get("bagging_freq"),
      "subsample_freq attr:", m.get_params().get("subsample_freq"))
