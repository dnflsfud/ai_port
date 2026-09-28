"""D-probe 5: the production-ON S13.41 option-IV covariance channel degrades to
"inert" with only a log warning when the iv30_z sheet (or its observed mask) is
absent, and nothing downstream fails or gates on it.

(a) UniverseData's O8 essential-sheet guard does not list iv30_z.
(b) run_backtest's S13.41 branch turns a missing sheet into logger.warning only.
(c) The export mirror returns None -> risk.json/TE audit report applied=False.
(d) The bundle validator never reads option_vol_cov_scaling_applied.
"""
import inspect
import re

import numpy as np
import pandas as pd
import yaml

from src.data_loader import UniverseData
from src import backtest
from src.harness import build_override_config
from scripts.export_operating_data import _load_optvol_scale, _apply_optvol_scale

src_ud = inspect.getsource(UniverseData)
ess = re.search(r"ESSENTIAL_SHEETS = \{(.*?)\}", src_ud, re.S).group(1)
print("(a) iv30_z in UniverseData ESSENTIAL_SHEETS:", "iv30_z" in ess)

src_bt = inspect.getsource(backtest.run_backtest)
print("(b) run_backtest missing-iv30_z branch is warning-only:",
      "except KeyError:" in src_bt and "diagonal scaling stays inert" in src_bt
      and "raise" not in src_bt.split("diagonal scaling stays inert")[0].rsplit("except KeyError:", 1)[1])

manifest = yaml.safe_load(open("variants/codex_causal_rank_65.yaml", encoding="utf-8"))
ov = dict(manifest["overrides"]); ov["enforce_oos_holdout"] = False
cfg = build_override_config(ov)


class WorkbookWithoutIV30Z:
    def get_sheet(self, name):
        raise KeyError(name)


tickers = ["A", "B", "C"]
scale = _load_optvol_scale(WorkbookWithoutIV30Z(), tickers, cfg)
_, applied = _apply_optvol_scale(np.eye(3), scale, pd.Timestamp("2026-09-04"), tickers)
print("(c) production cfg option_vol_covariance_enabled:", cfg.option_vol_covariance_enabled,
      "| export scale panel:", scale, "| applied:", applied)

vsrc = open("scripts/validate_portfolio_bundles.py", encoding="utf-8").read()
print("(d) validator checks option_vol_cov_scaling_applied:",
      "option_vol_cov_scaling_applied" in vsrc)
