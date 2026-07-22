"""§S11.8(b): effective_label_horizon — causal split이 블렌드 타깃의 최장
horizon으로 purge/embargo를 계산해야 한다(multi-horizon 활성 시 누수 방지).
OFF(기본)면 forward_horizon 그대로 → 기존 분할과 바이트 동일 파리티.
"""

import pandas as pd

from src.config import PipelineConfig
from src.model_trainer import build_walk_forward_split, effective_label_horizon


def test_effective_label_horizon_default_is_forward_horizon():
    assert effective_label_horizon(PipelineConfig()) == 20


def test_effective_label_horizon_uses_max_blend_horizon():
    cfg = PipelineConfig(
        multi_horizon_targets_enabled=True,
        multi_horizon_weights={20: 0.7, 63: 0.3},
    )
    assert effective_label_horizon(cfg) == 63


def test_effective_label_horizon_ignores_weights_when_disabled():
    cfg = PipelineConfig(
        multi_horizon_targets_enabled=False,
        multi_horizon_weights={20: 0.7, 63: 0.3},
    )
    assert effective_label_horizon(cfg) == 20


def test_effective_label_horizon_never_below_forward_horizon():
    cfg = PipelineConfig(
        multi_horizon_targets_enabled=True,
        multi_horizon_weights={5: 1.0},
    )
    assert effective_label_horizon(cfg) == 20


def test_walk_forward_split_stays_causal_at_63d_horizon():
    dates = pd.bdate_range("2018-01-01", periods=900)
    split = build_walk_forward_split(
        all_dates=dates,
        prediction_idx=800,
        train_window=756,
        val_window=126,
        forward_horizon=63,
    )
    audit = split["audit"]
    assert audit["causal_validation_ok"] is True
    # embargo must cover the full 63d label realization window
    assert audit["embargo_days"] == 63
