# Pictet 포트폴리오 로직 → cc2_rl 반영 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pictet Quest enhanced-index의 포트폴리오 구성 규율·설명력 로직(alpha attribution / 오버레이 정리 / beta 중립 / 스타일-팩터 중립)을 cc2_rl에 OFF-default 플래그로 이식하고, 단일 솔버(ECOS)·단일 프로토콜 ablation으로 검증한 뒤 프로덕션 variant에 반영한다.

**Architecture:** 모든 변경은 `PipelineConfig`(SSOT, default-OFF) 플래그 뒤에 게이트되며, OFF일 때 메트릭이 baseline과 동일함을 parity 테스트로 보증한다. 무거운 계산(harvest=walk_forward_train)은 한 번만 수행하고 `precomputed_*` kwargs로 arm을 재-MVO한다(`run_dr_ablation.py` 패턴 재사용). attribution이 모든 후보를 판정하는 공통 언어다.

**Tech Stack:** Python 3.12, pandas/numpy, LightGBM, cvxpy(ECOS→SCS), shap, pytest(무 fixture, plain `test_*` 함수).

**Spec:** `docs/superpowers/specs/2026-06-18-pictet-portfolio-logic-adoption-design.md`

---

## Pre-flight (실행 전 필수 확인)

- **PY** = `C:/Users/westl/PycharmProjects/pythonProject/venv_vf_new/Scripts/python.exe`
- **CC2** = `C:/Users/westl/PycharmProjects/pythonProject/venv_vf_new/machine/re_study/c2/ai_signal_cc2_rl`
- 모든 명령은 **CC2에서** 실행(`from src.X import ...` 임포트 규약). pytest: `<PY> -m pytest ...`.
- **ECOS**: 설치·검증 완료(`cp.installed_solvers()`에 `ECOS` 포함). 따라서 `_solve_problem`이 이제 ECOS를 1순위로 선택 → 책이 과거 SCS-fallback과 달라짐. **STEP 0에서 baseline을 ECOS로 재인증**하고, 이후 모든 arm을 동일 ECOS로 비교한다. 과거 SCS 기반 수치와 직접 비교 금지.
- **Git/tracking 주의(메모리)**: `ai_signal_cc2_rl` 소스가 외부 repo에서 untracked일 수 있음(메모리: "edit main checkout, not worktree"). 각 commit 스텝 전에 `git -C <CC2> status`로 추적 여부를 확인하고, **commit은 사용자가 승인할 때만** 수행(프로젝트 CLAUDE.md 규칙). 추적되지 않으면 commit 스텝은 건너뛰고 변경분만 보고한다.
- **좀비 hang 주의(메모리)**: python을 background로 반복 spawn 금지. 백테스트/ablation은 단일 foreground로 실행.
- **Off-path parity 원칙**: 모든 신규 플래그는 default-OFF, OFF일 때 결과가 baseline과 동일해야 한다. 각 Phase는 이 parity 테스트를 먼저 통과시킨다.

## File Structure (생성/수정 파일과 책임)

| 파일 | 동작 | 책임 |
|---|---|---|
| `src/utils.py` | Modify | `compute_beta(port, bm)` 순수함수 추가(실현 회귀 beta) |
| `src/backtest.py` | Modify | `compute_metrics`에 `realized_beta`/`realized_active_beta` 부착; (P3) per-date factor loadings 빌드 |
| `src/config.py` | Modify | 신규 OFF-default 필드: attribution / beta_neutral / factor_neutral |
| `src/attribution.py` | (read-only 호출) | 기존 `run_attribution` 재사용; 코드 변경 없음(가능하면) |
| `src/harness.py` | Modify | `compute_alpha_attribution` helper + `run_variant` attach |
| `src/portfolio_optimizer.py` | Modify | `_beta_neutral_penalty` / `_factor_neutral_penalty` helper + `optimize_portfolio` objective 항 |
| `run_variant.py` | Modify | CLI run()에 attribution attach + `SAFE_FOR_CACHE_REUSE` 키 추가 |
| `scripts/run_alpha_attribution.py` | Create | leg C(construction) re-MVO counterfactual + 레그 A/B 요약 |
| `scripts/run_overlay_ablation.py` | Create | 오버레이 2³+all-off+on-baseline ablation(harvest-once) |
| `scripts/run_beta_ablation.py` | Create | beta penalty 스윕 {1,5,10,25} |
| `scripts/run_factor_ablation.py` | Create | (P3) 단일 factor penalty, exposure-binding 체크 |
| `tests/test_realized_beta.py` | Create | compute_beta 단위테스트 |
| `tests/test_alpha_attribution_config.py` | Create | attribution 필드 off-path parity |
| `tests/test_beta_neutral.py` | Create | beta penalty disabled-identical / enabled-shrinks / degenerate |
| `tests/test_factor_neutral.py` | Create | (P3) factor penalty disabled-identical / impute / binding |

> **모듈성**: Phase 0가 모든 후보의 공통 진단(baseline+beta)을 깐다. 이후 **Phase 1(P0) → 2(P1) → 3(P2) → 4(P3)**는 각자 독립 실행/평가 가능. attribution(P0)을 먼저 깔아 이후 변경을 같은 언어로 판정한다.

---

# Phase 0 — Baseline & beta 진단 (ECOS 재인증)

### Task 0.1: 실현 회귀 beta 순수함수 추가

**Files:**
- Modify: `src/utils.py` (compute_performance_metrics 근처, 24-71)
- Test: `tests/test_realized_beta.py`

- [ ] **Step 1: 실패 테스트 작성** — `tests/test_realized_beta.py`

```python
"""compute_beta: realized regression beta of a return series vs benchmark.

Ground truth: if port = k*bm + noise, beta -> k. Tests the pure helper that
src/backtest.py compute_metrics() uses for the 'realized_beta' diagnostic.
"""
import numpy as np
import pandas as pd

from src.utils import compute_beta


def _series(arr, start="2015-01-01"):
    idx = pd.bdate_range(start, periods=len(arr))
    return pd.Series(arr, index=idx)


def test_beta_recovers_known_slope():
    rng = np.random.default_rng(0)
    bm = _series(rng.normal(0, 0.01, 500))
    port = 1.20 * bm  # exact, no noise
    assert abs(compute_beta(port, bm) - 1.20) < 1e-9


def test_beta_with_noise_is_close():
    rng = np.random.default_rng(1)
    bm = _series(rng.normal(0, 0.01, 2000))
    port = _series(0.90 * bm.values + rng.normal(0, 0.001, 2000))
    assert abs(compute_beta(port, bm) - 0.90) < 0.05


def test_beta_zero_variance_benchmark_is_nan():
    bm = _series(np.zeros(100))
    port = _series(np.ones(100) * 0.01)
    assert np.isnan(compute_beta(port, bm))


def test_beta_aligns_and_dropna():
    bm = _series([0.01, -0.02, np.nan, 0.03, 0.0])
    port = _series([0.02, -0.04, 0.01, 0.06, np.nan])
    # only rows where both finite are used; port=2*bm on those -> 2.0
    assert abs(compute_beta(port, bm) - 2.0) < 1e-9
```

- [ ] **Step 2: 실패 확인**

Run: `<PY> -m pytest tests/test_realized_beta.py -v`
Expected: FAIL — `ImportError: cannot import name 'compute_beta' from 'src.utils'`

- [ ] **Step 3: 최소 구현** — `src/utils.py`에 추가

```python
def compute_beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Realized regression beta cov(port,bm)/var(bm) on overlapping finite rows.

    Returns nan if <2 overlapping points or var(bm)==0. Sample (ddof=1) moments.
    """
    aligned = pd.concat(
        [portfolio_returns.rename("p"), benchmark_returns.rename("b")], axis=1
    ).dropna()
    if len(aligned) < 2:
        return float("nan")
    p = aligned["p"].values
    b = aligned["b"].values
    var_b = b.var(ddof=1)
    if not np.isfinite(var_b) or var_b <= 0.0:
        return float("nan")
    return float(np.cov(p, b, ddof=1)[0, 1] / var_b)
```

(`import numpy as np`, `import pandas as pd`는 utils.py에 이미 존재 — 확인 후 없으면 추가.)

- [ ] **Step 4: 통과 확인**

Run: `<PY> -m pytest tests/test_realized_beta.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit** (Pre-flight git 규칙 준수 시)

```bash
git -C <CC2> add src/utils.py tests/test_realized_beta.py
git -C <CC2> commit -m "feat(metrics): add compute_beta realized regression beta helper"
```

### Task 0.2: compute_metrics에 realized_beta 부착

**Files:**
- Modify: `src/backtest.py` `BacktestResult.compute_metrics` (550-616)

- [ ] **Step 1: 부착 코드 작성** — compute_metrics가 metrics dict를 return 하기 직전에 추가

```python
        # Realized regression beta diagnostic (Pictet beta=1.0 규율 판정용).
        # 가중치/IR/TE에 영향 없는 read-only 진단. portfolio vs benchmark.
        from src.utils import compute_beta
        _port = self.portfolio_returns.dropna()
        _bm = self.benchmark_returns.reindex(_port.index).ffill().fillna(0.0)
        metrics["realized_beta"] = compute_beta(_port, _bm)
        metrics["realized_active_beta"] = compute_beta(_port - _bm, _bm)
```

(변수명은 compute_metrics 내부의 기존 port/bm 지역변수가 있으면 그것을 재사용; 충돌 피하려 `_port`/`_bm` 사용. metrics dict 변수명이 다르면 맞춘다.)

- [ ] **Step 2: 스모크 — 기존 백테스트가 깨지지 않고 새 키가 생기는지**

Run: `<PY> -m pytest tests/ -q`
Expected: 기존 테스트 전부 PASS (회귀 없음).

- [ ] **Step 3: Commit**

```bash
git -C <CC2> add src/backtest.py
git -C <CC2> commit -m "feat(metrics): attach realized_beta/realized_active_beta to compute_metrics"
```

### Task 0.3: S0 baseline ECOS 재인증 + beta 전제 확인

**Files:** 없음(측정 task). 산출물: `outputs/<baseline label>/metrics.json`.

- [ ] **Step 1: ECOS 선택 확인**

Run: `<PY> -c "import cvxpy as cp; print('ECOS' in cp.installed_solvers())"`
Expected: `True`

- [ ] **Step 2: production variant를 ECOS로 재실행 (단일 foreground)**

Run: `<PY> run_variant.py --variant variants/iter15_65tkr_reb21_vtg.yaml`
Expected: 정상 종료, `outputs/.../metrics.json` 생성. `information_ratio`, `tracking_error`, `avg_annual_turnover`, **`realized_beta`** 기록.

- [ ] **Step 3: 결과 기록 + 의사결정 게이트**

`metrics.json`의 `information_ratio`(≈1.30 ballpark 확인), `tracking_error`(≤0.045 확인, SCS 드리프트 없는지), `realized_beta`를 spec §8 "baseline beta 전제"에 기록.
- **게이트**: `realized_beta`가 ~1.0이면 **P2(beta_neutral)는 코드 작성 전 shelve**(spec §8). 0.90~0.93 부근이면 P2 진행.
- 이 S0 수치가 이후 모든 arm의 단일 비교 기준. (커밋할 코드 없음 — 결과를 plan/spec 메모에 남긴다.)

---

# Phase 1 — P0: Alpha-source attribution (가중치 불변)

### Task 1.1: attribution config 필드 추가 (off-path parity)

**Files:**
- Modify: `src/config.py` (Attribution 섹션 477-484 근처)
- Test: `tests/test_alpha_attribution_config.py`

- [ ] **Step 1: 실패 테스트 작성** — `tests/test_alpha_attribution_config.py`

```python
"""alpha_attribution config fields: OFF by default, no perturbation of existing defaults."""
from src.config import PipelineConfig, DEFAULT_CONFIG


def test_attribution_off_by_default():
    c = PipelineConfig()
    assert c.alpha_attribution_enabled is False
    assert c.alpha_attribution_n_dates == 8


def test_attribution_fields_do_not_perturb_existing_defaults():
    c = PipelineConfig()
    # existing OFF-default precedent unchanged
    assert c.bm_proportional_cap_enabled == DEFAULT_CONFIG.bm_proportional_cap_enabled
    assert c.max_active_share == DEFAULT_CONFIG.max_active_share  # 0.45 (core_satellite)
    assert c.value_trap_gate_enabled == DEFAULT_CONFIG.value_trap_gate_enabled
```

- [ ] **Step 2: 실패 확인**

Run: `<PY> -m pytest tests/test_alpha_attribution_config.py -v`
Expected: FAIL — `AttributeError: ... 'alpha_attribution_enabled'`

- [ ] **Step 3: 필드 추가** — `src/config.py` Attribution 섹션(480-484 근처)에, 하우스 배너 스타일로

```python
    # ------------------------------------------------------------------
    # Alpha-source attribution — Pictet crystal-box (2026-06-18)
    # ------------------------------------------------------------------
    # Wires the existing (dead) src/attribution.run_attribution into the
    # run_variant / CLI metrics path. Read-only: changes no weights. When
    # enabled, attaches metrics['alpha_attribution'] (linear / marginal_nl /
    # interaction-upper-bound group shares). EXPENSIVE (SHAP over n_dates
    # model dates) — OFF by default, turn on only in research/ablation runs.
    alpha_attribution_enabled: bool = False
    alpha_attribution_n_dates: int = 8     # model dates subsampled for SHAP/decomp
```

- [ ] **Step 4: 통과 확인**

Run: `<PY> -m pytest tests/test_alpha_attribution_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C <CC2> add src/config.py tests/test_alpha_attribution_config.py
git -C <CC2> commit -m "feat(config): add alpha_attribution OFF-default flags"
```

### Task 1.2: compute_alpha_attribution helper (harness.py)

**Files:**
- Modify: `src/harness.py` (sub_period_irs 형제로 추가, 50-55 근처)

> `run_attribution(models, panel, feature_names, feature_groups, weights_history=None, n_sample_dates=8)`는 feature panel과 models dict를 받아 dict 반환. degenerate 분기에서 `nonlinear_ratio` 키가 누락될 수 있으므로 helper가 가드한다.

- [ ] **Step 1: helper 작성** — `src/harness.py`

```python
def compute_alpha_attribution(result, n_dates: int = 8) -> dict:
    """Wrap src.attribution.run_attribution into a compact, JSON-safe summary.

    Legs A/B only (signal-variance shares). Leg C (construction) is a separate
    re-MVO counterfactual in scripts/run_alpha_attribution.py. Returns {} on
    any failure so attach never breaks a run. interaction_ratio is an UPPER
    BOUND (clamped residual), labeled as such by the caller's reporting.
    """
    try:
        from src.attribution import run_attribution
    except Exception:
        return {}
    try:
        detail = run_attribution(
            models=result.models,
            panel=result.panel,
            feature_names=result.feature_names,
            feature_groups=result.feature_groups,
            n_sample_dates=n_dates,
        )
    except Exception as exc:  # never break the run on an attribution failure
        return {"error": str(exc)}
    # Average per-date linear/nonlinear ratios into headline shares, guarding
    # the degenerate (total_var<1e-10) branch that omits 'nonlinear_ratio'.
    lin, nl, n = 0.0, 0.0, 0
    for _date, pair in (detail.get("linear_ratios") or {}).items():
        try:
            l, nlr = float(pair[0]), float(pair[1])
        except (TypeError, ValueError, IndexError):
            continue
        if l == l and nlr == nlr:  # skip nan
            lin += l; nl += nlr; n += 1
    headline = {
        "linear_share": (lin / n) if n else float("nan"),
        "nonlinear_share_upper_bound": (nl / n) if n else float("nan"),
        "n_dates_used": n,
    }
    return {
        "headline": headline,
        "group_contributions": detail.get("group_contributions"),
        "feature_importance": (
            detail["feature_importance"].to_dict()
            if detail.get("feature_importance") is not None else None
        ),
    }
```

- [ ] **Step 2: import 가능 스모크**

Run: `<PY> -c "from src.harness import compute_alpha_attribution; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git -C <CC2> add src/harness.py
git -C <CC2> commit -m "feat(attribution): add compute_alpha_attribution harness helper (legs A/B)"
```

### Task 1.3: run_variant(harness) + CLI attach + 캐시 키

**Files:**
- Modify: `src/harness.py` `run_variant` (133-148, sub_periods 블록 직후 line 145)
- Modify: `run_variant.py` CLI `run()` (339 직후, persist 전) + `SAFE_FOR_CACHE_REUSE` frozenset(206-244)

- [ ] **Step 1: harness.run_variant attach** — sub_periods 블록(line 144) 직후에 삽입

```python
        if getattr(config, "alpha_attribution_enabled", False):
            metrics["alpha_attribution"] = compute_alpha_attribution(
                result, n_dates=getattr(config, "alpha_attribution_n_dates", 8)
            )
```

(여기서 `config`는 run_variant 내 override config. 없으면 build_override_config 결과 변수명에 맞춘다.)

- [ ] **Step 2: CLI run() attach** — `run_variant.py` line 339 `metrics["sub_periods"] = ...` 직후

```python
        if getattr(cfg, "alpha_attribution_enabled", False):
            from src.harness import compute_alpha_attribution
            metrics["alpha_attribution"] = compute_alpha_attribution(
                result, n_dates=getattr(cfg, "alpha_attribution_n_dates", 8)
            )
```

(CLI run()의 config 지역변수명이 `cfg`가 아니면 compose_config 반환 변수명에 맞춘다. `result`는 compute_metrics 호출 대상 객체.)

- [ ] **Step 3: 캐시 키 추가** — `run_variant.py` `SAFE_FOR_CACHE_REUSE` frozenset(206-244)에 두 키 추가

```python
        "alpha_attribution_enabled",
        "alpha_attribution_n_dates",
```

(attribution은 features/targets/models를 바꾸지 않으므로 cache-safe.)

- [ ] **Step 4: OFF parity 테스트(수동/스모크)** — attribution OFF로 한 번 실행, metrics.json에 키 없음 확인

Run: `<PY> run_variant.py --variant variants/iter15_65tkr_reb21_vtg.yaml`
Expected: `outputs/.../metrics.json`에 `"alpha_attribution"` 키 **없음**(기본 OFF) → S0과 IR/TE/turnover/realized_beta 동일.

- [ ] **Step 5: ON 스모크** — attribution을 켠 임시 variant로 한 번 실행

임시 variant yaml `variants/_attr_on_tmp.yaml`을 만들어 overrides에 `alpha_attribution_enabled: true` 추가 후:
Run: `<PY> run_variant.py --variant variants/_attr_on_tmp.yaml`
Expected: metrics.json에 `alpha_attribution.headline.linear_share` + `nonlinear_share_upper_bound`(합산 ≈1.0), IR/TE/turnover/realized_beta는 OFF와 **동일**(가중치 불변). 확인 후 임시 yaml 삭제.

- [ ] **Step 6: Commit**

```bash
git -C <CC2> add src/harness.py run_variant.py
git -C <CC2> commit -m "feat(attribution): wire alpha_attribution into run_variant + CLI, cache-safe"
```

### Task 1.4: leg C (construction) re-MVO counterfactual 스크립트

**Files:**
- Create: `scripts/run_alpha_attribution.py` (clone `scripts/run_dr_ablation.py` 28-52 prologue + 120-181 harvest-once)

> Leg C = `annualize(full-pipeline active) − annualize(overlay-OFF re-MVO active)`. 두 백테스트 모두 동일 MVO/TC/제약을 통과 → 단위 일치. uncosted book의 construction_ir 금지(annualized active-return **델타**로만 보고).

- [ ] **Step 1: 스크립트 작성** — `scripts/run_alpha_attribution.py`

```python
"""Alpha-source attribution: legs A/B (run_attribution) + leg C (construction).

Leg C is the annualized active-return DELTA between the full production
pipeline and an overlay-OFF re-MVO of the SAME single harvest. Cloned from
run_dr_ablation.py (harvest-once / re-MVO-many).
"""
# --- prologue: clone run_dr_ablation.py:28-52 verbatim (BLAS clamp, sys.path) ---
import os, sys, json
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1"); os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))


def main() -> int:
    import yaml
    from src.harness import build_override_config, inject_config, sub_period_irs, compute_alpha_attribution
    from src.backtest import run_backtest
    from src.data_loader import UniverseData
    from src.utils import annualise_return  # used for the construction delta

    variant = ROOT / "variants" / "iter15_65tkr_reb21_vtg.yaml"
    overrides = (yaml.safe_load(open(variant, encoding="utf-8")) or {}).get("overrides", {})

    # Production config (overlays ON per their config defaults).
    prod_cfg = build_override_config(dict(overrides)); inject_config(prod_cfg)
    data = UniverseData(prod_cfg.data_path)
    base = run_backtest(data, config=prod_cfg)              # single harvest, overlays ON

    # Overlay-OFF re-MVO arm: reuse the SAME harvested models/panel/targets.
    off = dict(overrides); off.update(
        value_trap_gate_enabled=False, growth_tilt_enabled=False,
        pead_boost_enabled=False, signal_stability_lambda=0.0,
    )
    off_cfg = build_override_config(off); inject_config(off_cfg)
    base_off = run_backtest(
        data, config=off_cfg,
        precomputed_panel=base.panel, precomputed_feature_names=base.feature_names,
        precomputed_feature_groups=base.feature_groups, precomputed_targets=base.targets,
        precomputed_models=base.models,
        precomputed_predictions=base.raw_predictions,      # overlay-free EMA base
        precomputed_raw_predictions=base.raw_predictions,
    )
    inject_config(prod_cfg)  # restore

    def _ann_active(res):
        p = res.portfolio_returns.dropna()
        b = res.benchmark_returns.reindex(p.index).ffill().fillna(0.0)
        return float(annualise_return((p - b), 252))

    construction_delta = _ann_active(base) - _ann_active(base_off)
    out = {
        "legA_B": compute_alpha_attribution(base, n_dates=prod_cfg.alpha_attribution_n_dates),
        "legC_construction_active_delta": construction_delta,
        "full_active": _ann_active(base),
        "overlay_off_active": _ann_active(base_off),
        "note": "interaction is an upper bound; legC is an annualized active-return delta, not summed with A/B.",
    }
    out_dir = ROOT / "outputs" / "alpha_attribution"; out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_dir / "summary.json", "w", encoding="utf-8"), indent=2, default=str)
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

> ⚠️ 실행자 검증 필요: `annualise_return` 임포트 경로(`src.utils`)와 시그니처, `prod_cfg.data_path`, `base.raw_predictions`가 overlay-free EMA base인지(Task 0/1 harvest 의미)를 첫 실행 로그로 확인. `raw_predictions`가 pre-EMA라면 overlay-OFF harvest를 별도로 한 번 더 수행해 그 `predictions`를 쓴다(주석 참조).

- [ ] **Step 2: round-trip identity 검증 실행**

Run: `<PY> scripts/run_alpha_attribution.py`
Expected: `summary.json` 생성. `full_active`가 S0 active와 일치, `overlay_off_active`는 overlay 기여를 뺀 값, `legC_construction_active_delta`는 둘의 차. 로그로 부호·크기 sanity 확인.

- [ ] **Step 3: Commit**

```bash
git -C <CC2> add scripts/run_alpha_attribution.py
git -C <CC2> commit -m "feat(attribution): add leg-C construction re-MVO counterfactual script"
```

---

# Phase 2 — P1: 수동 오버레이 정리 (config-toggle ablation)

> 오버레이 플래그(`value_trap_gate_enabled`/`growth_tilt_enabled`/`pead_boost_enabled`/`signal_stability_lambda`)는 **이미 존재** — config 변경 없음. confound 2건(EMA·이중오버레이)을 harvest-once 설계로 제거하고 OOS holdout으로 do-no-harm 판정.

### Task 2.1: run_overlay_ablation.py 작성

**Files:**
- Create: `scripts/run_overlay_ablation.py` (clone `run_dr_ablation.py` prologue + harvest-once)

- [ ] **Step 1: 스크립트 작성** — `scripts/run_overlay_ablation.py`

```python
"""Overlay ablation: 2^3 grid + all-off purity + on-baseline, single harvest.

CONFOUND FIXES:
  (1) EMA: harvest ONCE with all overlays OFF -> base.raw_predictions is the
      EMA-blended, overlay-FREE base. Feed THAT as precomputed_predictions so
      each arm re-applies only its OWN overlays once (no double-apply, no
      un-smoothed signal).
  (2) double-overlay: never feed post-overlay predictions back in.
Judged on OOS holdout (enforce_oos_holdout + train_cutoff_date).
"""
# --- prologue: clone run_dr_ablation.py:28-52 verbatim ---
import os, sys, json, itertools
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1"); os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))


def main() -> int:
    import yaml
    from src.harness import build_override_config, inject_config, sub_period_irs
    from src.backtest import run_backtest
    from src.data_loader import UniverseData

    variant = ROOT / "variants" / "iter15_65tkr_reb21_vtg.yaml"
    overrides = (yaml.safe_load(open(variant, encoding="utf-8")) or {}).get("overrides", {})

    # Harvest ONCE with all overlays OFF -> overlay-free EMA base predictions.
    off = dict(overrides); off.update(
        value_trap_gate_enabled=False, growth_tilt_enabled=False,
        pead_boost_enabled=False, signal_stability_lambda=0.0,
    )
    base_cfg = build_override_config(off); inject_config(base_cfg)
    data = UniverseData(base_cfg.data_path)
    base = run_backtest(data, config=base_cfg)
    overlay_free = base.predictions  # overlays were OFF in this harvest

    def _arm(vtg, growth, pead):
        arm_over = dict(overrides)
        arm_over.update(
            value_trap_gate_enabled=vtg, growth_tilt_enabled=growth,
            pead_boost_enabled=pead, enforce_oos_holdout=True,
        )
        cfg = build_override_config(arm_over); inject_config(cfg)
        res = run_backtest(
            data, config=cfg,
            precomputed_panel=base.panel, precomputed_feature_names=base.feature_names,
            precomputed_feature_groups=base.feature_groups, precomputed_targets=base.targets,
            precomputed_models=base.models,
            precomputed_predictions=overlay_free,            # overlay-free base
            precomputed_raw_predictions=base.raw_predictions,
        )
        m = res.compute_metrics()
        p = res.portfolio_returns.dropna(); b = res.benchmark_returns.reindex(p.index).ffill().fillna(0.0)
        m["sub_periods"] = sub_period_irs(p, b)
        return m

    rows = {}
    for vtg, growth, pead in itertools.product((False, True), repeat=3):
        key = f"vtg{int(vtg)}_grw{int(growth)}_pead{int(pead)}"
        rows[key] = _arm(vtg, growth, pead)
    inject_config(base_cfg)

    out_dir = ROOT / "outputs" / "overlay_ablation"; out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out_dir / "summary.json", "w", encoding="utf-8"), indent=2, default=str)
    # on-baseline = (1,1,0) per production defaults (VTG off, growth+pead on); all-off = (0,0,0)
    print(json.dumps({k: rows[k].get("information_ratio") for k in rows}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

> `enforce_oos_holdout`/`train_cutoff_date`의 실제 효과(어느 윈도로 IR을 계산하는지)를 첫 실행으로 확인. on-baseline arm = production defaults(VTG off, growth on, pead on) = `vtg0_grw1_pead1`이 S0 IR(~1.30)을 **재현**해야 신뢰.

- [ ] **Step 2: 실행 + 검증**

Run: `<PY> scripts/run_overlay_ablation.py`
Expected: `vtg0_grw1_pead1` arm IR ≈ S0. all-off `vtg0_grw0_pead0` 대비 각 overlay marginal dIR 출력.

- [ ] **Step 3: do-no-harm 판정 기록**

각 overlay leave-out arm의 OOS holdout marginal dIR을 spec §5.2 바(명확히 음(−) & 서브기간 부호 일관일 때만 제거)로 판정. 결과를 `docs/superpowers/plans/`에 메모.

- [ ] **Step 4: Commit**

```bash
git -C <CC2> add scripts/run_overlay_ablation.py
git -C <CC2> commit -m "feat(ablation): overlay 2^3 ablation, EMA+double-overlay confound-free"
```

---

# Phase 3 — P2: Beta-중립 soft penalty

> **선행 게이트(Task 0.3)**: S0 `realized_beta`가 ~1.0이면 이 Phase 전체 shelve. 0.90~0.93이면 진행.
> **설계 정제(spec 대비 더 surgical)**: spec은 `_build_mvo_constraints` 4-tuple + caller 2곳 수정을 제안했으나, risk/turnover penalty가 이미 `optimize_portfolio` objective에만 있고 projection에는 없다는 기존 패턴과 일치시키기 위해 **penalty를 `optimize_portfolio` 내부에서 inline 계산**한다(시그니처 변경 없음, 더 작은 diff). cov·bm_weights가 이미 해당 함수 파라미터로 존재.

### Task 3.1: beta_neutral config 필드 (off-path parity)

**Files:**
- Modify: `src/config.py`
- Test: `tests/test_beta_neutral.py` (config 부분)

- [ ] **Step 1: 실패 테스트(부분) 작성** — `tests/test_beta_neutral.py`

```python
"""beta_neutral: OFF-default, soft cov-implied penalty, disabled==identical weights."""
import numpy as np
from src.config import PipelineConfig, DEFAULT_CONFIG


def test_beta_neutral_off_by_default():
    c = PipelineConfig()
    assert c.beta_neutral_enabled is False
    assert abs(c.beta_neutral_penalty - 1.0) < 1e-12  # soft default, NOT 25
    assert abs(c.beta_active_band - 0.10) < 1e-12      # declared-unused
```

- [ ] **Step 2: 실패 확인**

Run: `<PY> -m pytest tests/test_beta_neutral.py -v`
Expected: FAIL — AttributeError

- [ ] **Step 3: 필드 추가** — `src/config.py` (mega_cap 섹션 인근, 옵티마이저 제약 영역)

```python
    # ------------------------------------------------------------------
    # Beta-neutral targeting — Pictet beta=1.0 (2026-06-18)
    # ------------------------------------------------------------------
    # SOFT objective penalty pulling cov-implied active beta toward 0
    # (portfolio beta toward the benchmark). beta_vec=(cov@bm)/(bm@cov@bm),
    # active_beta = beta_vec @ (w - bm). A soft term cannot cause MVO
    # infeasibility. NOTE: penalty=25 effectively HARD-PINS beta — start at
    # ~1.0 and sweep up. Headline diagnostic is the realized 252d OLS beta
    # (metrics.realized_beta), NOT this cov-implied number.
    # OFF by default — needs validation against baseline before promoting.
    beta_neutral_enabled: bool = False
    beta_neutral_penalty: float = 1.0      # soft weight; sweep {1,5,10,25}
    beta_active_band: float = 0.10         # declared-unused (hard band deferred)
```

- [ ] **Step 4: 통과 확인**

Run: `<PY> -m pytest tests/test_beta_neutral.py::test_beta_neutral_off_by_default -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C <CC2> add src/config.py tests/test_beta_neutral.py
git -C <CC2> commit -m "feat(config): add beta_neutral OFF-default soft-penalty flags"
```

### Task 3.2: beta penalty helper + objective 항

**Files:**
- Modify: `src/portfolio_optimizer.py` (helper 추가 + `optimize_portfolio` objective 522)
- Test: `tests/test_beta_neutral.py` (penalty 동작)

- [ ] **Step 1: 실패 테스트 추가** — `tests/test_beta_neutral.py`에 append

```python
def _toy_inputs(n=8, seed=0):
    import pandas as pd
    rng = np.random.default_rng(seed)
    tickers = [f"T{i}" for i in range(n)]
    mu = pd.Series(rng.normal(0, 0.02, n), index=tickers)
    A = rng.normal(0, 0.01, (n, n)); cov = A @ A.T / 252.0 + np.eye(n) * 1e-4
    bm = np.ones(n) / n
    return mu, cov, bm


def test_disabled_matches_baseline_weights():
    from src.portfolio_optimizer import optimize_portfolio
    from src.config import PipelineConfig
    mu, cov, bm = _toy_inputs()
    c0 = PipelineConfig(); c0.beta_neutral_enabled = False
    c1 = PipelineConfig(); c1.beta_neutral_enabled = False; c1.beta_neutral_penalty = 99.0
    w0 = optimize_portfolio(mu, cov, bm_weights=bm, config=c0)
    w1 = optimize_portfolio(mu, cov, bm_weights=bm, config=c1)
    assert np.allclose(w0, w1, atol=1e-9)  # penalty inert when disabled


def test_enabled_shrinks_active_beta():
    from src.portfolio_optimizer import optimize_portfolio, _beta_vec_cov_implied
    from src.config import PipelineConfig
    mu, cov, bm = _toy_inputs()
    beta_vec = _beta_vec_cov_implied(cov, bm)
    c_off = PipelineConfig(); c_off.beta_neutral_enabled = False
    c_on = PipelineConfig(); c_on.beta_neutral_enabled = True; c_on.beta_neutral_penalty = 50.0
    w_off = optimize_portfolio(mu, cov, bm_weights=bm, config=c_off)
    w_on = optimize_portfolio(mu, cov, bm_weights=bm, config=c_on)
    ab_off = abs(float(beta_vec @ (w_off - bm)))
    ab_on = abs(float(beta_vec @ (w_on - bm)))
    assert ab_on <= ab_off + 1e-9  # penalty pulls active beta toward 0


def test_degenerate_cov_no_penalty_no_crash():
    from src.portfolio_optimizer import optimize_portfolio
    from src.config import PipelineConfig
    import pandas as pd
    n = 5; tickers = [f"T{i}" for i in range(n)]
    mu = pd.Series(np.zeros(n), index=tickers)
    cov = np.zeros((n, n)); bm = np.ones(n) / n
    c = PipelineConfig(); c.beta_neutral_enabled = True; c.beta_neutral_penalty = 10.0
    w = optimize_portfolio(mu, cov, bm_weights=bm, config=c)  # must not raise
    assert np.all(np.isfinite(w))
```

- [ ] **Step 2: 실패 확인**

Run: `<PY> -m pytest tests/test_beta_neutral.py -v`
Expected: FAIL — `_beta_vec_cov_implied` / penalty 미구현

- [ ] **Step 3: 구현** — `src/portfolio_optimizer.py`

helper 추가(모듈 함수):

```python
def _beta_vec_cov_implied(cov_matrix: np.ndarray, bm_weights: np.ndarray) -> np.ndarray:
    """Cov-implied market beta of each name vs the benchmark portfolio.

    beta_i = (cov @ bm)_i / (bm @ cov @ bm). Returns zeros if denom<=1e-12 or
    any non-finite (so the penalty is inert and never triggers a fallback).
    """
    bm = np.asarray(bm_weights, dtype=float)
    cov = np.asarray(cov_matrix, dtype=float)
    n = len(bm)
    if cov.shape != (n, n) or not np.all(np.isfinite(cov)):
        return np.zeros(n)
    denom = float(bm @ cov @ bm)
    if not np.isfinite(denom) or denom <= 1e-12:
        return np.zeros(n)
    beta = (cov @ bm) / denom
    if not np.all(np.isfinite(beta)) or float(np.max(np.abs(beta))) > 5.0:
        return np.zeros(n)
    return beta
```

`optimize_portfolio`의 objective(현재 line 522)를 교체:

```python
    beta_pen = 0
    if getattr(config, "beta_neutral_enabled", False):
        beta_vec = _beta_vec_cov_implied(cov_matrix, bm_weights)
        if np.any(beta_vec):  # all-zero => inert
            beta_pen = config.beta_neutral_penalty * cp.square(beta_vec @ (w - bm_weights))
    objective = cp.Maximize(
        ret - risk_aversion * risk - turnover_penalty * turnover - beta_pen
    )
```

(disabled 시 `beta_pen=0`(int) → objective는 기존과 bit-identical.)

- [ ] **Step 4: 통과 확인**

Run: `<PY> -m pytest tests/test_beta_neutral.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: 누수 가드 스모크** — `_beta_vec_cov_implied`는 cov만 사용(미래수익 미사용)임을 코드리뷰로 확인.

- [ ] **Step 6: Commit**

```bash
git -C <CC2> add src/portfolio_optimizer.py tests/test_beta_neutral.py
git -C <CC2> commit -m "feat(optimizer): add soft cov-implied beta-neutral penalty (OFF default)"
```

### Task 3.3: beta penalty 스윕 ablation

**Files:**
- Create: `scripts/run_beta_ablation.py` (clone `run_dr_ablation.py` prologue + harvest-once; arms = penalty {0,1,5,10,25})

- [ ] **Step 1: 스크립트 작성** — 핵심 루프(prologue는 위 스크립트와 동일하게 clone)

```python
def main() -> int:
    import yaml
    from src.harness import build_override_config, inject_config, sub_period_irs
    from src.backtest import run_backtest
    from src.data_loader import UniverseData

    variant = ROOT / "variants" / "iter15_65tkr_reb21_vtg.yaml"
    overrides = (yaml.safe_load(open(variant, encoding="utf-8")) or {}).get("overrides", {})
    base_cfg = build_override_config(dict(overrides)); inject_config(base_cfg)
    data = UniverseData(base_cfg.data_path)
    base = run_backtest(data, config=base_cfg)  # harvest once

    rows = {}
    for pen in (0.0, 1.0, 5.0, 10.0, 25.0):
        ov = dict(overrides); ov.update(
            beta_neutral_enabled=(pen > 0.0), beta_neutral_penalty=pen)
        cfg = build_override_config(ov); inject_config(cfg)
        res = run_backtest(
            data, config=cfg,
            precomputed_panel=base.panel, precomputed_feature_names=base.feature_names,
            precomputed_feature_groups=base.feature_groups, precomputed_targets=base.targets,
            precomputed_models=base.models,
            precomputed_predictions=base.predictions,
            precomputed_raw_predictions=base.raw_predictions,
        )
        m = res.compute_metrics()
        p = res.portfolio_returns.dropna(); b = res.benchmark_returns.reindex(p.index).ffill().fillna(0.0)
        m["sub_periods"] = sub_period_irs(p, b)
        rows[f"pen_{pen}"] = {
            "information_ratio": m.get("information_ratio"),
            "tracking_error": m.get("tracking_error"),
            "avg_annual_turnover": m.get("avg_annual_turnover"),
            "realized_beta": m.get("realized_beta"),
            "realized_active_beta": m.get("realized_active_beta"),
            "sub_periods": m["sub_periods"],
        }
    inject_config(base_cfg)
    out_dir = ROOT / "outputs" / "beta_ablation"; out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out_dir / "summary.json", "w", encoding="utf-8"), indent=2, default=str)
    print(json.dumps({k: (v["realized_beta"], v["information_ratio"]) for k, v in rows.items()}, indent=2))
    return 0
```

- [ ] **Step 2: 실행 + 판정**

Run: `<PY> scripts/run_beta_ablation.py`
Expected: `pen_0.0`가 S0와 동일(off-parity). penalty↑ 시 **realized_beta가 1.0 쪽으로 상승**해야(헤드라인). IR은 노이즈(±0.36) 내 예상. fallback율/TE 가드 확인.
- **게이트**: realized_beta가 penalty=25에도 안 움직이면 음의 결과로 shelve. 단일 사전약정 penalty로만 채택(스윕-최대-IR 선택 금지).

- [ ] **Step 3: Commit**

```bash
git -C <CC2> add scripts/run_beta_ablation.py
git -C <CC2> commit -m "feat(ablation): beta-neutral penalty sweep {0,1,5,10,25}"
```

---

# Phase 4 — P3: 스타일-팩터 노출 중립화 (마지막, 가드레일)

> 가장 invasive. soft penalty가 **non-binding 가능성 높음**(TE-var가 이미 흡수). 가드레일: **단일 사전약정 penalty, growth+momentum 축 제외, non-IR 선택규칙(노출 하락), per-date non-finite impute→0**. P1 결과로 오버레이가 실 노출을 갖는지 확인 후 의미가 커진다.

### Task 4.1: factor_neutral config 필드

**Files:**
- Modify: `src/config.py`
- Test: `tests/test_factor_neutral.py` (config)

- [ ] **Step 1: 실패 테스트** — `tests/test_factor_neutral.py`

```python
"""factor_neutral: OFF-default, axes exclude growth+momentum, single pre-committed penalty."""
from src.config import PipelineConfig


def test_factor_neutral_off_by_default():
    c = PipelineConfig()
    assert c.factor_neutral_enabled is False
    assert c.factor_neutral_penalty >= 0.0
    # growth/momentum excluded a priori (conflict with growth_tilt/PEAD)
    axes = set(c.factor_neutral_axes)
    assert "growth" not in axes and "momentum" not in axes
    assert len(c.factor_neutral_axes) >= 1
```

- [ ] **Step 2: 실패 확인** → FAIL (AttributeError)

Run: `<PY> -m pytest tests/test_factor_neutral.py -v`

- [ ] **Step 3: 필드 추가** — `src/config.py` (mutable list는 `field(default_factory=...)` 필수)

```python
    # ------------------------------------------------------------------
    # Style-factor active-exposure neutralization — Pictet (2026-06-18)
    # ------------------------------------------------------------------
    # SOFT penalty pulling active style exposures (loadings @ active) toward 0.
    # Loadings are existing cross-sectional feature z-scores. GROWTH+MOMENTUM
    # axes excluded a priori (they conflict with growth_tilt/PEAD overlays).
    # Likely NON-BINDING (TE-var already absorbs systematic style variance) —
    # judge by exposure drop, NOT IR. OFF by default; single pre-committed
    # penalty (no sweep-by-IR selection = p-hacking).
    factor_neutral_enabled: bool = False
    factor_neutral_penalty: float = 5.0    # single pre-committed weight
    factor_neutral_axes: List[str] = field(
        default_factory=lambda: ["value", "quality", "size", "lowvol"]
    )
    # axis -> feature column used as the loading (must exist in the panel)
    factor_neutral_loadings: Dict[str, str] = field(default_factory=lambda: {
        "value": "best_peg_ratio_level_z",
        "quality": "best_roe_level_z",
        "size": "market_capitalisation_z",   # 실행자: 패널의 실제 size z 컬럼명으로 확정
        "lowvol": "idio_vol_63d",
    })
```

> ⚠️ `List`/`Dict`/`field`가 config.py 임포트에 이미 있음(확인됨). `factor_neutral_loadings`의 컬럼명은 **실행자가 `src/features/assembly.py` CORE_FEATURE_WHITELIST와 대조해 실재 컬럼으로 확정**(예: size/lowvol 컬럼명). 존재하지 않는 컬럼은 Task 4.2에서 impute→0으로 안전 처리되지만, 의미 있는 중립화를 위해 실재 컬럼이어야 한다.

- [ ] **Step 4: 통과 확인** → PASS

Run: `<PY> -m pytest tests/test_factor_neutral.py::test_factor_neutral_off_by_default -v`

- [ ] **Step 5: Commit**

```bash
git -C <CC2> add src/config.py tests/test_factor_neutral.py
git -C <CC2> commit -m "feat(config): add factor_neutral OFF-default flags (growth/momentum excluded)"
```

### Task 4.2: factor penalty helper + loadings 스레딩

**Files:**
- Modify: `src/portfolio_optimizer.py` (helper + `optimize_portfolio`에 `factor_loadings` 옵션 파라미터 + objective 항)
- Modify: `src/backtest.py` (`simulate_portfolio`의 optimizer 호출부에서 per-date loadings 전달)
- Test: `tests/test_factor_neutral.py` (penalty 동작)

- [ ] **Step 1: 실패 테스트 추가** — `tests/test_factor_neutral.py`

```python
import numpy as np


def test_factor_penalty_disabled_identical():
    from src.portfolio_optimizer import optimize_portfolio
    from src.config import PipelineConfig
    import pandas as pd
    n = 8; tk = [f"T{i}" for i in range(n)]; rng = np.random.default_rng(0)
    mu = pd.Series(rng.normal(0, 0.02, n), index=tk)
    A = rng.normal(0, 0.01, (n, n)); cov = A @ A.T / 252 + np.eye(n) * 1e-4
    bm = np.ones(n) / n; L = rng.normal(0, 1, (n, 2))
    c0 = PipelineConfig(); c0.factor_neutral_enabled = False
    w_none = optimize_portfolio(mu, cov, bm_weights=bm, config=c0)
    w_load = optimize_portfolio(mu, cov, bm_weights=bm, config=c0, factor_loadings=L)
    assert np.allclose(w_none, w_load, atol=1e-9)  # loadings ignored when disabled


def test_factor_penalty_reduces_active_exposure():
    from src.portfolio_optimizer import optimize_portfolio
    from src.config import PipelineConfig
    import pandas as pd
    n = 8; tk = [f"T{i}" for i in range(n)]; rng = np.random.default_rng(2)
    mu = pd.Series(rng.normal(0, 0.02, n), index=tk)
    A = rng.normal(0, 0.01, (n, n)); cov = A @ A.T / 252 + np.eye(n) * 1e-4
    bm = np.ones(n) / n; L = rng.normal(0, 1, (n, 2))
    c_off = PipelineConfig(); c_off.factor_neutral_enabled = False
    c_on = PipelineConfig(); c_on.factor_neutral_enabled = True; c_on.factor_neutral_penalty = 50.0
    w_off = optimize_portfolio(mu, cov, bm_weights=bm, config=c_off, factor_loadings=L)
    w_on = optimize_portfolio(mu, cov, bm_weights=bm, config=c_on, factor_loadings=L)
    e_off = np.abs(L.T @ (w_off - bm)).sum()
    e_on = np.abs(L.T @ (w_on - bm)).sum()
    assert e_on <= e_off + 1e-9


def test_factor_loadings_nonfinite_imputed_no_crash():
    from src.portfolio_optimizer import optimize_portfolio
    from src.config import PipelineConfig
    import pandas as pd
    n = 6; tk = [f"T{i}" for i in range(n)]
    mu = pd.Series(np.zeros(n), index=tk)
    cov = np.eye(n) * 1e-4; bm = np.ones(n) / n
    L = np.full((n, 2), np.nan)  # all non-finite -> impute 0 -> inert
    c = PipelineConfig(); c.factor_neutral_enabled = True; c.factor_neutral_penalty = 10.0
    w = optimize_portfolio(mu, cov, bm_weights=bm, config=c, factor_loadings=L)
    assert np.all(np.isfinite(w))
```

- [ ] **Step 2: 실패 확인** → FAIL

Run: `<PY> -m pytest tests/test_factor_neutral.py -v`

- [ ] **Step 3: 구현 — optimizer** — `src/portfolio_optimizer.py`

`optimize_portfolio` 시그니처에 `factor_loadings: Optional[np.ndarray] = None` 추가. helper + objective 항:

```python
def _factor_penalty_expr(w, bm_weights, factor_loadings, config):
    """Soft penalty on active style exposures: sum_squares(L.T @ (w - bm)).

    Non-finite loadings imputed to 0 (inert). Returns int 0 when disabled or
    no usable loadings, so the objective is bit-identical when OFF.
    """
    if not getattr(config, "factor_neutral_enabled", False) or factor_loadings is None:
        return 0
    L = np.asarray(factor_loadings, dtype=float)
    L = np.where(np.isfinite(L), L, 0.0)        # impute-to-0
    if L.ndim != 2 or L.shape[0] != len(bm_weights) or not np.any(L):
        return 0
    return config.factor_neutral_penalty * cp.sum_squares(L.T @ (w - bm_weights))
```

objective(beta_pen 추가한 줄)에 factor penalty도 빼기:

```python
    factor_pen = _factor_penalty_expr(w, bm_weights, factor_loadings, config)
    objective = cp.Maximize(
        ret - risk_aversion * risk - turnover_penalty * turnover - beta_pen - factor_pen
    )
```

- [ ] **Step 4: optimizer 단위테스트 통과**

Run: `<PY> -m pytest tests/test_factor_neutral.py -v`
Expected: PASS (config + 3 penalty 테스트)

- [ ] **Step 5: 구현 — simulate에서 per-date loadings 전달** — `src/backtest.py`

`simulate_portfolio`(시그니처 1167 근처)에서, 각 리밸런스 날짜의 panel 행에서 `config.factor_neutral_loadings`의 컬럼을 뽑아 `(n_tickers × k)` 배열로 만들고 `optimizer_fn(..., factor_loadings=L_date)`로 전달. enabled가 아닐 때는 `None`(기존 경로 보존). 구체:

```python
    # --- factor-neutral loadings for this rebalance date (P3) ---
    factor_loadings = None
    if getattr(config, "factor_neutral_enabled", False) and panel is not None:
        cols = [config.factor_neutral_loadings[a] for a in config.factor_neutral_axes
                if a in config.factor_neutral_loadings]
        try:
            sub = panel.xs(date, level="date").reindex(tickers)[cols]
            factor_loadings = np.where(np.isfinite(sub.values), sub.values, 0.0)
        except (KeyError, ValueError):
            factor_loadings = None
```

그리고 optimizer 호출에 `factor_loadings=factor_loadings` 추가.
> ⚠️ 실행자: `simulate_portfolio`가 `panel`/`date`/`tickers`를 해당 스코프에서 접근 가능한지 확인(시그니처에 panel이 없으면 run_backtest에서 closure로 주입하거나 `precomputed_panel`을 사용). optimizer_fn closure 시그니처에 `factor_loadings` kwarg를 통과시킨다. 이 스레딩이 P3에서 가장 큰 변경점.

- [ ] **Step 6: 회귀 스모크 — disabled 경로 불변**

Run: `<PY> -m pytest tests/ -q`
Expected: 기존 테스트 전부 PASS (factor_neutral OFF 기본 → 경로 불변).

- [ ] **Step 7: Commit**

```bash
git -C <CC2> add src/portfolio_optimizer.py src/backtest.py tests/test_factor_neutral.py
git -C <CC2> commit -m "feat(optimizer): soft style-factor neutralization penalty + per-date loadings (OFF default)"
```

### Task 4.3: factor ablation (단일 penalty, 노출 바인딩 체크)

**Files:**
- Create: `scripts/run_factor_ablation.py` (harvest-once; arms = OFF vs 단일 사전약정 penalty)

- [ ] **Step 1: 스크립트 작성** — 2 arm(off / on@factor_neutral_penalty), 각 arm에서 평균 `|active style exposure|`/축 + IR/TE/turnover 기록. (구조는 run_beta_ablation.py와 동일, override만 `factor_neutral_enabled`.)

- [ ] **Step 2: 실행 + 판정**

Run: `<PY> scripts/run_factor_ablation.py`
Expected: on arm에서 penalized 축의 mean|active exposure|가 측정 가능하게 **하락**(바인딩)해야. 안 움직이면 "TE-var가 이미 중립화"로 결론하고 **shelve**(유효한 crystal-box 결론). IR은 guardrail only(선택 기준 아님).

- [ ] **Step 3: Commit**

```bash
git -C <CC2> add scripts/run_factor_ablation.py
git -C <CC2> commit -m "feat(ablation): single-penalty factor-neutral exposure-binding check"
```

---

# Phase 5 — 프로덕션 롤아웃 (바 통과 후보만)

### Task 5.1: variant yaml 플래그 flip + 재검증 (후보당 1회)

**Files:**
- Modify: `variants/iter15_65tkr_reb21_vtg.yaml` (overrides 블록)

- [ ] **Step 1: 후보별 게이트 확인** — 각 후보가 spec §5.2 성공 바를 통과했는지 표로 확정:
  - attribution: 즉시 가능(bit-identical, 가중치 불변).
  - overlay 변경: OOS holdout 명확한 음(−) & 부호 일관일 때만.
  - beta_neutral: realized_beta가 1.0 쪽으로 이동 + 단일 사전약정 penalty.
  - factor_neutral: 노출 바인딩 확인 시에만(아니면 shelve).

- [ ] **Step 2: 통과 후보 1개의 플래그를 overrides에 추가** (예: attribution)

```yaml
overrides:
  # ... 기존 ...
  alpha_attribution_enabled: true
```

(config.py 기본은 OFF 유지 — variant override로만 활성화.)

- [ ] **Step 3: S0 대비 재검증 (동일 ECOS)**

Run: `<PY> run_variant.py --variant variants/iter15_65tkr_reb21_vtg.yaml`
Expected: IR/TE/turnover/realized_beta/fallback율이 기대대로(attribution은 불변, 가중치 변경 후보는 검증된 델타). 집중 캐릭터(active share~4.75%/TE~3.2%) 보존 확인.

- [ ] **Step 4: Commit (후보당 독립)**

```bash
git -C <CC2> add variants/iter15_65tkr_reb21_vtg.yaml
git -C <CC2> commit -m "prod: enable <candidate> in baseline variant after ablation pass"
```

- [ ] **Step 5: 롤백 확인** — 플래그를 false로 되돌리면 S0와 바이트동일 복원됨을 1회 확인.

---

## Self-Review (작성자 체크)

- **Spec coverage**: §4 후보 P0(Phase1)/P1(Phase2)/P2(Phase3)/P3(Phase4) 매핑 ✓; §3.1 ECOS 재인증(Task 0.3) ✓; §3.2 attribution dead-wiring(Task1.2-1.4) ✓; §5 ablation/성공바(각 ablation Task의 게이트 + Phase5) ✓; §7 롤백(Task5.1 Step5)/롤아웃(Phase5) ✓; §8 결정지점: baseline beta(Task0.3 게이트), 저-beta 판단(Task3.3 게이트), OOS 검정력(Task2.1 주석), DSR(아래 open) — DSR 회계는 본 plan 범위 밖, 실행 중 결정으로 남김.
- **Placeholder scan**: 모든 코드 스텝에 실제 코드 포함. `factor_neutral_loadings` 컬럼명과 simulate_portfolio panel 접근은 ⚠️로 실행자 검증 지점 명시(추정 아닌 확인 요구). beta penalty의 spec-대비-inline 정제는 Phase3 머리에 근거 기재.
- **Type consistency**: helper 이름 `_beta_vec_cov_implied`/`_factor_penalty_expr`/`compute_beta`/`compute_alpha_attribution` 전 Task 일관. config 필드명 `alpha_attribution_enabled`/`beta_neutral_*`/`factor_neutral_*` 일관. `optimize_portfolio(... factor_loadings=...)` 시그니처 Task4.2에서 정의→Task4.2 simulate에서 사용 일치.

## Open (실행 중 해소)
- DSR/deflation: 새 ablation arm을 `experiment_inventory.json`+`run_selection_bias.py` 해킷에 포함할지(spec §8). sub-haircut delta는 비액션 간주 입장 일관 적용.
- `src/attribution.py`의 harness 사본(`ai_signal_cc2_harness/src/attribution.py`) 동기화 여부 — 본 plan은 attribution.py를 **수정하지 않으므로**(호출만) 사본 동기화 불필요.
