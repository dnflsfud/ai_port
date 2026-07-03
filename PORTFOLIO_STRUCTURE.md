# 포트폴리오 코드 구조 (ai_port — self-contained)

> **이 프로젝트는 무엇인가**: cc2_rl 신호·백테스트 엔진을 `./src`에 **벤더링**하여,
> cc2_rl 체크아웃 없이 단독 실행되는 자립 프로젝트. 그 위에 Pictet식 리스크 규율
> (attribution·overlay·beta·factor) **평가(EVALUATE) 레이어**와 게이트 오케스트레이터를 얹었다.
> 출처·독립성 경계는 [`ENGINE_PROVENANCE.md`](./ENGINE_PROVENANCE.md), 운영 계약은 [`CLAUDE.md`](./CLAUDE.md).

---

## 0. 한눈에 — 레이어 지도

```
                            ┌───────────────────────────────────────────────┐
  평가/오케스트레이션 레이어   │  run_pictet_adoption.py  (스테이지 실행 + 게이트) │
  (이 프로젝트 고유)          │  scripts/run_*_ablation.py  (P0/P1/P3 연구)      │
                            └───────────────────┬───────────────────────────┘
                                                │ from src.X import …
  ┌─────────────────────────────────────────────▼──────────────────────────────┐
  │  벤더링된 엔진  ./src  (cc2_rl 스냅샷, ~8,600 LOC)                            │
  │                                                                              │
  │  data_loader → features/ → target_engine → model_trainer → backtest         │
  │                                                  │            │              │
  │                                                  │            └─ portfolio_optimizer (MVO)
  │                                                  └─ harness (config 글루) · attribution (SHAP)
  │                                                     config (PipelineConfig) · utils            │
  └──────────────────────────────────────────────────────────────────────────────┘
                                                │ reads
                            ┌───────────────────▼───────────────────┐
                            │  공유 Excel (re_study/ai_signal_data.xlsx, 레포 밖)  │
                            └────────────────────────────────────────┘
```

---

## 1. 최상위 디렉터리 레이아웃

```
ai_port/
├─ run_pictet_adoption.py      ★ 오케스트레이터: 4 스테이지 실행 → 게이트 판정 → adoption_summary.json
├─ run_variant.py                 프로덕션 진입점 (variant YAML → 백테스트 → metrics.json) + 캐시 로직
├─ run_selection_bias.py          DSR/selection-bias 게이트 (§2-7)
├─ experiment_inventory.json      DSR 실험 인벤토리
├─ requirements.txt               pip 의존성 (cvxpy·lightgbm·shap·sklearn·pandas…)
│
├─ src/                         ★ 벤더링 엔진 (자체 소스)
│   ├─ config.py                   PipelineConfig (dataclass, ~50 플래그, 신규는 default-OFF)
│   ├─ data_loader.py              UniverseData: Excel 시트 → prices/returns/market_cap/펀더멘털
│   ├─ feature_engine.py           build_all_features (피처 조립 진입점)
│   ├─ features/                   피처 모듈 (price·accounting·factor·sellside·regime·interaction·macro_cross·short_interest·conditioning·utils)
│   ├─ target_engine.py            build_targets (학습 타깃)
│   ├─ model_trainer.py            walk_forward_train (LightGBM 워크포워드) + apply_prediction_ema
│   ├─ backtest.py                 ★ 중심: run_backtest·simulate_portfolio·오버레이·BacktestResult·compute_metrics
│   ├─ portfolio_optimizer.py      ★ MVO: optimize_portfolio·_build_mvo_constraints·_solve_problem(ECOS; SCS는 예외 시만)
│   ├─ harness.py                  build_override_config·inject_config·run_variant·sub_period_irs·compute_alpha_attribution
│   ├─ attribution.py              run_attribution (SHAP/3-성분 분해) — P0가 사용
│   ├─ utils.py                    compute_beta·annualise_return·compute_performance_metrics
│   ├─ analytics.py·metadata.py·logging_config.py   부가 유틸
│   └─ rl/                         CS-DR-Alpha (dr_alpha·dr_walkforward) — 프로덕션 OFF 오버레이
│
├─ scripts/                       연구용 ablation (harvest-once / re-MVO-many)
│   ├─ run_alpha_attribution.py    P0: leg A/B(SHAP) + leg C(construction 델타)
│   ├─ run_overlay_ablation.py     P1: 2³ 오버레이 그리드 + S0 round-trip 게이트
│   ├─ run_factor_ablation.py      P3: OFF vs penalty + exposure 바인딩 + 붕괴 신호
│   ├─ run_dr_ablation.py·run_dr_alpha.py   기존 DR-alpha (위 스크립트들의 원형)
│   └─ build_dashboard_data.py     대시보드 데이터
│
├─ variants/                      실험 설정 YAML (iter15_65tkr_reb21_vtg.yaml = 프로덕션)
├─ tests/                         plain 함수 단위테스트 (8개 파일, 34 테스트)
├─ outputs/                       스테이지 산출물 (CWD 상대 — 이 폴더에 격리)
│   ├─ iter15_65tkr_reb21_vtg/metrics.json   S0 baseline
│   ├─ alpha_attribution/ · overlay_ablation/ · factor_ablation/summary.json
│   └─ adoption_summary.json      최종 게이트 판정
├─ logs/                          스테이지별 실행 로그
│
├─ CLAUDE.md                      운영 계약 (불변식·게이트·실행 루프)
├─ ENGINE_PROVENANCE.md           벤더링 출처·독립성·재동기화
├─ PORTFOLIO_STRUCTURE.md         (이 문서)
├─ …-decision-log.md              결정 로그 (S0~S5 + DSR + 리뷰 기록)
└─ …-design.md / …-adoption.md / README.md   spec·plan·codex 리뷰
```

---

## 2. 엔진 데이터 흐름 (`run_backtest` 한 번)

```
UniverseData(Excel) → features/assembly → panel(피처)
        │                                    │
        │                            walk_forward_train (LightGBM, 워크포워드)
        │                                    │
        │                            apply_prediction_ema → result.raw_predictions (PRE-EMA)
        │                                    │               result.predictions    (post-EMA)
        │                                    ▼
        │              ┌── 오버레이 (backtest.py, walk_forward 後) ──┐
        │              │  value_trap_gate · growth_tilt · pead_boost │  예측 점수 조정
        │              │  signal_stability_shrinkage                  │
        │              └──────────────────┬──────────────────────────┘
        │                                 ▼
        │              simulate_portfolio (리밸런스 루프, 일별 drift, 턴오버·IC 추적)
        │                                 │  매 리밸런스일:
        │                                 ▼
        │              _optimizer_fn: estimate_covariance → factor_loadings(P3) → optimize_portfolio
        │                                 ▼
        └─ get_benchmark_fn ──────► BacktestResult → compute_metrics() : IR·TE·turnover·realized_beta
```

### harvest-once / re-MVO-many (ablation 핵심 패턴)
비싼 단계(harvest = 데이터→피처→학습→EMA)와 포트폴리오 구성(re-MVO = 오버레이+optimizer)을 **분리**:

```python
base = run_backtest(data, config=overlays_OFF)        # 1) 한 번만 harvest → 순수 EMA base
overlay_free = base.predictions
for arm in arms:                                       # 2) harvest 재사용, optimizer/오버레이만 교체
    run_backtest(data, config=arm,
                 precomputed_predictions=overlay_free,
                 precomputed_panel=base.panel, precomputed_models=base.models, …)
```
→ arm 차이가 optimizer/오버레이로 격리되고, **OFF arm이 S0를 정확히 재현**(round-trip identity).
§4.2 이중오버레이 confound는 `base.predictions`(post-EMA)를 쓰고 `raw_predictions`(PRE-EMA)를 안 쓰는 걸로 차단.

---

## 3. Optimizer 제약 레이어 (`portfolio_optimizer.py`)

**설계 핵심**: 하드 제약 1벌(`_build_mvo_constraints`)을 **두 소비자**가 공유 →
`optimize_portfolio`(타깃 북)와 `project_portfolio_weights`(실행 투영)가 동일 feasible region을 쓴다.

### 목적함수 (soft — 트레이드오프)
```python
Maximize( mu·w  −  risk_aversion·risk  −  turnover_penalty·turnover  −  factor_pen )
#  risk     = quad_form(active, cov),  active = w − bm     (일별 액티브 분산)
#  turnover = norm1(w − prev)                              (리밸런스당 양방향 턴오버)
#  factor_pen = penalty·sum_squares(Lᵀ(w−bm))             (P3 스타일 중립, default 0)
```

### 하드 제약 카탈로그
| 제약 | 식 | 의미 |
|---|---|---|
| 완전투자 | `sum(w) == 1` | 현금 0 |
| 롱온리 | `w >= 0` | 공매도 금지 |
| 종목 상한 | `w <= max_weight` | 종목당 절대 비중 캡 |
| **TE 예산** ★ | `risk <= max_te_annual²/252` | 일별 액티브 분산 한도(가드 4.5%) — 바인딩 리스크 제약 |
| 턴오버 캡 | `turnover <= max_single_turnover` | 리밸런스 1회 거래량 |
| BM 바닥 | `w[i] >= bm[i]·bm_weight_floor` | 종목 탈락 방지 |
| 종목 액티브 캡 | `±(w[i]−bm[i]) <= max_active_per_stock` | 종목별 OW/UW 한도 |
| 액티브 셰어 캡 | `norm1(w−bm) <= max_active_share` | 총 액티브 비중(L1) |
| 섹터 밴드 | `sector_bm ± sector_deviation` | 섹터 노출 |
| (옵션) score-gated OW · mega-cap protection/funding · BM-proportional cap | | 조건부 캡 조정 |

- **리스크/턴오버 이중처리**: 목적함수 soft penalty이자 동시에 hard cap — 평소엔 penalty가 형태를 다듬고 극단에서만 hard가 자름.
- **리스크 모델**: `estimate_covariance` = Ledoit-Wolf 수축 + 메가캡 변동성 수축(PSD 보존) + 조건수 경고. `cp.psd_wrap(cov)`로 PSD 단언.
- **솔버 + fallback**: `_solve_problem` = ECOS 1순위, **SCS는 ECOS가 예외(SolverError / NaN·Inf 입력)를 raise할 때만** 시도. ECOS가 예외 없이 non-optimal status를 반환하면 **SCS 없이 곧장 BM fallback**(SCS 재시도는 solve를 바꿔 §2-2 단일ECOS/parity 위반이라 의도적으로 안 함). `optimize_portfolio`는 ① 솔버 예외 ② non-optimal status ③ non-finite 해 시 **전부 `bm_weights.copy()`** + `diag["used_fallback"]`/`fallback_reason` 기록 → `optimizer_failure_rate`로 집계(S0 6.38%). **이게 §2-5 벤치마크 붕괴 경로**(리뷰 M3가 감시).
- **factor penalty 위치(P3, §4.1)**: 목적함수에만(`_factor_penalty_expr`), 제약/투영엔 미적용. disabled 시 `0`(int) → 목적함수 바이트 동일(OFF 파리티).
- **beta는 제약 아님**: 북은 TE·액티브셰어·섹터로 통제, `realized_beta≈1.0`은 cap-weighted BM + 타이트 캡에서 자연발생 → **P2 shelve 근거**.

---

## 4. 평가/오케스트레이션 레이어

```
run_pictet_adoption.py
  STAGES = {0: S0 baseline(run_variant), 1: alpha attr, 2: overlay abl, 3: factor abl}
    │  각 스테이지 = 단일 foreground 서브프로세스 (PYTHONPATH=ai_port, CWD=ai_port, logs/stageN.log)
    ▼
  _verdict_baseline / _attribution / _overlay / _factor   (각 JSON 출력에 CLAUDE.md 게이트 적용)
    │   beta SHELVE(0.90–0.93 밖) · overlay do-no-harm(부호 방향고정) ·
    │   factor 붕괴 FAIL(TE/active share/fallback) · round-trip harvest_invalid
    ▼
  outputs/adoption_summary.json
```

게이트 판정(현재 S0 기준):
- **S0**: IR 1.485 · TE 0.0310 · turnover 1.144 · realized_beta 1.024 → **P2 SHELVED**
- **overlay**: 3개 전부 KEEP (do-no-harm)
- **factor**: binds하나 IR 트레이드오프 → **CONFIRMED LEVER, OFF-default 유지**
- **프로덕션 가중치 무변경** — 책이 이미 잘 행동(beta≈1, overlay do-no-harm, 스타일=의도적 alpha)

---

## 5. 설정 & 규율 (전체 관통)

- **OFF-default + parity**: `PipelineConfig`의 신규 플래그(`alpha_attribution_*`, `factor_neutral_*`)는 전부 default-OFF. OFF면 메트릭이 baseline과 **바이트 동일**(단위테스트로 고정).
- **단일 ECOS 프로토콜**: 모든 arm이 동일 솔버·경로. 과거 SCS 수치와 직접 비교 금지(§2-2).
- **config 한 곳**: 모든 동작이 `PipelineConfig` 플래그 → variant YAML override로만 프로덕션 활성화.
- **evaluate-then-gate**: P0~P3 전부 측정, 게이트 통과 후보만 활성화(이번 라운드 활성화 0).

---

## 6. Pictet 적응 작업이 꽂힌 위치

| 단계 | 위치 | 내용 |
|---|---|---|
| P0 attribution | `harness.compute_alpha_attribution` + `attribution.py` + `scripts/run_alpha_attribution.py` | 신호분산 leg A/B + construction leg C |
| P1 overlay | `scripts/run_overlay_ablation.py` | 2³ do-no-harm ablation |
| P2 beta (shelved) | (미구현) `optimize_portfolio` inline penalty 자리 | realized_beta≈1.0 게이트로 보류 |
| P3 factor | `config.factor_neutral_*` + `backtest._optimizer_fn` 로딩 스레딩 + `portfolio_optimizer._factor_penalty_expr` | soft 스타일 중립 penalty |
| 리뷰 수정 | 위 레이어들의 **관측성·verdict 로직** (M1~M3·L1~L8·GAP2/5) | 계산식·OFF 파리티 불변 |

---

## 7. 실행 방법

```bash
PY=…/venv_vf_new/Scripts/python.exe
cd …/c2/ai_port

# 단위테스트 (cc2_rl 불필요 — 독립성 증명)
PYTHONPATH=. "$PY" -m pytest tests/ -q                 # 34 passed

# 전체 평가 재현 (4 스테이지 from-scratch → 게이트)
"$PY" run_pictet_adoption.py                            # 전부
"$PY" run_pictet_adoption.py --stages 0 1               # 일부
"$PY" run_pictet_adoption.py --summary-only             # 기존 출력으로 판정만

# 프로덕션 백테스트 단독
PYTHONPATH=. "$PY" run_variant.py --variant variants/iter15_65tkr_reb21_vtg.yaml
```

- **데이터**: `src/config.py:data_path`의 공유 Excel을 읽음(레포 밖). 위치 변경 시 그 한 줄만 수정.
- **독립성**: `PYTHONPATH=.`(ai_port)만으로 `from src.X`가 `ai_port/src`로 해석 — cc2_rl 체크아웃 불필요.
