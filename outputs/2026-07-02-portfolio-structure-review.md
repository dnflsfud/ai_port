# 포트폴리오 구성 파이프라인 구조 리뷰 (2026-07-02)

대상: `src/portfolio_optimizer.py`, `src/backtest.py`(simulate_portfolio·run_backtest), `src/config.py`, `src/utils.py`, `src/data_loader.py` + 실데이터 프로브 2회(ai_signal_data.xlsx 읽기 전용).
방법: 코드 정독 + 데이터 검증(late-entrant/중간결측/cap-weight 집중도). 6-24 전체 리뷰의 잔존 항목은 재확인만.

**요약: 총 9건 · Critical 1 · High 2 · Medium 3 · Low 3.** 모든 수정 후보는 baseline 변경 성격이므로 CLAUDE.md §6/§8 결정 로그 게이트 대상 (auto-fix 금지).

## 🔴 Critical

### 1. 상장 전 유령 데이터가 소스 xlsx에 baked-in — PLTR·GEV·BE
- 증거 (프로브 실측):
  - PLTR(상장 2020-09-30): 상장 전 2,462일 return=0, PX 상수 7.25, cap 상수 → BM weight ~0.135%. 상장일 return +31.0%.
  - GEV(스핀오프 2024-04-02): 상장 전 3,742일 99.95% zero-return, cap 상수 → BM weight 최대 0.69%.
  - BE(상장 2018-07-25): 상장 전 1,664일 return=0. 상장일 return +66.7% (15→25 유령 점프).
  - VRT는 실제 SPAC(GS Acquisition) 거래 이력이 있어 제외.
- 원인: data_loader의 median-impute가 아니라 **소스 파일의 상장 전 상수 backfill**. leading NaN이 없어 코드의 대체 경로는 미발동.
- 영향 경로:
  1. cap-weighted BM이 미존재 종목에 비중 배정 + 상장일 점프를 BM 수익률에 계상 (PLTR +31%×0.135%≈+4bp/일 등).
  2. zero-vol 종목이 공분산에서 사실상 무위험 취급 → 모델이 상장 전 구간에 양의 z를 주면 TE 소모 없이 OW 가능 (zero-cov + 양의 mu = MVO 공짜 알파).
  3. 학습 패널 오염: 학습은 2014년부터이므로 세 종목 모두 전 구간 해당. 백테스트 구간(~2019-)에서도 GEV는 ~5년 유령 상태.
  4. bm_weight_floor(2%×bm)로 책에도 미세 보유 강제 — 다만 규모는 무시 가능.
- 권고: (a) 먼저 영향 정량화 — result.daily_weights에서 상장 전 PLTR/GEV/BE 보유 여부·크기 확인, (b) 종목별 상장일 이전 마스킹(NaN)을 결정 로그에 등재 후 §8 절차로 적용. baseline 수치 변경 확실.

## 🟠 High

### 2. 동적 체결 confidence의 '신호 선명도' 항이 구조적으로 포화 — `backtest.py:850`
- predictions/raw_predictions는 날짜별 cross-sectional z-score(`model_trainer.py:240-244`) → top/bottom-6 spread ≈ 3+.
- `spread_score = clip(raw_spread/0.20, 0.20, 1.00)` ≡ 1.0 상시. 0.20 정규화는 codex_v2의 수익률-단위 예측 기준 캘리브레이션이 z-score 단위로 이식되며 무력화된 것.
- 결과: confidence = trailing-IC 항 단독. REDESIGN K의 "선명도 낮으면 덜 거래" 설계 의도 소실. 재캘리브레이션(z-단위 스케일 ≈3 기준)은 baseline 변경 → 결정 로그.

### 3. cap-weight 상한 경계 취약성 — top BM weight 14.75% vs `max_weight` 15% — `portfolio_optimizer.py:329,364-369`
- 표본 최대 정규화 cap weight 0.1475 (AAPL 2022-09-27). 여유 0.25%p. top>0.12인 날이 1,087/4,544 (24%).
- 초과 시 체인: 비-funding mega cap에 w≥bm + w≤0.15 → infeasible → ECOS 실패 → 전 리밸런스 조용한 BM fallback (그 BM 자체가 15% 캡 위반 상태로 보유됨). projection까지 실패하면 eta·턴오버 캡 무시 풀 점프.
- 권고: per-name cap을 `max(max_weight, bm_i)`로 완화 또는 infeasible 진단 로그. 현 데이터 미발동이므로 게이트 통과가 쉬운 방어적 변경.

## 🟡 Medium

### 4. mega_cap_protection: funding_mode=False면 무동작 — `portfolio_optimizer.py:341-369`
config.py:325-341 주석은 비대칭 보호(no_uw/wide_uw 존)를 설계로 기술하나 코드는 funding 모드만 구현. protection=True+funding=False는 제약을 하나도 안 만듦 → ablation 라벨 오독 위험.

### 5. TE 하드 제약이 의도적으로 왜곡된 공분산 위에서 계산 — `portfolio_optimizer.py:95-109`
mega-cap vol shrinkage로 축소된 Σ로 TE≤4.5%를 강제 → mega-cap active의 실제 TE 기여 과소평가. 리스크 예산 준수가 전적으로 ex-post 검증(invariant #5)에 의존. cap-weighted BM인데 mean_bm=1/n(>2/n 트리거)도 EW 시절 관례 잔재.

### 6. projection 실패 fallback이 풀스텝 target_weights — `backtest.py:1204-1222`
부분 체결(eta)로 줄이려던 트레이드가 projection 실패 시 최대 크기로 체결. MVO도 실패한 경우 target=bm → 턴오버 캡 무시 풀 점프. prev_weights(무거래) fallback이 취지에 부합.

## ⚪ Low
7. `bottom_indices` 죽은 변수 — `portfolio_optimizer.py:325` (항상 빈 set).
8. `max_single_turnover` 이름-의미 불일치 — per-name이 아닌 총 L1 턴오버 캡 — `portfolio_optimizer.py:376`.
9. validate_backtest 레거시 실패 휴리스틱 `w.std()≈0` — `backtest.py:1636-1639`. EW-BM 가정, cap-weighted에선 부정확 (현재는 dead path).

## 검증 후 기각 (클린 확인)
- 중간 결측 ffill 이중계상: 전 종목 내부 NaN 0 (KR 종목 포함) → 미발동.
- data_loader 대체 로직에 의한 유령 생성: leading NaN 없음 → 미발동 (원인은 소스 파일).
- 체결 타이밍 look-ahead: entering-weights PnL → drift → close 리밸런스(익일 반영) 순서 올바름.
- 공분산 룩어헤드 없음 (t 이전 구간만, raw_returns 사용).
- optimizer/projection의 `_build_mvo_constraints` 공유 — feasible region 일관성 좋은 구조.

## 6-24 리뷰 잔존 항목 (재확인, 여전히 열림)
IC 이중정의 `backtest.py:1269-1280`, fallback 판정 이중화 `backtest.py:1152-1156`, first-rebal 앵커 `backtest.py:906-911`, macro_cross ON-default `config.py:79`, data_path 절대경로 `config.py:30` — 모두 결정 로그 대기 상태 그대로.

## 적용 내역 (2026-07-02, #1~#7 구현 완료)

파이프라인: test-designer(합격 테스트 선작성) → implementer → verifier, 3개 유닛 전부 PASS. 테스트: 기존 39 → 총 67 (acceptance 23 신규 + stem 테스트 5 신규). #8(네이밍)·#9(레거시 휴리스틱)는 사용자 지시 범위 밖으로 미적용.

| # | 조치 | 방식 | default 상태 |
|---|---|---|---|
| 1 | listing mask 인프라 | `listing_mask_enabled=False` + `listing_dates`(PLTR/GEV/BE). ON 시 Daily_Returns·raw_returns·targets·predictions(상장일 포함 마스킹), PX_LAST·CUR_MKT_CAP(상장일 전만), capweight BM은 마스킹 티커 weight 0 | OFF (바이트동일) |
| 2 | confidence spread 스케일 노출 | `confidence_spread_scale=0.20`, `compute_signal_confidence(spread_scale=)` | 0.20 = 현행 (포화 유지) |
| 3 | per-name cap을 bm까지 완화 | `max_weight_per = maximum(max_weight, bm)` + 초과 시 warning | 항상 적용이나 현행 데이터에서 inert 증명(top bm 0.1475 < 0.15). bm>cap 최초 발생 시 S0와 달라질 수 있음 — S0 재인증 시 결정 로그 명시 필요 |
| 4 | mega-cap silent no-op 경고 | protection ON + funding OFF 시 logger.warning | 경고만, 동작 불변 |
| 5 | cov mega-cap vol shrink 플래그화 | `cov_megacap_vol_shrink_enabled=True` | ON = 현행 (ablation용 노출) |
| 6 | projection 실패 fallback 모드 | `projection_fallback_mode="target"` / "prev"(무거래) 옵션, `__post_init__` 검증 | "target" = 현행 |
| 7 | `bottom_indices` 죽은 변수 제거 | 직접 제거 | inert (항상 빈 set이었음) |

부수: TDD 가드 훅 요구로 stem 테스트 4파일 신규(tests/test_config.py, test_data_loader.py, test_backtest.py, test_portfolio_optimizer.py). OFF-default parity는 유닛별 verifier가 확인(기존 39 스위트 유지 + capweight OFF 경로 바이트동일 실측).

### 잔여 결정 (사용자/게이트 — §8 절차로만)
1. `listing_mask_enabled=True` 활성화: ablation(ON vs OFF 동일 솔버) 후 결정 로그 등재 → production variant flip.
2. `confidence_spread_scale` 재캘리브레이션: 사전등록 단일 값(z-단위 spread≈3 기준, 예: 3.5)으로 1회 실험 — 스윕 금지(invariant #4).
3. `projection_fallback_mode="prev"` 채택 여부: S0에서 projection fallback 발생률 0이면 inert flip.
4. verifier 권고(비차단): `confidence_spread_scale > 0` 가드 부재 — 0.0 override 시 ZeroDivisionError(시끄러운 실패라 수용).

### 후속: listing mask ablation (2026-07-02) — STOP / OFF 유지
OFF arm이 저장 baseline과 부동소수점 동일(7건 수정 parity 증명). ON arm은 v1이 타깃 엔진 dense 요구와 충돌해 설계 수정(Daily_Returns 시트 마스킹 제외) 후 완주 — IR 1.481→0.942(ΔIR −0.539, 서브기간 부호 일관)로 사전등록 게이트 ④ STOP. 단, 이 델타는 공분산 추정기 스왑 confound(raw_returns NaN → 2024-10 이전 전 리밸런스 LW→pairwise)가 지배적 의심이라 정화 효과의 측정치가 아님. 상세·후속 옵션: 결정 로그 §S6.
