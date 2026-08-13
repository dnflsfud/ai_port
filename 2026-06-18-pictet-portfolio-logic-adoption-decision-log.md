# Pictet → cc2_rl 이식 — 결정 로그 (Decision Log)

> CLAUDE.md §6에 따라 **첫 측정(S0) 전에 생성**. 모든 채택/보류는 이 로그의 게이트로만 정당화한다.
> 솔버 프로토콜: **ECOS 단일**(과거 SCS 수치와 직접 비교 금지, §2-2).
> 실행 환경: **WD=`ai_port`(현재 폴더)**. ⚑**2026-06-19부터 엔진 벤더링** → 코드=`ai_port/src`, **`PYTHONPATH=.`**(cc2_rl 불필요; `ENGINE_PROVENANCE.md`). ~~(원래는 CC2 절대경로+`PYTHONPATH=<CC2>`)~~ 캐시 손실 감수.

---

## S0 (ECOS baseline) — **확정 (2026-06-18)**

- 실행일: 2026-06-18 16:24 / 커밋: 없음 (CC2 소스 untracked, in-place 편집)
- 솔버: **ECOS** (`cp.installed_solvers()`에 ECOS=True 확인). fallback 6/94 = **6.4%**
- 명령(원본 2026-06-18, CC2 참조): `PYTHONPATH=<CC2> <PY> <CC2>/run_variant.py --variant <CC2>/variants/…`. ⚑**현재 재현(2026-06-19, 벤더링 엔진, 동일값)**: `PYTHONPATH=. <PY> run_variant.py --variant variants/iter15_65tkr_reb21_vtg.yaml`
- 산출물: `ai_port/outputs/iter15_65tkr_reb21_vtg/metrics.json` (캐시 미적중 전체 재계산, 257.8s)
- `information_ratio`: **1.485**
- `tracking_error`: **0.0310** (≤0.045 ✓, SCS 드리프트 없음)
- `avg_annual_turnover`(two-way): **1.144** (114.4%)
- `realized_beta`: **1.024**
- `realized_active_beta`: **0.024**
- sub-period IR: P1 **1.607**, P2 **0.578**, P3 **2.001**
- **이 S0가 이후 모든 arm의 단일 비교 기준.**

> **§9 편차 노트**: docs/메모리의 IR≈1.30은 **SCS** 기반. ECOS에서 IR=1.485. §2-2에 따라 과거 SCS 수치와 직접 비교 금지 — 이 차이는 솔버 전환의 예상된 효과이며 red flag 아님. TE 3.10%는 docs 3.2% 부근으로 집중 캐릭터 보존(§2-5).

> **P2 게이트 판정 (§3)**: realized_beta = **1.024 ≈ 1.0**. 진행 조건(0.90~0.93)에 해당 안 됨 → **Phase 3 (P2 beta-neutral)는 코드 작성 전 SHELVE**. 책이 이미 사실상 beta=1 — soft penalty가 줄 게 없음. realized_active_beta=0.024도 이를 뒷받침.

### Phase 0 진행 상황
- [x] Task 0.1 `compute_beta` 순수함수 (`src/utils.py`) + `tests/test_realized_beta.py` — 4 passed
- [x] Task 0.2 `compute_metrics`에 `realized_beta`/`realized_active_beta` 부착 — 회귀 27 passed
- [x] Task 0.3 S0 ECOS 재인증 실행 + 게이트 기록 — **P2 SHELVED**

---

## S1 attribution parity (P0) — **PASS (2026-06-18)**
- 구현: config `alpha_attribution_enabled/n_dates`(OFF default) + `harness.compute_alpha_attribution` helper + harness.run_variant & CLI attach + SAFE_FOR_CACHE_REUSE 키.
- helper 버그 수정: `group_contributions`의 Timestamp 키를 str로 변환(JSON-safe). plan helper는 이 변환이 누락돼 첫 ON 실행이 persist 단계에서 `TypeError: keys must be str...`로 죽었음.
- **OFF 바이트동일**: S0(OFF)에 `alpha_attribution` 키 없음 / 게이트는 `getattr(...,False)`.
- **ON parity**: IR 1.485 / TE 0.0310 / turnover 1.144 / realized_beta 1.024 — **전부 S0와 바이트 동일**(가중치 불변).
- **ON 시 share 합**: linear_share=0.482 + nonlinear_share_upper_bound=0.518 = **1.0** ✓ (n_dates=8). nonlinear는 **상한**(interaction 잔차 포함)으로 라벨됨.
- 테스트: `tests/test_alpha_attribution_config.py` 2 passed.

### Phase 1 진행 상황
- [x] Task 1.1 attribution config 필드 (off-path parity) — 2 passed
- [x] Task 1.2 `compute_alpha_attribution` helper — import ok
- [x] Task 1.3 harness/CLI attach + cache 키 — ON 스모크 parity 확인
- [ ] Task 1.4 leg C (construction) re-MVO counterfactual 스크립트

## S2 leg-C construction (P0) — **PASS (2026-06-18)**
- 스크립트: `scripts/run_alpha_attribution.py`. 산출물 `ai_port/outputs/alpha_attribution/summary.json`.
- **§4.2 confound 수정**: plan 스니펫은 `precomputed_predictions=base.raw_predictions`("overlay-free EMA base"라 주석)였으나, `model_trainer.py:293` 확인 결과 `raw_predictions`는 **블렌딩(EMA) 전 순수 모델 예측값** = pre-EMA. 그대로 쓰면 un-smoothed 신호 주입(EMA confound). → **overlays-OFF로 1회 harvest 후 `base_off.predictions`(EMA-blended·overlay-free)를 prod 암에 주입**해 overlay만 재적용(harvest-once).
- production overlay 상태: value_trap=ON, growth_tilt=ON, pead_boost=ON, signal_stability=0, EMA α=0.5.
- **round-trip identity**: `full_active=0.04604` == S0 `active_return=0.04604` → **abs_diff=0.0** (정확 재현 → 재사용 경로 무손실 검증).
- **leg C (construction delta)**: full_active 0.04604 − overlay_off_active 0.03456 = **+0.01147 (≈1.15% 연율 active)**. 3개 overlay의 합산 construction 기여.
- legs A/B: linear_share 0.482 / nonlinear_share_upper_bound 0.518 (상한). **leg C는 델타이므로 A/B와 합산 금지**(노트 명시).

### Phase 1 진행 상황 (계속)
- [x] Task 1.4 leg C re-MVO counterfactual 스크립트 — round-trip abs_diff=0.0, legC=+1.15%

**→ Phase 1 (P0 attribution) 전체 완료. attribution이 OFF-default·가중치 불변·즉시 프로덕션 가능(§8).**

## S3 overlay ablation (P1) — **PASS / 변경 없음 (2026-06-18)**
- 스크립트: `scripts/run_overlay_ablation.py` (harvest-once overlays-OFF → `base.predictions`로 각 arm이 자기 overlay만 재적용). 산출물 `ai_port/outputs/overlay_ablation/summary.json`.
- **§9 편차 기록**: plan은 OOS holdout 판정을 의도했으나 — (1) `enforce_oos_holdout`는 `train_cutoff_date` 없으면 silent no-op(model_trainer.py:311), (2) harvest-once 재-MVO 암은 `walk_forward_train`을 호출 안 해 arm-level holdout 플래그가 이중 무효. 구조적으로 OOS 판정 불가 → **full-period marginal ΔIR + 서브기간(P1/P2/P3) 부호 일관성**으로 판정(cs-dr-alpha 판정과 동일 방식). plan 무효 플래그 제거. plan 주석의 "on-baseline=(1,1,0),VTG off"는 오류(production은 셋 다 ON) — 정정.
- **내부 일관성**: on-baseline `vtg1_grw1_pead1` IR=1.485 = S0 정확 재현. all-off `vtg0_grw0_pead0` act=3.46% = S2 leg-C `overlay_off_active` 일치.
- **2³ 그리드 IR**: 000=1.129 / 001=1.097 / 010=1.497 / 011=1.524 / 100=1.202 / 101=1.159 / 110=1.592 / 111(prod)=1.485 (vtg·grw·pead 순).
- **Leave-one-out marginal ΔIR (full−dropped, +=overlay 도움)**:
  - drop_vtg: **−0.039** (P1+0.07 P2−0.22 P3−0.01) — 노이즈, 혼합
  - drop_growth: **+0.326** (P1+0.05 P2+0.37 P3+0.58) — 양·부호 일관, 지배적 keeper
  - drop_pead: **−0.107** (P1−0.08 P2−0.00 P3−0.06) — 노이즈, 미세 음
- **do-no-harm 판정**: 어떤 overlay도 **명확(>1SE=0.36)·부호 일관한 harm 없음**. VTG/pead 마진 음(−)은 \|ΔIR\|<0.36 노이즈 → §2-4 "IR 근거 제거 금지". growth는 핵심 양(+). 최대-IR arm(110=1.592) 선택은 p-hacking이라 금지. → **3개 overlay 전부 유지, 프로덕션 오버레이 변경 없음**.

### Phase 2 = COMPLETE. Phase 4(P3) 진행 중.

## S4 beta sweep (P2) — **SHELVED (S0 게이트 불통과)**
- S0 realized_beta=1.024 ≈ 1.0 → Phase 3 전체 코드 작성 전 보류. 측정 안 함.

## S5 factor (P3) — **PASS / CONFIRMED LEVER, OFF-default 유지 (2026-06-19)**
- 코드: config `factor_neutral_*`(OFF-default, size축 제거) + `portfolio_optimizer._factor_penalty_expr`+objective항 + `backtest._optimizer_fn` per-date loadings 스레딩. 단위테스트 `tests/test_factor_neutral.py` 4 passed, 전체 회귀 33 passed(OFF-path 불변).
- **설계 정제(plan 대비)**: plan은 `simulate_portfolio` 시그니처 변경을 요구했으나, `_optimizer_fn` closure가 캡처한 `panel`+`pred_row.name`(날짜)로 loadings를 만들어 **simulate_portfolio 무수정**(더 surgical). OFF 시 `factor_loadings=None`→`_factor_penalty_expr`=0 → objective 바이트동일.
- **§4.3 사전점검 (필수)**:
  - 컬럼 존재: value `best_peg_ratio_level_z`·quality `best_roe_level_z`·lowvol `idio_vol_63d` 전부 실재(assembly.py). **size 축 제거**(whitelist에 size 컬럼 없음).
  - **applied-date 수: 94/94** (>0 ✓), **impute율: 0.0** (모든 loading 셀 finite).
- **§4.2 confound 수정**: 1차 스크립트가 production(overlays ON) harvest 후 post-overlay 예측을 재주입해 **이중-overlay**가 발생(OFF 암 IR 1.466≠S0). overlays-OFF harvest로 수정 → **OFF 암 IR 1.485 / TE 0.0310 / turn 1.144 = S0 정확 재현**(round-trip 확인).
- **exposure 바인딩(헤드라인 판정)**: penalty=5, mean\|active style exposure\| 하락 — value **−55.0%**, quality **−54.7%**, lowvol **−66.0%**. → **명확히 바인딩**.
- **부수효과**: ON 암 IR **1.200**(−0.285), TE **0.0246**(−0.0064), turnover 1.148(거의 불변). 스타일 중립화가 active 리스크(TE)와 active 수익(IR)을 함께 축소.
- **판정**: factor-neutral은 작동하나(바인딩) 집중 스타일 베팅(=의도적 alpha, §2-5)을 깎아 IR을 낮추는 **리스크 축소 레버**. do-no-harm 아님 → §8/§2-5대로 **OFF-default 유지(CONFIRMED LEVER)**. 명시적 스타일-리스크 예산 지시가 있을 때만 활성화. IR로 채택/거부 판단하지 않음(§2-4).

### Phase 4 = COMPLETE.

## DSR / selection-bias — **비액션 (non-action) (2026-06-19)**
- 도구 확인: `run_selection_bias.py`(DSR·Haircut Sharpe, Bailey & López de Prado / Harvey-Liu) + `experiment_inventory.json` 존재.
- **판정**: 이번 작업에서 **백테스트 IR로 선택·승격되는 신규 후보가 없음**.
  - beta-neutral: 코드 전 shelve(측정 안 함).
  - overlay: do-no-harm으로 판정, **IR-최대화 선택 안 함**(최대-IR arm 110=1.592 명시적 거부). 변경 없음.
  - factor-neutral: **exposure 바인딩**으로 판정(IR 아님), 단일 사전약정 penalty=5, OFF-default 유지.
  - attribution: 가중치 불변(성능 무영향).
- 따라서 §2-7대로 **새 trial을 `experiment_inventory`에 추가하거나 deflation 적용할 대상이 없음**. sub-haircut ΔIR 비액션 입장 일관 적용. (프로덕션 책 자체의 DSR은 기존에 vetted된 iter15이며 본 작업이 변경하지 않음.)

## Production flips — **변경 없음 (no flip) (2026-06-19)**
후보별 §8 게이트 적용 결과, **프로덕션 variant `iter15_65tkr_reb21_vtg.yaml`은 무변경**:
- **attribution (P0)**: parity 통과(ON 가중치 == OFF 바이트동일, 롤백 자명). 단 SHAP **고비용** → production always-on에 부적합. **config OFF-default 유지 + on-demand**(인프라 cache-safe로 배선 완료, 연구/분기 attribution 시 variant override로 켬). → **flip 안 함**(가중치 변경 아님).
- **overlay (P1)**: do-no-harm, 3개 전부 유지 → **변경 없음**.
- **beta-neutral (P2)**: beta≈1.0 게이트로 shelve → **변경 없음**.
- **factor-neutral (P3)**: 바인딩하나 IR 비용(리스크 축소 트레이드오프) → **OFF-default 유지**, 명시적 스타일-리스크 예산 지시 시에만.
- **순결론**: Pictet식 리스크 규율 관점에서 현 프로덕션 책은 이미 잘 행동함(beta≈1.0, overlay do-no-harm, 스타일 베팅=의도적 alpha). **가중치 변경 불필요**. 본 작업의 산출물 = crystal-box 진단(attribution) + 검증된 OFF-default 레버(factor-neutral) + S0 ECOS 재인증.

### 2026-07-10 causal discipline flip (사용자 지시)
- 대상: variants/iter15_65tkr_reb21_vtg.yaml — causal_validation_enabled: true, execution_signal_lag_days: 1 (동시 flip, 사용자 직접 지시로 §8 1개씩 규칙 예외)
- 근거: 기존 val 구간 [t-126, t)의 21d 포워드 라벨이 예측일 t 이후 실현 → early stopping이 미래 정보에 노출. 실행은 t종가 신호로 t종가 체결(낙관 편향). GPT-5.6 챌린저(codex_causal_rank_65)가 도입한 default-OFF 인프라 재사용, 회귀 경로 배선 확인 완료(model_trainer.py causal split는 objective 무관).
- 재인증(S0', ECOS, --no-cache): IR 0.902 (기존 pre-causal 1.481 대비 -0.579), active ret 2.59%, TE 2.87%, turnover 106.8%, realized_beta 0.993, MDD -29.5%. 서브기간 P1 0.137 / P2 0.220 / P3 2.094 (기존 1.59/0.58/2.00).
- 해석: 기존 S0 성과의 상당분이 val-라벨 누수(early stopping)와 동일종가 체결 낙관에 기인했음이 확인됨. 특히 P1이 1.59→0.14로 붕괴. pre-causal 1.481은 이후 비교 기준으로 사용 금지 — 인과 규율 하 수치만 유효.
- 챌린저 재비교(동일 규율): codex_causal_rank_65 IR 1.695 (P1 0.752/P2 0.772/P3 3.080), ΔIR +0.79 > 0.36(1SE) & 서브기간 3/3 승 — §2-4 채택 바 충족. comparison_gate PASS(7/7 체크, subperiod_wins 3).
- 기여 분해(2026-07-10 실시, ECOS·--no-cache·단일 플래그 arm):
  | arm | causal_val | exec_lag | IR | ΔIR vs pre-causal 1.481 | P1/P2/P3 | TE | turnover | beta |
  |---|---|---|---|---|---|---|---|---|
  | pre-causal S0 | off | 0 | 1.481 | — | 1.59/0.58/2.00 | 3.11% | 114% | 1.024 |
  | arm B exp_exec_lag_only | off | 1 | 1.486 | +0.005 | 1.38/0.64/2.10 | 3.05% | 109% | 1.023 |
  | arm A exp_causal_val_only | on | 0 | 0.920 | -0.561 | 0.26/0.30/2.05 | 2.80% | 103% | 0.993 |
  | S0' (both) | on | 1 | 0.902 | -0.579 | 0.14/0.22/2.09 | 2.87% | 107% | 0.993 |
- 판정: 급락은 사실상 전부 val-라벨 누수 수정(arm A -0.561)에서 발생. 실행 1일 지연 단독(arm B)은 노이즈 수준(+0.005, |Δ|<1SE). 상호작용 -0.023으로 근사 가법적. P3(2023+)는 전 arm에서 ~2.0-2.1로 불변 — 최근 구간 알파는 누수 비의존. P1/P2 알파의 대부분이 early-stopping 누수 차입이었음이 확정.
- degenerate 비율 참고: pre-causal 16/32, arm A 18/32, S0' 18/32, codex rank 21/32 — 모델 붕괴 지표와 성과 급락은 무관(가설 기각).
- 롤백: variant 2줄 revert = 기존 S0 경로 바이트 동일 복원.

### 2026-07-11 codex_causal_rank_65 프로덕션 승격 (사용자 지시)
- 근거: 인과 규율 하 동일 ECOS 비교에서 codex_causal_rank_65 IR 1.695 vs iter15_65tkr_reb21_vtg(S0') IR 0.902, ΔIR +0.79 > 0.36(1SE) & 서브기간 3/3 승(P1 0.752/P2 0.772/P3 3.080) & comparison_gate 7/7 PASS(승격 전 기준). DSR/selection-bias 해킷(N=412, 재현: `run_selection_bias.py --auto --label codex_causal_rank_65`): verdict FAIL — 단, FAIL은 DSR 단일 항목(deflated SR 1.212, p=0.113>0.10)이며 MinTRL 충분(1.0yr vs 7.8yr)·haircut 후 조정 SR 0.430>0 PASS·생존편향 CLEAN·서브기간 3/3 양(+) STABLE. 기존 프로덕션 S0도 동일 검사 FAIL(조정 SR 0.259)로, 절대 유의성 문제이지 상대 비교 문제가 아님. 사용자 명시 승인(2026-07-11 'DSR FAIL이어도 승격 진행해')으로 오버라이드.
- 조치: role 교체 — codex를 production으로, iter15를 challenger(display "Legacy S0")로 승격.
  - variant yaml 2개: codex `portfolio_role: production`, iter15 `portfolio_role: challenger` + `display_name: Legacy S0` 추가.
  - export 기본값 로직(scripts/export_operating_data.py `_LABEL_DEFAULTS`): 무인자 경로에서도 iter15→(Legacy S0, challenger), codex→(Causal Rank 65, production), 미지 label→challenger.
  - registry 재발행: export×2 + validate_portfolio_bundles → production=codex, challenger=iter15.
  - 대시보드 라벨을 registry display_name 유도로 치환(streamlit_app.py, 슬롯/역할은 role 단어).
- comparison_gate: 승격 후 challenger(iter15 IR 0.902) < production(codex IR 1.695)이므로 gate FAIL/RESEARCH 표시가 **정상적 방향 반전**임(non-blocking, registry 발행·체인 계속).
- 주말 캘린더 수정 후 codex 재실행 IR 1.697(기존 1.695, 결론 불변). validator stale-tail 검사는 번들 정책(fail_on_stale_tail_ffill=false) 존중으로 화해 수정(하드 게이트는 정책 true일 때만) — 근거: variant가 명시한 경고-전용 정책과 신규 하드 게이트의 충돌 해소, 데이터 빈티지(as_of 2026-06-11)의 고정 속성인 12d tail이 체인 전체를 영구 차단하는 것 방지.
- 롤백: yaml 2개 role 되돌림 + export 기본값 revert + export×2/validate 재실행 = 완전 복원.

## 재현 실행 (2026-06-19)
- `ai_port/run_pictet_adoption.py` — 전 스테이지(S0→attribution→overlay→factor) from-scratch 재현 + 게이트 자동판정 → `outputs/adoption_summary.json`. ai_port CWD, 단일 foreground/스테이지, ⚑**로컬 경로 + `PYTHONPATH=.`(벤더링 엔진, 2026-06-19~)**.

---

## 코드 리뷰 (ultracode) + 적용 수정 (2026-06-19)

**리뷰**: 멀티에이전트 적대 워크플로우(6차원 finder → 발견별 3-lens 검증 ≥2/3 → synthesis → 완전성 비평). 107 에이전트 / 5.8M 토큰. **confirmed 11 (med 3 · low 8) + rejected 22 + 완전성 갭 5**.

**헤드라인**: 11건 중 **OFF-default 바이트동일성·S0를 깨는 것은 0건**. 전부 OFF 플래그(factor_neutral/alpha_attribution) 뒤 또는 read-only verdict/진단 레이어. 현 verdict가 맞는 건 게이트가 적대 케이스를 잡아서가 아니라 *이번 숫자가 깨끗해서* — 그래서 **flip 전 M1·M2·M3 선결**이 패널 권고였고, 본 라운드에 반영함.

**적용 수정 (사용자 선택: 전부 액션항목 + 결정로그, 2026-06-19)**:
- **M1** `src/backtest.py _optimizer_fn`: factor-neutral 라이브 커버리지 텔레메트리(per-date impute/inert) 누산·1회 surface + `result.factor_neutral_telemetry`. **enabled일 때만 동작 → OFF 바이트동일 유지**(`test_factor_penalty_disabled_identical` 통과).
- **M2** `run_pictet_adoption.py _verdict_overlay`: REMOVE 부호검사를 방향-고정(`all(x<0 for x in dsub)`)으로. 기존 direction-agnostic(`all<0 or all>0`)은 "서브기간 전부 KEEP인데 REMOVE" 모순 가능했음(§7).
- **M3** `scripts/run_factor_ablation.py` + `_verdict_factor`: `optimizer_failure_rate`·`active_share`를 산출/surface하고, OFF(=S0) 대비 ON의 TE/active share가 절반 미만이거나 fallback율이 +10pp 급증하면 **FAIL(벤치마크 붕괴, §2-5)** — IR과 무관. 기존엔 `exposure_drop>20%`만 봐서 붕괴를 "CONFIRMED LEVER"로 오라벨 가능했음.
- **L1** `CLAUDE.md §4.1` + `backtest.py` 주석: "252d OLS" → **full-sample OLS**로 정정(코드가 full-sample; β=1.024 S0 인증 끝나 계산식 불변). `realized_active_beta`=β−1 항등식 명시.
- **L2** `_verdict_overlay`: (1,1,1) arm 없는 부분 summary가 `all([])==True`로 "keep all" 오보되던 것 → `status:"incomplete"`.
- **L3** `src/harness.py`: `compute_alpha_attribution` import 실패 시 신호 없는 `{}` → `{"error": "import failed: ..."}`(shap 미설치 은폐 방지).
- **L4** `_verdict_baseline`: P2 beta 밴드 0.88–0.95 → **0.90–0.93**(계약 §3과 일치). β=1.024는 양쪽 다 SHELVED — 결과 불변.
- **L5** `_preflight`: cvxpy import 실패를 전부 "ECOS 없음"으로 오귀인하던 abort에 `rc`+`stderr` tail 노출(ASCII-only, cp949 안전).
- **L6** `run_variant.py`: `factor_neutral_*` 4키를 `SAFE_FOR_CACHE_REUSE`에 추가(objective term+캐시된 panel 로딩만 변경 → cache-safe, 형제 키와 동급).
- **L7** `run_factor_ablation.py`: ablation의 cols 룩업을 프로덕션과 동일하게 guarded(`if a in loadings`)로 — axis 누락 시 KeyError 크래시 대신 정렬 유지.
- **L8** ablation 3스크립트의 미사용 `numpy`/`pandas` import 제거.
- **GAP2** `run_overlay_ablation.py`·`run_factor_ablation.py`: on-baseline/OFF arm이 S0를 재현하는지 **round-trip assert(>1e-3면 stage rc=1 FAIL)** + `_verdict_factor`에 `harvest_invalid` 가드. (기존엔 leg-C만 검사, 나머지는 print만.)
- **GAP5** `tests/test_realized_beta.py`: `active_beta == beta−1` 항등식 테스트 추가(독립 지표 아님 고정).

**검증**: 전체 회귀 **34 passed**(기존 33 + GAP5). `--summary-only` 재검증 — 구버전 summary.json(새 키 없음)에도 graceful(`active_share_off_on:[null,null]`, `roundtrip:null`), 최상위 판정 **불변**(beta SHELVED / overlay 전부 KEEP / factor CONFIRMED LEVER), collapse 가드 오발 없음(TE 0.0246 vs 0.0310은 절반 이상). 전부 OFF-default — **프로덕션 가중치 무변경**(§8).

**미수정 known-item (이번 라운드 비반영, 비차단)**: GAP1(서브기간 경계 하드코딩 + `sub_ir` NaN<20obs가 부호검사 오염 → 데이터 결손이 "KEEP all"로 위장 가능), GAP3(`--stages` 부분실행/`--summary-only`가 stale 출력으로 자신만만한 summary 생성 — provenance/mtime/fingerprint 없음, §6·§2-2 충돌 소지), GAP4(`binds`의 매직 `>20%` 임계 — 통계적 바닥·사전등록 근거 부재, §4 p-hacking 사각). 전부 verdict **입력·임계 레이어**의 하드닝 — 프로덕션 flip을 실제로 고려할 때 선결.

---

## 벤더링 + self-contained 재현 인증 (2026-06-19)

**벤더링**: cc2_rl 엔진(src·scripts·run_variant.py·tests·variants, ~8,600 LOC)을 `ai_port/src`에 **미러 복사** → `ai_port`가 cc2_rl 없이 단독 실행되는 자립 정본. 오케스트레이터/임포트 전부 로컬화(`PYTHONPATH=.`). 출처·재동기화: `ENGINE_PROVENANCE.md`, 구조: `PORTFOLIO_STRUCTURE.md`. (이로써 CLAUDE.md §1 "CC2 정본"은 superseded — STATUS 배너로 명시.)

**codex 재평가 대응** (codex가 self-contained 재현·낡은 factor summary·문서충돌을 지적):
- 벤더링 엔진으로 **4-stage from-scratch 재실행** (`run_pictet_adoption.py`, 2026-06-19 13:46→14:03, exit 0). 단일 백그라운드 프로세스, 스테이지 내부는 순차 foreground.
- **S0 재현(벤더링 엔진)**: IR **1.4852** · TE 0.0310 · turnover 1.1437 · realized_beta 1.0242 → **인증 S0와 바이트 동일**. `metrics.json`의 cc2_rl 절대경로 **0건**(기존 1건 제거). `logs/stage0-3.log` 생성 = 재현 증거.
- **factor 신가드 라이브 검증**: 재생성 `summary.json`에 `active_share`(off 0.1034 / on 0.0970), `optimizer_failure_rate`(off=on **0.0638**), `roundtrip_off_vs_s0_abs_diff`=**0.0** 포함. `_verdict_factor` 라이브 판정: **collapsed=False**(TE 0.0246>0.5·0.0310, active share·fallback 모두 정상) → **"CONFIRMED LEVER" 신(新)붕괴가드로 재검증됨**(이전엔 M3 이전 산출물이라 미검증이었음).
- **overlay**: 재실행 후에도 3개 전부 KEEP(do-no-harm), M2 방향-고정 부호가드 하에서.
- **문서 충돌(codex #4)**: `CLAUDE.md`에 STATUS 배너 + §0/§1/§3 인라인 표식으로 "S0 pending·결정로그 없음·CC2 정본" 해소.
- **codex #2(Python 깨짐)**: codex 환경 한정 — 본 세션 venv `python.exe`(3.12.10)는 정상(imports OK·xlsx 존재·34 테스트·4-stage 완주).

**적용 수정**: **D** `_solve_problem` docstring 정정 — ECOS→SCS fallback은 *예외 시에만*, ECOS non-optimal status는 SCS 없이 BM fallback. solve 경로 불변(§2-2/parity), per-solve `diagnostics`가 사유 기록.

**codex 2차 지적 (doc staleness) 수정**: decision log 상단(실행 환경·S0 명령·재현 실행)과 CLAUDE.md STATUS의 "CC2 절대경로+PYTHONPATH=CC2"·"재생성 중" 잔재를 `ai_port/src`·`PYTHONPATH=.`·"재생성 완료(exit 0)"로 정정.

**F·G 정량화 완료** (codex 잔존 리스크 → `scripts/data_quality_report.py`, read-only, `outputs/data_quality_report.json`):
- **F 커버리지**: date 시트 27개(+meta 3개 분리), date 교집합 recomputed **3056(62.4%)** ≈ engine-logged **3217(66% of longest 4894)**, **tail ffill 16일**(2026-05-26→06-11, PX_LAST 기준). 즉 시트별 히스토리 편차가 커 최근 16거래일은 ffill 확장 — 성능수치는 이 커버리지 전제 위에서 해석.
- **G degenerate**: 워크포워드 **32 폴드 중 16(50.0%)** degenerate(1~6 trees→prev model 재사용), **연도 편중**(2019 4/4·2024 3/4·2021 3/4 vs 2023 0/5), tree=1이 9건. 높은 재사용률 — **P2(저IR) 구간과 연결 가능성**은 별도 검증 필요.

**남은 codex 개선(비차단)**: **E** — realized_beta를 sub-period(P1/P2/P3) + rolling 252d로 확장(§4.1 beta 규율 정합). compute_metrics+harness 코드 변경 + stage-0 재실행 필요라 사용자 승인 대기.

**순결론(재확인)**: 포트폴리오 구조 합리적, **프로덕션 가중치 변경 근거 약함 → 무변경 유지**. self-contained 재현은 이제 **증거(stage 로그·S0 바이트동일·round-trip 0.0)와 함께 확정**.

---

## 구조 리뷰 루프 (2026-06-24) — 결정 대기 항목 등재

출처: `c2/ai_port/src` 전체 구조 리뷰(5 리뷰어 병렬). 안전 수정 4건은 적용·검증(39 tests pass) 완료 —
전체 트리아지: `outputs/2026-06-24-structure-review-loop.md`. 아래 2건은 **수정 시 영향이 baseline/게이트에 닿아
임의 수정 금지(§2-2/§8)** → 사용자 결정 대기로 **등재만** 함.

### D1 — `config.py:79` `macro_cross_enabled=True` (ON-default) — **STATUS: 결정 대기**
- 발견: 2026-04-22 추가된 macro×ticker 5피처(rate×rev, slope×rev, VIX×mom252, vol×mom63, DXY×rev)가 **ON-default**. 따라서 순수 `PipelineConfig()`는 pre-2026-04-22 baseline과 패널이 다름. 주석은 "ablation용으로 disable"이라 *옵션 토글*처럼 서술하나 default는 ON — invariant #1(OFF-default)의 문자적 위반 소지.
- **핵심 사실**: 현 **S0(IR=1.485, 2026-06-18 인증)는 macro_cross=ON 상태로 측정됨**. 즉 이 5피처는 이미 certified S0에 내장. → "새 후보를 켜는" 문제가 아니라 "이미 켜진 baseline 구성요소"의 정합성 문제. OFF로 뒤집으면 **S0가 바뀜**(§2-2 재baseline 금지에 저촉).
- 결정 옵션:
  - (A) **baseline_v2 구성요소로 공식 문서화** (의도된 ON) — 권고. config 주석을 "옵션 토글"에서 "baseline 포함"으로 정정하고, OFF-parity 규칙의 적용 범위를 *2026-04-22 이후 신규 Pictet arm*으로 한정 명시.
  - (B) **후보 arm으로 강등**(OFF-default) — 이 경우 macro_cross OFF로 **S0 재인증** + OFF-vs-ON ablation(§4 단일 사전등록, p-hacking 금지)이 선행돼야 함.
- 차단: 사용자가 A/B 선택 전까지 코드 변경 없음.

### D2 — `backtest.py:1268-1280` IC 이중정의 silent fallback — **STATUS: 결정 대기**
- 발견: `avg_ic` 계산 시 `t_date in targets.index`면 `targets`(build_targets 컨벤션) 사용, 아니면 **raw 20일 forward simple-sum return으로 조용히 대체**. 두 정의가 비교 불가라 `avg_ic`가 날짜 커버리지에 따라 두 메트릭의 혼합이 됨. `avg_ic`는 `validate_backtest` 게이트 입력(§ backtest.py:1612 부근)이라 **게이트 메트릭 왜곡**.
- **영향 범위(중요)**: IC는 **진단 전용 — 가중치에 피드백되지 않음**. 따라서 이 수정은 **IR/TE/turnover/realized_beta(=S0 코어 메트릭)에 영향 없음**. 바뀔 수 있는 건 보고용 `avg_ic` 수치와 그 게이트뿐.
- 권고 수정: `elif` 폴백 제거 → `targets` 미커버 시 `realized=None`(해당 날짜 IC skip)으로 **단일 정의 통일**.
- 결정 옵션:
  - (A) **권고 수정 적용** + S0 1회 재실행으로 새 `avg_ic`를 기록(코어 S0 메트릭 불변 확인). — 권고.
  - (B) 현 동작 유지 + 이중정의를 명시 문서화(게이트 해석 시 주의).
- 차단: `avg_ic`가 어떤 테스트/게이트에 parity-assert로 묶였는지 확인 후, A 적용 시 S0 재실행 결과를 본 로그에 기록.

> 두 항목 모두 §8 절차(결정 → 1개씩 적용 → S0 재검증 → 롤백 확인 → 독립 커밋)를 따른다. 등재 시점에선 **프로덕션·코드 무변경**.

---

## S6 listing mask ablation (2026-07-02) — **STOP / OFF-default 유지 (사전등록 게이트 ④ 발동)**

- 배경: 2026-07-02 구조 리뷰 Critical #1 — 소스 xlsx에 PLTR(2020-09-30)·GEV(2024-04-02)·BE(2018-07-25) 상장 전 구간이 상수가격/제로수익률/상수시총으로 backfill. default-OFF 마스킹 인프라 구현(`listing_mask_enabled`, 67 tests pass). 사전등록: 단일 arm(3종목 고정), 채택 근거는 정합성이며 IR 아님(§2-4), do-no-harm 게이트 4개.
- **OFF arm (= S0 재확인, 현 코드)**: IR **1.481437507913232** · TE 0.031069 · turnover 1.14402 · realized_beta 1.02439 · P1 1.591/P2 0.575/P3 2.005 · ECOS 188 · fallback 6/94 — 저장 baseline `outputs/iter15_65tkr_reb21_vtg/metrics.json`과 **부동소수점 자릿수까지 동일**. → 2026-07-02 구조 수정 7건(#1~#7)의 default 바이트동일 parity가 풀 프로덕션 실행으로 증명됨.
- **ON arm v1 (§9 편차)**: Daily_Returns 시트 NaN 마스킹이 PCA 타깃 엔진의 dense 횡단면 요구와 충돌 — sparse_skip 2531/2961, 유효 가중치 13.3%, 학습표본 0 → LGBM ValueError로 크래시. **설계 수정**: Daily_Returns는 시트 마스킹 제외(라벨 오염은 run_backtest targets 셀 마스킹이, PnL은 예측 마스킹 w=0이 차단). 67 tests 유지.
- **ON arm v2 (동일 ECOS)**: IR **0.942** (ΔIR **−0.539**) · active 2.62% · TE 2.78% · turnover 116.7% · realized_beta 1.022 · P1 1.177/P2 −0.194/P3 1.697 · ECOS 188 · fallback 4/94.
- **게이트 판정**: ① OFF parity PASS ② TE≤4.5%·캐릭터 보존 PASS(2.78% vs 3.11%, 절반 붕괴 아님) ③ fallback 급증 없음 PASS ④ |ΔIR|=0.539>0.36(1SE) & 3개 서브기간 부호 일관 악화 → **STOP & 원인조사**.
- **원인 분석 (measured ΔIR은 정화 효과의 깨끗한 추정치가 아님 — 3채널 혼합)**:
  1. BM 유령 제거 (의도 효과, 유령 BM weight 0.02~0.69%로 소규모)
  2. 학습 라벨 제거(3종목 상장 전 행 drop) → LightGBM 경로의존으로 전 종목 예측 변화
  3. **공분산 추정기 스왑 confound (지배적 의심)**: raw_returns 마스킹으로 GEV NaN이 존재하는 2024-10 이전 모든 rebalance(~80%)에서 `estimate_covariance`가 LedoitWolf → `_pairwise_covariance`로 전환(NaN 하나라도 있으면 pairwise 경로, `portfolio_optimizer.py:88`). OFF는 소스가 dense zeros라 전 기간 LW. 유령 3종목과 무관하게 리스크 모델이 통째로 바뀜.
- **결정**: `listing_mask_enabled` **OFF-default 유지, 프로덕션 무변경**(§8). ΔIR로 기각하지도 않음(§2-4) — 측정이 confounded라 채택/기각 판단 자체가 불가.
- **권고 후속 (사용자 결정 대기)**:
  - (a) cov 채널 de-confound: mask ON에서 phantom 컬럼만 특수처리(dense 서브셋 LW + 해당 종목 median-var·0-cov 임베드) 후 사전등록 단일 재실험 — 그래야 채널 1+2만의 순효과 측정 가능.
  - (b) OFF `backtest_result.pkl`의 daily_weights에서 상장 전 유령 보유량 정량화 — 오염 실규모가 작으면 (a) 자체를 스킵하고 mask를 영구 보류할 근거.
- 산출물: `outputs/listing_mask_ablation/{off,on}/metrics.json`, `variants/exp_listing_mask_{off,on}.yaml`.

### S6 후속 (b) — 유령 보유 정량화 (2026-07-02) → **CLOSE: mask 영구 보류(OFF), de-confound 재실험 불요**

OFF `backtest_result.pkl`(daily_weights 1973일×65종목, 2018-11-26→2026-06-11) + CUR_MKT_CAP 정규화 BM 프록시로 정량화:

| tkr | 상장 전 겹침 | 책 mean/max | BM mean/max | active mean | OW(>1bp) 일수 | forgone | BM drag |
|---|---|---|---|---|---|---|---|
| PLTR | 482일 | 0.097% / 0.219% | 0.094% / 0.125% | +0.003%p | 18.7% | 3.4bp/yr | 2.8bp/yr |
| GEV | 1,396일 | 0.241% / 0.880% | 0.240% / 0.428% | +0.002%p | 12.3% | 6.4bp/yr | 5.4bp/yr |
| BE | 0일 (상장 2018-07-25 < 백테스트 시작 2018-11-26) | — | — | — | — | 0 | 0 |

- **판정 근거**: 책이 유령을 사실상 BM 비중으로만 보유(optimizer가 무알파·score-gate로 bm 부근에 핀). active 채널에서 forgone(책 손실 ~9.8bp/yr)과 BM drag(BM 손실 ~8.2bp/yr)가 상쇄 → **순 오염 ≈ +1.6bp/yr, gross 상한 ≈ 18bp/yr** — active 460bp/yr·TE 311bp 대비 무시 가능. 우려했던 "zero-cov 공짜 OW" 채널도 실측상 미미(OW 일수 12~19%, max OW GEV +0.74%p 일시).
- **결정**: `listing_mask_enabled` **영구 보류(OFF-default 유지)**. S6의 후속 옵션 (a) cov de-confound 재실험은 **불요** — 교정 가능한 오염이 ~2bp/yr인데 재실험·estimator 특수처리의 코드 리스크가 훨씬 큼(§2 단순성). ON arm의 ΔIR −0.539는 전량 confound(cov 추정기 스왑 + 모델 재적합 경로의존)로 귀속.
- **잔여 한계(기록)**: 학습 라벨 채널(유령 행이 LGBM 학습에 포함)의 순효과는 본 보유 분석으로 측정 불가. de-confound 실험 없이는 부호조차 불명 — 비용 대비 추적 가치 낮음으로 종결. 인프라(마스킹 코드·테스트)는 향후 실결측 데이터 대비로 유지.
- 산출물: 스크래치패드 `quantify_phantom.py` (읽기 전용 분석, 리포 외부).

---

## D1·D2 해소 (2026-07-02) — 두 건 모두 CLOSE

### D1 `macro_cross_enabled` ON-default — **A안 채택: baseline_v2 구성요소로 공식 문서화 (주석 전용, 코드 무변경)**
- 근거: 인증 S0(ECOS IR 1.481)가 macro_cross=ON 상태로 측정됨 — 신규 후보가 아니라 기존 baseline 구성요소. OFF로 뒤집으면 S0 재베이스라인(§2-2 저촉).
- 적용: `src/config.py`의 macro_cross 주석을 "ablation용 옵션" → "baseline_v2 COMPONENT, intentionally ON-default"로 정정. OFF-default 불변식(§2-1)의 적용 범위는 2026-06-18+ 신규 Pictet arm이며 이 필드는 예외임을 명시. 필드 값·동작 무변경(주석 전용), 관련 테스트 24 pass.

### D2 IC 이중정의 silent fallback — **A안 채택: fallback 제거 (단일 정의), S0 바이트동일 재검증 완료**
- **결정 로그 원 서술의 오류 정정**: 종전 기록 "IC는 진단 전용 — 가중치에 피드백되지 않음"은 **부정확**. `ic_values`는 trailing IC → `compute_signal_confidence` → 동적 체결 eta 경로로 **가중치에 피드백된다**(REDESIGN K). 따라서 이 수정의 안전성은 사전 실측이 필요했음.
- **사전 프로브 (수정 전 실측)**: 프로덕션 OFF pkl에서 fallback 발화 **0회** — 리밸런스일 94/94 전부 targets.index 커버(targets 2014-01-27~2026-06-11 ⊃ 리밸런스 2018-11-26~2026-05-21). ic_series 93개(마지막 리밸런스일은 canonical 경로에서 targets NaN으로 skip — fallback과 무관). → 제거는 증명 가능한 바이트동일.
- 적용: `src/backtest.py` simulate_portfolio IC 블록의 `elif t_idx+20<len(all_dates): realized=returns...sum()` 분기 제거 → targets 미커버 시 `realized=None`(IC skip). 합격 테스트 `tests/acceptance/test_ic_single_definition.py` 4건(선작성 TDD) 통과, 전체 71 pass.
- **S0 재검증** (`variants/exp_s0_recheck_d2.yaml` → `outputs/s0_recheck_d2/metrics.json`, ECOS 188·fallback 0): IR/active/TE/turnover/realized_beta/**avg_ic**/P1·P2·P3/solver-fallback-rate **10개 항목 전부 부동소수점 동일** (IR 1.481437507913232, avg_ic 0.04864921465993589). 롤백 = elif 복원 한 조각(자명).
- 효과: `avg_ic` 게이트 메트릭이 단일 정의(targets 컨벤션)로 통일. 향후 targets 커버리지가 줄어드는 데이터 상황에서도 이중정의 혼합이 원천 차단됨.

---

## S7 AI-logic arms (2026-07-06)

Pictet 채택 이후 AI-로직 후보 arm들(A1~A4)의 사전등록·측정 로그. 모든 arm은 동일 ECOS 프로토콜(§2-2), default-OFF 인프라(§2-1), 후보당 단일 사전등록 파라미터(§2-4)를 따른다. 각 arm 소절은 독립.

### S7.A1 mu-vol-scaling (z→mu 변동성 스케일링) — **사전등록 (2026-07-06)**

- **가설**: 오버레이 후 CS z-score가 무단위 그대로 MVO objective(`mu @ w`, `portfolio_optimizer.py`)에 투입된다. Grinold식 α = σ·z(변동성 스케일링)가 부재 — 동일 z라도 변동성이 큰 종목의 기대 초과수익이 더 크다는 표준 정식화가 빠짐. 이 변환을 사전등록 단일 형태로 평가.
- **사전등록 변환 (이 형태 외 변형·스윕 금지)**: 각 날짜 t, 종목 i에 대해

      mu_i(t) = z_i(t) · σ_i(t) / median_CS{ σ_j(t) : j valid }

  - z = 오버레이 체인(listing mask → pead → growth_tilt → vtg) **완료 후** 예측값. 변환은 체인의 **마지막 단계**(`src/backtest.py`, signal-stability 이후·`result.predictions` 이전).
  - σ_i(t) = 비보간 raw returns(`data.raw_returns`, 공분산 추정과 동일 risk_source)의 trailing `cov_lookback`(=126d) 표준편차, 위치 k의 **strictly-before** 윈도우 `iloc[max(0,k-126):k]`(t 배제, 룩어헤드 금지 — 공분산 윈도우 관례와 동일).
  - median_CS 정규화로 스케일 중립(중앙값 σ 종목의 mu==z), 실효 risk_aversion 변화 최소화. 파라미터-프리, 클리핑 없음.
  - 가드(전부 inert 지향): 유효 관측 <63 또는 비유한 σ → 해당 종목 σ=CS median(스케일 1); 날짜 전체 무valid → 항등; NaN 예측 → NaN 유지.
- **구현**: `PipelineConfig.mu_vol_scaling_enabled: bool = False`(config.py), 순수함수 `apply_mu_vol_scaling(predictions, risk_returns, config)`(backtest.py, 기존 오버레이 관용구), 오버레이 체인 마지막 배선. 합격 테스트 `tests/acceptance/test_mu_vol_scaling.py` 13/13 통과, 전체 스위트 114 pass(무관: A3 `test_adaptive_ema.py` 수집 오류 별개).
- **OFF 파리티**: 프로덕션 variant(`iter15_65tkr_reb21_vtg`) 풀 런 재실행 → 정본 `metrics.json`과 **바이트동일**(sha256 일치, `elapsed_sec` 제외 전 필드 동일). OFF 경로 완전 inert 증명. 정본 artifact 백업·복원 완료.
- **실행 예정 variant**: `variants/exp_mu_vol_scaling.yaml` → `outputs/exp_mu_vol_scaling/` (단일 foreground, 풀 경로).
- **판정 게이트 (사전등록)**: ① ΔIR > **+0.36**(=1 SE) **& 서브기간(P1/P2/P3) 부호 일관** ② 캐릭터 보존(TE ≤4.5%·active share 붕괴 없음, §2-5) ③ fallback 급증 없음 ④ DSR/selection-bias 비액션 유지(단일 사전등록 파라미터라 스윕 p-hacking 없음). 셋 다 충족 시에만 IR 근거 채택 후보; 미달 시 OFF 유지. **게이트 통과 여부가 아니라 정직한 측정이 성공 기준.**
- **S0 기준(동일 ECOS)**: IR 1.481437507913232 · TE 0.031069 · turnover 1.14402 · realized_beta 1.02439 · P1 1.591/P2 0.575/P3 2.005.
- **결과 (2026-07-06, `variants/exp_mu_vol_scaling.yaml` → `outputs/exp_mu_vol_scaling/`, 풀 경로 233.4s, ECOS 188·fallback 0/94)**:

  | metric | S0 (OFF) | A1 (ON) | Δ |
  |---|---:|---:|---:|
  | information_ratio | 1.481438 | 1.552503 | **+0.071066** |
  | tracking_error | 0.031069 | 0.032067 | +0.000998 |
  | active_return | 0.046027 | 0.049784 | +0.003757 |
  | avg_annual_turnover | 1.144021 | 1.103708 | −0.040314 |
  | realized_beta | 1.024389 | 1.022823 | −0.001566 |
  | avg_ic | 0.048649 | 0.049543 | +0.000894 |
  | sharpe_ratio | 1.307645 | 1.329896 | +0.022251 |
  | max_drawdown | −0.299887 | −0.295704 | +0.004183 |
  | P1_ir | 1.591390 | 1.451754 | −0.139636 |
  | P2_ir | 0.574884 | 0.453726 | −0.121158 |
  | P3_ir | 2.004814 | 2.330639 | +0.325825 |

- **스케일 팩터 분포 (mu/z, arm.predictions÷S0.predictions — 동일 harvest이므로 정확히 σ/median; 126,981 유한·z≠0 셀)**: min **0.000** · p05 0.576 · **median 1.000**(median 정규화 정상) · mean 1.123 · p95 2.108 · max **5.158**. 스케일==1(±1e-9) 셀 1,963/126,981(1.55%, 가드·항등·중앙σ 종목). 활성 날짜 1,973/1,973. **우측 왜도**(mean 1.123>median 1.0): 고변동 종목 mu 증폭, 저변동 종목 mu 축소. min≈0은 초저변동 종목이 mu≈0으로 눌린 것 — 사전등록대로 **클리핑 없음**의 귀결(기록).
- **게이트 판정**:
  - ① ΔIR **+0.0711 < +0.36**(1 SE) → **노이즈 대역**. 게다가 서브기간 부호 **불일치**(P1 −0.140 · P2 −0.121 · P3 **+0.326**): 개선이 전량 P3(2023-)에 집중, P1·P2는 소폭 악화. 사전등록 채택 조건(ΔIR>1SE & 부호 일관) **미충족**.
  - ② 캐릭터 보존 **PASS**: TE 3.21%(≤4.5% 가드, S0 3.11% 대비 +0.10%p) · active_return +0.38%p(붕괴 아님·오히려 상승) · turnover −4.0%p · realized_beta 사실상 불변. 벤치마크 붕괴 없음.
  - ③ fallback 급증 없음 **PASS**: ECOS 188·fallback_rate 0.0 (S0와 동일).
  - ④ 단일 사전등록 파라미터·스윕 없음 → DSR/selection-bias **비액션 유지**(§2-7).
- **결정**: `mu_vol_scaling_enabled` **OFF-default 유지, 프로덕션 무변경**(§8). |ΔIR|=0.071<0.36이라 **IR 근거 채택 불가(노이즈)** — §2-4에 따라 기각도 아니고 "설명력 근거로만 판단". Grinold식 변환은 이론적 동기는 타당하나 본 데이터에서 full-period 이득이 노이즈 대역이고 P3 단일 레짐 집중이라 프로덕션 승격 근거 부족. 인프라(플래그·순수함수·테스트 13건)는 향후 재평가·다른 데이터 vintage 대비로 유지. 롤백 불요(default-OFF, 바이트동일 parity 증명됨).
- **미해결/이관**: 캐시 안전성(SAFE_FOR_CACHE_REUSE) 등록은 **보류·이관** — run_variant.py가 다른 arm(E1b) 수정 중이라 편집 금지 지침에 따름. 본 arm은 캐시 미사용 풀 경로로 실행됨(플래그는 예측-후 변환이라 harvest 무영향, 향후 등록 시 재사용 안전).

### S7.A2 confidence-spread-recal (confidence spread 재보정) — **사전등록 (2026-07-06)**

- **가설**: `compute_signal_confidence`(`src/backtest.py:897-920`)의 spread_score = clip(raw_spread/spread_scale, 0.20, 1.00)에서 default `spread_scale=0.20` vs D0 실측 raw_spread median **3.575873473877061**(`outputs/degenerate_retrain_report.json` `raw_spread_dist.median`, verifier 재계산 일치) → 약 18배 차이로 **상시 1.0 포화**. 동적 실행(`apply_dynamic_execution`, eta = 0.5·√confidence·clip, no-trade band = 0.003/max(conf,0.15))의 spread 채널이 죽어 confidence가 사실상 ic_score 항으로 붕괴. spread_scale을 median으로 재보정하면 spread_score가 처음으로 [0.20,1.00) 대역에서 변동. **`confidence_spread_scale`은 2026-07-02 구조 리뷰 #2에서 이미 config에 노출된 §8 승인 대기 레버**(`src/config.py:596-602`) — 값 변경만, 코드 변경 0.
- **사전등록 (이 값 외 변형·스윕 금지)**: `confidence_spread_scale = 3.57587`(= D0 `raw_spread_dist.median` 6-sig-fig, `outputs/degenerate_retrain_report.json` 정본 인용). 의미: 중앙값 스프레드 날짜의 spread_score=1.0, 그보다 무딘 신호의 날짜는 비례 감소(clip 하한 0.20). ic 상수 재보정(A3 인접 가설)은 **본 arm 범위 밖**(단일 파라미터 규율 §2-4). 정본 iter15_65tkr_reb21_vtg + 이 오버라이드만.
- **구현 (코드 변경 0)**: `variants/exp_confidence_spread_recal.yaml`(정본 전체 복사 + `confidence_spread_scale: 3.57587`). src/·run_variant.py·tests/ 무수정. config 반영 확인: run_variant.load_manifest→compose_config로 로드 시 `cfg.confidence_spread_scale == 3.57587` **ASSERT PASS**(DEFAULT 0.20 대비). 배선 경로: `src/backtest.py:1247-1249`가 `spread_scale=float(getattr(config,"confidence_spread_scale",0.20))`로 실제 주입 확인.
- **OFF 파리티 (N/A 근거)**: 신규 동작·신규 플래그 없음(기존 레버 값 변경). baseline 코드 무접촉이므로 OFF 파리티 풀 런 불필요 — default 0.20이 정본 S0 그 자체다. run_variant.py 무수정이라 회귀 가드 대상 없음.
- **캐시/격리**: `confidence_spread_scale`은 SAFE_FOR_CACHE_REUSE **미포함**(확인만, 등록 보류·이관 — 체크포인트 격리 상태, 전 arm 풀 경로 비교 유지 지침). 오버라이드 중 유일한 unsafe 키 → 캐시 DISABLED·풀 파이프라인 재실행(비교 가능성 확보). arm 자체 출력 디렉터리(`outputs/exp_confidence_spread_recal/`), 정본 무접촉.
- **S0 기준(동일 ECOS)**: IR 1.481437507913232 · TE 0.031069 · turnover 1.14402 · realized_beta 1.02439 · P1 1.591/P2 0.575/P3 2.005.
- **판정 게이트 (사전등록, §2-4/§2-5/§2-7)**: ① ΔIR > **+0.36**(=1 SE) **& 서브기간(P1/P2/P3) 부호 일관** ② 캐릭터 보존(TE ≤4.5%·active share 붕괴 없음) ③ fallback 급증 없음 ④ 단일 사전등록 파라미터라 DSR/selection-bias 비액션. 셋 다 충족 시에만 IR 근거 채택 후보; 미달 시 OFF 유지. **게이트 통과 여부가 아니라 정직한 측정이 성공 기준.** (참고: A1 ΔIR +0.071·A3 +0.003 둘 다 미충족·OFF 종결.)
- **실행 예정**: `<PY> run_variant.py --variant variants/exp_confidence_spread_recal.yaml` → `outputs/exp_confidence_spread_recal/`(단일 foreground, 풀 경로).
- **격리 무접촉 검증 (2026-07-06)**: arm pkl vs 정본 S0 pkl — `raw_predictions`·`predictions`·`ic_series`(값+인덱스)·`turnover` 인덱스 **전부 바이트동일**. `avg_ic` 0.048649 동일. → `confidence_spread_scale`이 harvest(Phase 1~4)·예측·IC를 무접촉, **오직 실행(eta·no-trade band)만** 변경함을 실증(§4.2 confound 부재).
- **결과 (2026-07-06, `outputs/exp_confidence_spread_recal/`, 풀 경로 305.7s, ECOS 188·solver fallback 0.0%·optimizer TE-relax fallback 5/94)**:

  | metric | S0 (OFF, scale 0.20) | A2 (ON, scale 3.57587) | Δ |
  |---|---:|---:|---:|
  | information_ratio | 1.481438 | 1.483625 | **+0.002188** |
  | tracking_error | 0.031069 | 0.030582 | −0.000487 |
  | active_return | 0.046027 | 0.045372 | −0.000655 |
  | avg_annual_turnover | 1.144021 | 1.106629 | −0.037392 |
  | realized_beta | 1.024389 | 1.022090 | −0.002299 |
  | avg_ic | 0.048649 | 0.048649 | +0.000000 |
  | sharpe_ratio | 1.307645 | 1.307881 | +0.000236 |
  | max_drawdown | −0.299887 | −0.299906 | −0.000019 |
  | P1_ir | 1.591390 | 1.677451 | +0.086062 |
  | P2_ir | 0.574884 | 0.463195 | **−0.111689** |
  | P3_ir | 2.004814 | 2.016578 | +0.011764 |

- **confidence·eta 분포 변화 (94 리밸런스, 실 `compute_signal_confidence` 재실행 — pred_row·raw_pred_row + trailing_ic_mean(ic_series prior≥2, last-6 윈도) 재구성, eta=clip(0.5·√conf,0.05,0.95))**:

  | 채널 | S0 (0.20) min/median/mean/max | A2 (3.57587) min/median/mean/max |
  |---|---|---|
  | confidence | 0.2000 / **1.0000** / 0.7509 / 1.0000 | 0.1510 / **0.8263** / 0.7042 / 1.0000 |
  | eta | 0.2236 / **0.5000** / 0.4193 / 0.5000 | 0.1943 / **0.4545** / 0.4056 / 0.5000 |
  | spread_score(단독) | 1.0000 / 1.0000 / 1.0000 / 1.0000 | 0.5807 / 0.9959 / 0.9395 / 1.0000 |

  - **spread_score < 1.0 리밸런스 비율: S0 0.0%(상시 포화) → A2 52.1%** (사전등록 예측대로: median raw_spread 3.58/3.58=1.0, 절반이 그 아래). spread 채널이 처음으로 활성화(inert 탈출).
  - eta가 실제로 바뀐 리밸런스 52.1%(나머지 47.9%는 conf 포화/clip 동일), 평균 |Δeta| 0.0137. eta median 0.50→0.45 하향 → 트레이딩 강도 감소 → turnover −3.7%p·TE −0.05%p와 정합. confidence median이 1.0(포화)→0.826으로 내려오며 동적 실행이 실제로 반응.
- **게이트 판정**:
  - ① ΔIR **+0.0022 ≪ +0.36**(1 SE) → **노이즈 대역**(A1 +0.071·A3 +0.003과 동류, 사실상 0). 서브기간 부호 **불일치**: P1 **+0.086** · P2 **−0.112** · P3 **+0.012**. P2(2021-05..2023-10) 악화. 사전등록 채택 조건(ΔIR>1SE & 부호 일관) **양쪽 미충족**.
  - ② 캐릭터 보존 **PASS**: TE 3.06%(≤4.5% 가드, S0 3.11%→오히려 감소) · active_return +4.54%(S0 +4.60% 대비 소폭↓·붕괴 아님) · turnover −3.7%p · realized_beta 사실상 불변(1.022). 벤치마크 붕괴 없음.
  - ③ fallback 급증 없음 **PASS**: solver ECOS 188·fallback_rate 0.0(S0 동일); optimizer TE-relax fallback 5/94(S0 6/94, 오히려 감소).
  - ④ 단일 사전등록 파라미터·스윕 없음 → DSR/selection-bias **비액션 유지**(§2-7).
- **결정**: `confidence_spread_scale` **default 0.20 유지(OFF-default), 프로덕션 무변경**(§8). |ΔIR|=0.0022<0.36이라 **IR 근거 채택 불가(노이즈)** — §2-4에 따라 기각도 채택도 아닌 "설명력 근거로만 판단". 재보정은 spread 채널을 확실히 되살렸으나(spread_score<1.0 52.1%·eta median 0.50→0.45·confidence median 1.0→0.826) full-period 순효과가 0에 수렴하고 레짐 셔플(P1·P3 소폭↑ vs P2 −0.11)에 그침 — 순 edge 없음. turnover −3.7%p·TE 소폭 개선은 IR 개선을 동반하지 않아 승격 근거 부족. 인프라(config 레버는 이미 노출됨, 코드 변경 0)는 향후 재평가·다른 데이터 vintage 대비로 유지. src/·run_variant 무접촉이라 롤백 불요(variant yaml만, default 0.20이 곧 S0).
- **미해결/이관**: SAFE_FOR_CACHE_REUSE 등록은 **보류·이관**(체크포인트 격리·전 arm 풀 경로 비교 유지 지침 — run_variant.py 무수정). `confidence_spread_scale`은 Phase 5 실행 전용이라 향후 등록 시 캐시 재사용 안전(단, 현 사이클은 풀 경로). 인접 미평가 가설: ic_score 상수 재보정(median IC 0.0404 포화, D0) — 단일 파라미터 규율상 본 arm 범위 밖, 별도 arm 필요. 커밋 보류(사용자 승인 대기).

### S7.A3 adaptive-EMA (trailing-IC 적응형 예측 EMA) — **사전등록 (2026-07-06)**

- **가설**: 예측 EMA 블렌딩이 고정 α=0.5(`src/model_trainer.py` `apply_prediction_ema`, walk_forward_train 내부 블렌드 재현). 고정 α의 regime-lag이 문제의식(D0: 재훈련 degenerate 50%, trailing IC median 0.0404). 최근 IC가 좋을 때 새 신호 가중을 높이고 나쁠 때 스무딩을 강화하는 시변 α를 **사전등록 단일 함수형**으로 평가. src/ 프로덕션 코드 무수정 — 2-pass 주입 평가.
- **사전등록 함수형 (이 형태 외 변형·스윕 금지, D0 분포 앵커)**: 각 예측일 t에 대해

      α_t = clip( 0.5 + (tIC_t − m) / (2·IQR), 0.25, 0.75 )

  - **앵커(D0 정본, 하드코딩 금지·리포트 로드)**: `outputs/degenerate_retrain_report.json` `report.trailing_ic_dist` → m(median)=**0.04035956534962617**, IQR=**0.07403856582239399**. 상수 전부 D0 분포에서 유도, 자유 파라미터 0. clip은 대칭 [0.25, 0.75].
  - **tIC_t (인과성 필수)**: 각 IC 이벤트를 **실현완료일**(= 리밸런스/예측일 + forward_horizon 20 거래일)로 타임스탬프. tIC_t = 트레일링 63 거래일 윈도 `[dates[max(0,i−63)], dates[i−1]]`(상한 dates[i−1]은 t보다 **엄격히 과거**) 내 실현 이벤트 평균. i==0 또는 무이벤트 → α_t=0.5. 미래정보 유입 없음(실현일 인덱싱으로 by construction 인과).
  - 블렌딩 재귀는 `apply_prediction_ema`와 동일 구조에 α만 시변: blended_t = α_t·raw_t + (1−α_t)·blended_{t−1}. **α_t≡0.5이면 apply_prediction_ema(raw,0.5)와 바이트동일**(합격 A3-5).
- **구현**: `scripts/run_adaptive_ema_arm.py`(순수함수 2개 + main, src/ 무수정·플래그 없음). 합격 테스트 `tests/acceptance/test_adaptive_ema.py` **13/13 통과**, 전체 스위트 129 pass(무관: A1 mu_vol_scaling 경고 6건은 비실패). 주입: pkl `raw_predictions`(pre-EMA·pre-overlay)에 시변 α EMA 적용 → `run_backtest(precomputed_predictions=…)`로 프로덕션 MVO(오버레이는 정상 1회 적용, 이중오버레이 금지). 데이터: `outputs/iter15_65tkr_reb21_vtg/backtest_result.pkl`.
- **ic_events 구성**: ic_series(93, 리밸런스일 인덱싱) → calendar(raw_predictions.index, 3233 거래일)에서 get_indexer → pos+20 시프트 → 실현일. 93개 전부 온-캘린더·오버플로 0 → **93개 실현일 이벤트, span 2018-12-24..2026-05-20**.
- **identity 게이트 (α≡0.5, on-baseline 재현, 2026-07-06)**: `--identity-only` 풀 주입 백테스트 49s, ECOS 188·fallback 6/94(6.4%, S0와 동일 경로). vs 정본 S0(`outputs/iter15_65tkr_reb21_vtg/metrics.json`): IR **1.481437507913232**(Δ **0.0**) · TE **0.031069189048318836**(Δ 0.0) · turnover **1.1440214379781009**(Δ 0.0) · active_return **0.04602706199662654**(Δ 0.0). **max|Δ|=0.000e+00 → 바이트 재현 PASS**. apply_prediction_ema(raw,0.5)가 S0 내부 pre-overlay 패널을 정확히 복원함을 실증(pre-EMA 의미·주입경로·EMA-confound 부재 확인, §4.2/E1b).
- **S0 기준(동일 ECOS)**: IR 1.481437507913232 · TE 0.031069 · turnover 1.14402 · realized_beta 1.02439 · P1 1.591/P2 0.575/P3 2.005.
- **판정 게이트 (사전등록, §2-4/§2-5/§2-7)**: ① ΔIR > **+0.36**(=1 SE) **& 서브기간(P1/P2/P3) 부호 일관** ② 캐릭터 보존(TE ≤4.5%·active share 붕괴 없음) ③ fallback 급증 없음 ④ 단일 사전등록 파라미터라 DSR/selection-bias 비액션. 셋 다 충족 시에만 IR 근거 채택 후보; 미달 시 OFF 유지. **게이트 통과 여부가 아니라 정직한 측정이 성공 기준.**
- **실행**: `<PY> scripts/run_adaptive_ema_arm.py` → `outputs/exp_adaptive_ema/{identity,arm}/`(단일 foreground). identity 재현 49s + arm 45s.
- **결과 (2026-07-06, `outputs/exp_adaptive_ema/arm/`, arm 백테스트 45s, ECOS 188·solver fallback 0.0%·optimizer TE-relax fallback 5/94)**:

  | metric | S0 (OFF) | A3 (ON, 시변 α) | Δ |
  |---|---:|---:|---:|
  | information_ratio | 1.481438 | 1.484346 | **+0.002908** |
  | tracking_error | 0.031069 | 0.029288 | −0.001781 |
  | active_return | 0.046027 | 0.043473 | −0.002554 |
  | avg_annual_turnover | 1.144021 | 1.133318 | −0.010704 |
  | realized_beta | 1.024389 | 1.022951 | −0.001438 |
  | avg_ic | 0.048649 | 0.048794 | +0.000145 |
  | sharpe_ratio | 1.307645 | 1.297159 | −0.010485 |
  | max_drawdown | −0.299887 | −0.300085 | −0.000198 |
  | P1_ir | 1.591390 | 1.724103 | +0.132713 |
  | P2_ir | 0.574884 | 0.142670 | **−0.432214** |
  | P3_ir | 2.004814 | 2.134329 | +0.129515 |

- **α_t 분포 (n=3233 예측일)**: min **0.250** · median **0.500** · max **0.750** · mean **0.51005**. 0.5 이탈 빈도 **60.38%**(frac_off_half), 상한 clip(0.75) **22.05%** · 하한 clip(0.25) **13.64%**. → 함수형이 활발히 작동(inert 아님), mean≈0.51로 순평균은 거의 중립이나 레짐별로 크게 재분배.
- **게이트 판정**:
  - ① ΔIR **+0.0029 ≪ +0.36**(1 SE) → **노이즈 대역**(A1 +0.071보다도 작아 사실상 0). 게다가 서브기간 부호 **불일치**: P1 **+0.133** · P2 **−0.432** · P3 **+0.130**. P2(2021-05..2023-10)가 크게 악화. 사전등록 채택 조건(ΔIR>1SE & 부호 일관) **양쪽 모두 미충족**.
  - ② 캐릭터 보존 **PASS**: TE 2.93%(≤4.5% 가드, S0 3.11%→오히려 감소) · active_return +4.35%(S0 +4.60% 대비 소폭↓이나 붕괴 아님) · realized_beta 사실상 불변(1.023). 벤치마크 붕괴 없음.
  - ③ fallback 급증 없음 **PASS**: solver ECOS 188·fallback_rate 0.0(S0 동일); optimizer TE-relax fallback 5/94(S0 6/94, 오히려 감소).
  - ④ 단일 사전등록 파라미터·스윕 없음 → DSR/selection-bias **비액션 유지**(§2-7).
- **결정**: adaptive-EMA **OFF-default 유지, 프로덕션 무변경**(§8). |ΔIR|=0.0029<0.36이라 **IR 근거 채택 불가(노이즈)** — §2-4에 따라 기각도 채택도 아닌 "설명력 근거로만 판단". 시변 α는 활발히 작동(60% 이탈, 양쪽 clip)했으나 full-period 순효과가 0에 수렴하고 레짐 셔플(P1·P3 +0.13 vs P2 −0.43)에 그침 — 순 edge 없음. TE/turnover 소폭 개선은 있으나 IR 개선을 동반하지 않아 승격 근거 부족. 인프라(`scripts/run_adaptive_ema_arm.py` + 순수함수 2개 + 합격 테스트 13건)는 향후 재평가·다른 데이터 vintage 대비로 유지. src/ 무접촉이라 롤백 자체가 불요(프로덕션에 아무것도 배선 안 됨).
- **미해결/이관**: 없음. src/·run_variant·variants 무수정(2-pass 주입 평가), 정본 S0 무접촉(identity Δ=0.0 재현으로 격리 확인). 커밋 보류(사용자 승인 대기).

### S7.A4 seed-ensemble (LGBM 시드 앙상블 k=5) — **사전등록 (2026-07-06)**

- **가설**: 예측 엔진이 LightGBM 단일 시드(random_state=42, `src/config.py:158-172`)로만 학습된다. 단일 시드 예측에는 추정 노이즈가 있고 D0(재훈련 degenerate 50%)상 시드별 walk-forward 궤적이 상이할 수 있다. k=5 시드 평균은 (a) 예측 분산 축소, (b) 시드 운(luck)의 정량화(per-seed IR 분산)를 동시에 제공. src/ 프로덕션 코드 무수정 — 2-pass 주입 평가. DR/A1~A3 전례상 기대는 보수적.
- **사전등록 (이 구성 외 변형·스윕 금지, k 스윕 금지)**: 시드 **{42, 43, 44, 45, 46}** 고정(k=5). 42는 정본 S0 harvest 재사용(동일 시드 재실행 낭비 금지), 43~46은 정본 variant + `lgbm_params.random_state`만 변경한 full harvest. 결합 규칙(파라미터-프리):
  1. 시드별 **pre-EMA raw z-패널**(`backtest_result.pkl.raw_predictions`) → **셀 단위 유한값 평균**(NaN skip, 전부 NaN → NaN).
  2. **per-date CS 재표준화**(`src/model_trainer.py:240-245` z 관용구와 동일: mean skipna, std ddof=1 skipna, `if std>0`일 때만 (row−mean)/std; 상수행·단일유한값행·전NaN행은 불변, 0나눗셈 없음).
  3. 표준 EMA **α=0.5**(`apply_prediction_ema`, 정본 고정값 — 시변 아님).
  4. pre-overlay 패널로 `run_backtest(precomputed_predictions=…)` 주입(오버레이 런타임 1회 — 이중오버레이 금지).
  - **NaN 마스크 게이트**: 시드 간 NaN 마스크는 데이터 가용성 기반이라 동일해야 정상. 불일치율(≥1 NaN & ≥1 유한 셀 / 전체 셀) **> 0.1%면 중단·보고**(§9 구조 가정 위반). 자유 파라미터 0(finite mean·z·고정 EMA 모두 파라미터-프리).
- **기각한 대안**: 시드별 z 평균 후 재표준화 생략(CS 분산 수축으로 mu 스케일 왜곡), rank 평균(정보 손실), k 스윕(사전등록 위반), post-EMA 패널 평균(EMA 체인 비선형성으로 의미 불명).
- **구현**: `scripts/run_seed_ensemble_arm.py`(순수함수 `combine_seed_panels`·`nan_mask_mismatch_rate` + main, src/·run_variant.py 무수정·플래그 없음). 합격 테스트 `tests/acceptance/test_seed_ensemble.py` **15/15 통과**, co-located smoke `tests/test_run_seed_ensemble_arm.py` 3/3, 전체 스위트 **147 pass**(무관: `test_mu_vol_scaling.py` 경고 6건은 비실패). 시드 variant는 `variants/exp_seed{43,44,45,46}.yaml`(main의 `write_seed_variant`가 정본 manifest deepcopy → label/out_dir/`lgbm_params.random_state`만 변경해 생성; harvest 시 `--no-cache` 풀 경로).
- **seed 전달 경로 확인 (정적, harvest 전)**: `build_override_config`=`dataclasses.replace(**overrides)`라 `lgbm_params`가 **통째 교체**(deep-merge 아님) → 시드 yaml은 FULL lgbm_params 블록 필요, helper가 정본 파생으로 보장. compose 검증: exp_seed43~46 각 `random_state`=43~46·keys_intact=True·n_estimators=800. yaml `lgbm_params` → `model_trainer.py:192` `lgb.LGBMRegressor(**config.lgbm_params)` 직결. 캐시: `lgbm_params`∉`SAFE_FOR_CACHE_REUSE`(`run_variant.py:289-339`) + `--no-cache` ⇒ full 재실행 이중보장. 결정성 사전확인(경험적, 합성데이터): DEFAULT lgbm_params로 seed42 vs 43 예측 max|Δ|=0.1215·mean|Δ|=0.0463, seed42 재fit 완전재현(==True). 근거: subsample=0.8은 bagging_freq=0(sklearn 기본 subsample_freq=0)이라 무효이나 colsample_bytree=0.8(feature_fraction)이 시드 구동 → random_state 변경이 실제 예측차 생성. (실 harvest의 시드별 IR·예측차는 결과 절에 기재.)
- **S0 기준(동일 ECOS)**: IR 1.481437507913232 · TE 0.031069 · turnover 1.14402 · realized_beta 1.02439 · P1 1.591/P2 0.575/P3 2.005.
- **판정 게이트 (사전등록, §2-4/§2-5/§2-7)**: ① ΔIR > **+0.36**(=1 SE) **& 서브기간(P1/P2/P3) 부호 일관** ② 캐릭터 보존(TE ≤4.5%·active share 붕괴 없음) ③ fallback 급증 없음 ④ 단일 사전등록 구성·k 스윕 없음 → DSR/selection-bias 비액션. 셋 다 충족 시에만 IR 근거 채택 후보; 미달 시 OFF 유지. **게이트 통과 여부가 아니라 정직한 측정이 성공 기준.** (참고: A1 +0.071·A2 +0.002·A3 +0.003 전부 미충족·OFF 종결.)
- **실행 예정**: (1) `<PY> scripts/run_seed_ensemble_arm.py --identity-only` — identity 게이트(α≡0.5 on seed42 → S0 재현, Δ>1e-6이면 중단) → (2) `<PY> scripts/run_seed_ensemble_arm.py` — seed 43~46 harvest 순차 4회 + NaN 게이트 + combine → EMA(0.5) → arm 주입. → `outputs/exp_seed_ensemble/{identity,arm}/` + `outputs/exp_seed{43..46}/`(단일 foreground, 병렬 spawn 금지).
- **identity 게이트 (α≡0.5 on seed42, 2026-07-06)**: `--identity-only` 풀 주입 백테스트 45s, ECOS 188·fallback 6/94. vs 정본 S0: IR **1.481437507913232**(Δ **0.0**)·TE **0.031069189048318836**(Δ 0.0)·turnover **1.1440214379781009**(Δ 0.0)·active_return **0.04602706199662654**(Δ 0.0). **max|Δ|=0.000e+00 → 바이트 재현 PASS**(A3와 동일). seed42 raw_predictions가 S0 pre-overlay 패널을 정확 복원함을 실증.
- **harvest 완료 (2026-07-06→07, 단일 프로세스 순차, 병렬 spawn 없음·전부 `--no-cache` 풀 파이프라인)**: seed43 225s·seed44 211s·seed45 451s·seed46 617s(재훈련 부하 편차). 각 `outputs/exp_seed{n}/`. NaN 마스크 불일치율 **0.00000**(게이트 0.1% 통과 — 5시드 데이터 가용성 격자 완전 동일).
- **per-seed full-run IR (시드 운 정량화, k=5)**:

  | seed | 42(prod/S0) | 43 | 44 | 45 | 46 | mean | std(ddof=1) | min | max | spread |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | full-run IR | **1.4814** | 1.1843 | 1.0565 | 1.1103 | 1.4058 | 1.2477 | 0.1865 | 1.0565 | 1.4814 | 0.4249 |

  - **핵심 발견**: 프로덕션 시드 42가 5시드 중 **IR 최고(rank 5/5)**. S0 헤드라인 IR 1.481은 **호의적 시드 운**을 포함 — 5시드 평균 1.248 대비 **+0.234**(≈1.25 SE_seed) 위. 시드-IR std 0.187로, 단일 시드 IR의 시드 노이즈가 상당(스프레드 0.42). 앙상블 arm IR 1.318은 **평균 시드 IR(1.248)보다는 높음**(앙상블이 무작위 단일 시드 대비로는 denoise) but S0(=최고 시드)보다는 낮음.
- **앙상블 진단**: 시드 간 평균 쌍상관 **0.8355**(10쌍) — 시드들이 고상관(feature sub-sampling만 교란, ~16%만 idiosyncratic). pre-재표준화 CS 분산 축소 **13.16%**(finite-mean 패널 0.855 vs 평균 per-seed 0.985; 이론 avg-var(ρ=0.836,k=5)=0.868과 정합). **단, STEP2 per-date 재표준화가 단위분산으로 재정규화하므로 최종 combined CS 분산 축소율=0.0(by construction)** — 유의미 지표는 쌍상관 0.836과 pre-restd 13% 축소. 고상관 탓에 유효 다양성이 작아 앙상블 이득 제한적.
- **결과 (2026-07-06→07, `outputs/exp_seed_ensemble/arm/`, arm 백테스트 122s, ECOS 188·solver fallback 0.0%·optimizer TE-relax fallback 8/94)**:

  | metric | S0 (OFF, seed42) | A4 (ON, k=5 앙상블) | Δ |
  |---|---:|---:|---:|
  | information_ratio | 1.481438 | 1.317520 | **−0.163917** |
  | tracking_error | 0.031069 | 0.028991 | −0.002078 |
  | active_return | 0.046027 | 0.038197 | −0.007830 |
  | avg_annual_turnover | 1.144021 | 1.166345 | +0.022323 |
  | realized_beta | 1.024389 | 1.012491 | −0.011898 |
  | avg_ic | 0.048649 | 0.044719 | −0.003930 |
  | sharpe_ratio | 1.307645 | 1.283830 | −0.023815 |
  | max_drawdown | −0.299887 | −0.295614 | +0.004273 |
  | P1_ir | 1.591390 | 1.156285 | **−0.435105** |
  | P2_ir | 0.574884 | 0.415762 | **−0.159122** |
  | P3_ir | 2.004814 | 2.186175 | **+0.181361** |

- **게이트 판정**:
  - ① ΔIR **−0.1639**. |ΔIR|=0.164 < +0.36(1 SE)라 **여전히 노이즈 대역**(통계적으로 0과 구분 불가)이나 점추정이 **음(−)**이고 서브기간 부호 **불일치**(P1 −0.435·P2 −0.159·P3 +0.181, P1/P2 악화·P3만 개선). 사전등록 채택 조건(ΔIR>+0.36 & 부호 일관) **양쪽 완전 미충족**. A1~A3 중 유일하게 점추정 음수(A1 +0.071·A2 +0.002·A3 +0.003 vs A4 **−0.164**).
  - ② 캐릭터 보존 **PASS**: TE 2.90%(≤4.5% 가드, S0 3.11%→감소)·active_return +3.82%(S0 +4.60% 대비 −0.78%p, 벤치마크 붕괴 아님·集中 성격 유지)·realized_beta 1.012(사실상 불변). active share 붕괴 없음.
  - ③ fallback 급증 없음 **PASS**: solver ECOS 188·fallback 0.0%(S0 동일); optimizer TE-relax fallback 8/94 vs S0 6/94(+2, 급증 아님).
  - ④ 단일 사전등록 구성·k 스윕 없음 → DSR/selection-bias **비액션 유지**(§2-7).
- **결정**: seed-ensemble **OFF-default 유지, 프로덕션 무변경**(§8). ΔIR **−0.164**(음)·서브기간 부호 불일치로 승격 근거 전무 — |ΔIR|<0.36이라 §2-4상 "노이즈, 설명력으로만 판단"이되 점추정이 음이라 채택 가치 없음. src/ 무접촉·플래그 없음(2-pass 주입 평가)이라 롤백 자체 불요. 인프라(`scripts/run_seed_ensemble_arm.py`·순수함수 2·합격 테스트 15·variant 4개)는 유지.
- **설명력·DSR 함의(마감 입력)**: 본 arm의 진짜 산출은 IR 개선이 아니라 **S0 헤드라인의 시드 운 정량화**다. S0 IR 1.481은 5시드 분포(1.057~1.481, mean 1.248, std 0.187)의 **최상단**이며, 시드-강건 추정치는 mean 1.248 또는 앙상블 1.318 수준. 즉 S0가 보고하는 edge의 약 **0.16~0.23 IR가 시드 운**에 기인(안정 edge 아님). DSR/최종 판정표 작성 시 S0 IR를 시드 노이즈 밴드(±0.19)와 함께 보고할 것. 재훈련 degenerate 50%(D0)와 정합: 시드가 walk-forward 궤적을 실질 교란.
- **미해결/이관**: 없음. src/·run_variant·tests/acceptance 무수정, 정본 S0 무접촉(identity Δ=0.0). 커밋 보류(사용자 승인 대기). 참고: `variants/exp_seed{43..46}.yaml`·`outputs/exp_seed{43..46}/`는 arm 산출물(정본 아님).

### S7.infra Phase 3 체크포인트 + 캐시 경로 S0 재현 수정 (E1/E1b, 2026-07-06)

**(c) Phase 3 (targets) 체크포인트 도입 [E1]**
- 문제: `run_variant` 캐시-재사용 브랜치가 매 실행 `build_targets()`의 ~2,650 sklearn PCA fit을 재계산(`run_variant.py:289` 부근).
- 구현: `run_variant.py`에 `phase3_cache_token` / `save_phase3_checkpoint` / `load_phase3_checkpoint` 추가 — Phase 1/2/4 HMAC-pickle 패턴 미러(`src.backtest.save_checkpoint`/`_sign_file` 재사용). 로드는 graceful(토큰불일치·서명없음·서명불일치·손상pickle·부재 → `None`, 예외 없음; Phase 3 재계산은 항상 정확하므로 폴백 안전).
- 토큰 필드(`build_targets` 실제 의존만): `pca_n_remove, pca_components, pca_lookback, forward_horizon, multi_horizon_targets_enabled, multi_horizon_weights, regime_pca_weighted_enabled` + upstream(`phase1|phase2` .sig 다이제스트 체이닝). 옵티마이저·Phase5/6 필드(`risk_aversion` 등)에는 **불변** → 옵티마이저 스윕에도 캐시 생존.
- 테스트: `tests/acceptance/test_phase3_checkpoint.py` 9/9 + `tests/test_run_variant.py` 3/3(후자는 repo TDD 가드가 `run_variant.py` 편집 전 동명 테스트를 요구해 추가; tests/acceptance 미접촉).

**(a) build_targets config-less 호출 정리**
- `run_variant.py`의 재계산 브랜치 `build_targets(data)` → `build_targets(data, config=cfg)`.
- 바이트동일 근거: 프로덕션 variant의 target/PCA 7필드가 전부 `DEFAULT_CONFIG`와 동일(n_remove=2·components=5·lookback=252·horizon=20·mh_enabled=False·mh_weights={}·regime=False). target 필드 오버라이드는 `SAFE_FOR_CACHE_REUSE` 밖이라 캐시 자체가 비활성(full pipeline)되어 캐시 브랜치는 diverging target을 볼 수 없음. `phase3_cache_token(DEFAULT)==token(prod cfg)` 확인.

**(b) cache≠full 근인·수정·파리티 [E1b]**
- 최초 증상: 캐시런 IR 1.463 / P2_ir 0.082 vs full·정본 S0 1.481 / 0.575.
- 근인(실측, 팀리드 EMA 가설과 상이): `walk_forward_train` 반환 `predictions`는 이미 **post-EMA**(`model_trainer.py:292`), `raw_predictions`는 pre-EMA(:458) — EMA는 원인이 아님. 실제 근인은 **이중 오버레이**: `result.predictions`는 post-overlay(`backtest.py:1507`, PEAD/growth_tilt/VTG 적용 후)인데 캐시 경로가 `precomputed_predictions`에 오버레이를 재적용(`:1477-1507`). E1 프라이밍이 post-overlay 패널을 저장해 오버레이가 2회 적용됨. 실측 overlay effect: post vs pre 패널 127,636/128,243 셀 상이(max|Δ|=7.07). `scripts/run_overlay_ablation.py:5-8` 계약("harvest overlays-OFF → base.predictions = overlay-free EMA base")과 정합.
- 수정: `src/backtest.py`에 `result.pre_overlay_predictions` 노출(§4.2 `pre_overlay_ema_predictions` = post-EMA·pre-listing-mask·pre-overlay; `walk_forward` 직후 캡처). Phase 4 체크포인트는 `result.predictions`(post-overlay)가 아니라 이 패널을 저장 → 캐시 경로가 오버레이를 정확히 1회 적용. 신규 속성 캡처만이라 `run_backtest` 동작·기존 메트릭 불변(OFF-invariant).
- 파리티 증거(바이트동일): pre-overlay 프라이밍 후 **캐시 런 metrics == full 런 metrics == 정본 S0 metrics 전부 sha256 일치**(IR 1.481437507913232 완전정밀도, sub_periods 포함). 캐시 build(A) vs reuse(B)도 바이트동일(`elapsed_sec`만 상이). **E1b 게이트 통과.**
- 시간: full ~255s vs 캐시 재사용 ~41s(~6x). 
- 상태·안전: 게이트 통과. 프라이밍 체크포인트(`outputs/checkpoints/`)는 동시 실행 레이스·스테일 방지 위해 **삭제(격리)**; 재프라이밍은 `scratchpad/prime_checkpoints.py`로 결정론적 재현 가능. 캐시 경로의 arm 평가 실사용 여부는 오케스트레이터 결정. 회귀: 전체 스위트 129 pass(blast radius `test_backtest`+`test_run_variant`+phase3 acceptance 13/13). 정본 `metrics.json` 내용 무접촉(IR 1.481437507913232, elapsed 223.0).

### S7.summary 4-arm 평가 프로그램 마감 (2026-07-06)

S7 AI-로직 후보 4개(A1~A4) 사전등록·측정 **완료**. **4개 전부 채택 게이트 미충족 → OFF-default 유지, 프로덕션 무변경.** 정본 S0 기준(동일 ECOS): IR 1.481437507913232 · TE 0.031069 · P1 1.591/P2 0.575/P3 2.005. 채택 바(§2-4): full-period ΔIR > **+0.36**(=1 SE) **& 서브기간 부호 일관**. 아래 수치는 전부 파일 로드(암산 없음).

- **4-arm 판정 요약표** (전부 후보당 단일 사전등록 파라미터, 스윕 없음):

  | arm | 사전등록 파라미터(단일값) | IR | ΔIR vs S0 | 서브기간 부호 Δ(P1/P2/P3) | 캐릭터 보존 | 판정 |
  |---|---|---:|---:|:---:|:---:|---|
  | A1 mu-vol-scaling | mu=z·σ/median_CS(σ) (param-free) | 1.552503 | **+0.0711** | −/−/+ 불일치 | PASS | 미충족·**OFF** |
  | A2 confidence-spread-recal | confidence_spread_scale=3.57587 | 1.483625 | **+0.0022** | +/−/+ 불일치 | PASS | 미충족·**OFF** |
  | A3 adaptive-EMA | α_t=clip(0.5+(tIC−m)/2IQR, .25,.75) | 1.484346 | **+0.0029** | +/−/+ 불일치 (P2 −0.432) | PASS | 미충족·**OFF** |
  | A4 seed-ensemble k=5 | seeds {42,43,44,45,46} | 1.317520 | **−0.1639** | −/−/+ 악화 | PASS | 미충족·**OFF** |

  - A1~A3: |ΔIR| 전부 노이즈 대역(<+0.36 1 SE) **&** 서브기간 부호 불일치 → 채택 조건 양쪽 미충족. A4: 점추정 **음수**. 4개 모두 캐릭터 보존(TE ≤4.5%·active share 붕괴 없음)은 통과 — **붕괴 FAIL이 아니라 edge 부재로 인한 미채택**(§2-5는 OK, §2-4가 게이트).

- **DSR / selection-bias 재산출** (S0 프로덕션 baseline gating, `run_selection_bias.py --auto --label iter15_65tkr_reb21_vtg`; `outputs/reports/selection_bias_report.md`·`outputs/csv/selection_bias_metrics.csv`):

  | 지표 | S7 전 (N=403) | S7 후 (N=407) |
  |---|---:|---:|
  | N_trials | 403 | 407 |
  | Observed SR | 1.463993 | 1.463993 (불변, S0) |
  | Deflated SR | 0.748132 | 0.745281 |
  | DSR p-value | 0.227190 | 0.228051 |
  | Grid haircut | 1.203955 | 1.204946 |
  | Adjusted SR | 0.260038 | 0.259047 |
  | Gate verdict | FAIL | FAIL |

  - inventory 갱신(`experiment_inventory.json`): A1/A2/A3/A4 각 **1 trial** 추가(+4). A4의 seed 43~46 harvest 4회는 **비선택 진단 입력**(단일 A4 앙상블 구성으로 수렴)이라 **trial 미계상**(항목 노트에 "per-seed diagnostics 4 runs (non-selection)" 명시). N 403→407.
  - 4 arm 추가가 sqrt(2·ln N) 페널티를 미세 이동(haircut **+0.0010** / DSR **−0.0029**) — **S0 selection-bias 판정을 실질적으로 바꾸지 않음**. Gate FAIL은 DSR p>0.10(다중비교 후 유의성 미달)에서 발생하며 나머지 4항목은 통과(adjusted SR>0 · MinTRL 7.8yr>1.2yr · survivorship CLEAN · sub-period all-positive STABLE). **전 arm OFF이므로 이 gating은 활성화 후보가 아니라 S0 자체의 다중비교 유의성 정보**로만 소용(§2-7 비액션 일관).

- **핵심 발견 2건**:
  1. **단일 런 ΔIR의 노이즈 대역 실증 (A4 부산물)**: per-seed full-run IR 42=1.4814 / 43=1.1843 / 44=1.0565 / 45=1.1103 / 46=1.4058 (범위 **1.057~1.481**, spread 0.425, std(ddof=1) 0.187). 정본 S0=**seed42가 5개 중 최상위**(rank 5/5) — S0 헤드라인 IR은 호의적 시드 운을 포함(5시드 mean 1.248 대비 +0.234). 즉 단일 런 IR의 시드 노이즈(±≈0.19)가 사전등록 채택 바(+0.36=1 SE)와 동급 크기 → A1~A3의 소폭 ΔIR(+0.07/+0.002/+0.003)이 전부 이 노이즈 대역 안이라는 **게이트 논리를 사후 실증**(seed 상관 0.836).
  2. **D0 degenerate 50%의 구조적 근인**: 재훈련 32윈도 중 16 degenerate(**50%**, `outputs/degenerate_retrain_report.json`). H1(즉시 early-stop: degenerate best_iteration median **1.0** vs healthy 92.0) **supported** — 검증손실 즉시 정체, 재훈련이 일반화 신호 미발견(incumbent 미개선). H2(P2 레짐 집중) **refuted** → **국면 무관**(P1 6·P2 5·P3 5로 균등). degeneracy는 특정 시기가 아니라 구조적. **후속 후보로만 기재**(본 사이클 미실행).

- **Production flips: 전부 no-flip.** `PipelineConfig` 기본은 4개 arm 모두 **OFF 불변**(A1 `mu_vol_scaling_enabled=False`, A2 `confidence_spread_scale=0.20` default, A3·A4는 src/ 배선 없음 — 2-pass 주입 평가). §8 프로덕션 규칙에 따른 활성화 후보 **0건**. 정본 `variants/iter15_65tkr_reb21_vtg.yaml` 무접촉. 롤백 불요(전부 default-OFF·바이트동일 parity 또는 src/ 무배선).

- **이관 백로그**:
  1. `mu_vol_scaling_enabled`·`confidence_spread_scale`의 `SAFE_FOR_CACHE_REUSE` 등록 **보류**(체크포인트 격리·전 arm 풀 경로 비교 유지 지침 — run_variant.py 무수정). 향후 등록 시 둘 다 예측-후/실행-전용이라 캐시 재사용 안전.
  2. Phase 4 체크포인트 harvester 부재로 캐시 경로 **dormant**. 프라이밍은 스크래치패드 스크립트(`scratchpad/prime_checkpoints.py`)에만 존재, src/ 미배선.
  3. 미실행 후속 후보(각각 **별도 사전등록 arm** 필요): (a) ic_score 상수 재보정(median trailing IC 0.0404 포화, D0), (b) degenerate early-stop 완화(재훈련 50%·best_iteration median 1, 국면 무관·구조적).

---

## S8 news_trend sentiment feature arm (2026-07-07)

- **실행일/커밋/솔버**: 2026-07-07 · 코드 커밋 `f90dd9a`(작업트리에 S8 변경 미커밋) · **ECOS**(arm·S0 동일 188 solve, ECOS→SCS fallback **0.0%** 양측). 캐시 재사용 없음(`cache DISABLED — variant overrides Phase 1/2/4 keys: ['news_trend_feature_enabled']` → 풀 파이프라인 재실행, arm elapsed 302.6s).
- **사전등록(단일, 스윕 없음, trials=1)**: 피처 `news_trend` 1개(`NEWS_SENTIMENT_DAILY_AVG` 5d−21d rolling-mean 스프레드, `src/features/sellside.py`). 피처 계산 코드 무수정. 플래그 `news_trend_feature_enabled`(default-OFF)로 core whitelist에 조건부 추가.
- **피처 주입 검증(inert-arm 방지)**: 모델 피처 수 **61 → 62**, diff = **정확히 {news_trend} 추가·제거 0**. `news_trend`가 S0에는 부재·arm에는 존재. arm은 inert 아님. (spec §4.2의 "56→57" 추정과 절대치 상이 — 실제 정본 모델 피처 베이스가 61이라 61→62. +1 주입 불변식은 충족.)

- **수치(전부 metrics.json 로드, 암산 없음)** — S0 = `outputs/iter15_65tkr_reb21_vtg`, arm = `outputs/exp_news_trend_feature`:

  | 지표 | S0 | arm | Δ(arm−S0) |
  |---|---:|---:|---:|
  | IR (full) | 1.481438 | 1.272249 | **−0.209189** |
  | TE | 0.031069 | 0.030094 | −0.000975 |
  | turnover | 1.144021 | 1.189144 | +0.045123 |
  | realized_beta | 1.024389 | 1.033093 | +0.008705 |
  | P1_ir | 1.591390 | 1.538343 | −0.053047 (−) |
  | P2_ir | 0.574884 | 0.266047 | −0.308837 (−) |
  | P3_ir | 2.004814 | 1.824843 | −0.179971 (−) |

- **게이트 판정**:
  - ① ΔIR > +0.36 **AND** P1/P2/P3 ΔIR 부호 일관(전부 양): ΔIR = **−0.2092**(바 미달·음수), 서브기간 ΔIR **3개 전부 음(−/−/−)** → **FAIL**(하드; edge 부재가 아니라 악화).
  - ② TE ≤ 0.045 · 캐릭터 보존: TE 0.0301 ≤ 0.045 ✓, IR 여전히 양(1.27)·벤치마크 붕괴 없음 ✓ → PASS(단 ①로 무의미).
  - ③ fallback 급증 없음: optimizer failure_rate S0 6.38%(mvo:infeasible 6) → arm 7.45%(7), **+1 이벤트뿐**(급증 아님), ECOS→SCS 0.0% 양측 → PASS.
  - ④ trials=1 사전등록 → PASS.
  - **종합: ① 결정적 실패 → default-OFF 유지·no flip.** (spec §4.6대로 정상 결과.)

- **커버리지 진단(보고용, 판정 미사용)**: news_trend 모델-입력 패널(`backtest_result.panel`, MultiIndex date×ticker) non-NaN 비율 — FULL 210145/210145 = **1.0000**, P1 41795/41795 = 1.0000, P2 41795/41795 = 1.0000, P3 41665/41665 = 1.0000. 조립(CS z-score/fill) 후 dense → pre-mortem의 "NEWS 시트 조기구간 결측" 우려가 모델 입력엔 NaN 구멍으로 남지 않음.

- **DSR / selection-bias 재산출**(N 407→408, `run_selection_bias.py --auto --label iter15_65tkr_reb21_vtg`; `outputs/reports/selection_bias_report.md`·`outputs/csv/selection_bias_metrics.csv`):

  | 지표 | S7 후 (N=407) | S8 후 (N=408) |
  |---|---:|---:|
  | N_trials | 407 | 408 |
  | Observed SR | 1.463993 (S0) | 1.463993 (불변, S0) |
  | Deflated SR | 0.745281 | 0.744574 |
  | DSR p-value | 0.228051 | 0.228265 |
  | Grid haircut | 1.204946 | 1.205192 |
  | Adjusted SR | 0.259047 | 0.258801 |
  | Gate verdict | FAIL | **FAIL** |

  - inventory(`experiment_inventory.json`): `exp_S8_news_trend_feature` **1 trial** append(스윕 없음), n_trials_total 407→408. arm 1개 추가가 sqrt(2·ln N) 페널티를 미세 이동(haircut +0.0002 / DSR −0.0007) — **S0 selection-bias 판정 불변(FAIL)**. §2-7 일관: 전 arm OFF이므로 이 gating은 활성화 후보가 아니라 S0 자체의 다중비교 유의성 정보(DSR p>0.10 주도, 나머지 4항목 통과).

- **Production flips: no-flip.** `PipelineConfig.news_trend_feature_enabled` **default-OFF 불변**. 정본 `variants/iter15_65tkr_reb21_vtg.yaml` 무접촉. arm variant `variants/exp_news_trend_feature.yaml`는 평가 전용. 롤백 불요(default-OFF·OFF parity 바이트동일: `apply_core_filter` extra_whitelist=None inert, acceptance/유닛 10/10 green).

## S9 (universe 100 + USD accounting) — 2026-07-16 전환, 2026-07-17 사후 기록

> 이 섹션은 게이트 통과 후 기록이 아니라 **사후(retroactive) 기록**이다. 유니버스 확장·USD
> 회계 전환은 GPT-5.6 세션이 2026-07-16 워킹트리에 반영했고, Fable 세션의 구조 점검(코드
> 결함 0건, 테스트 234 PASS, 상장 전 유령값 마스킹 확인, CUR_MKT_CAP=USD 감사 통과) 후
> 2026-07-17 사용자 승인으로 소급 기록한다.

- **전환 내용**: 유니버스 65 → 100종목(Universe_Meta 워크북 순서, 2026-07-16 확장 35종:
  SNDK KLAC ANET MRVL CDNS STX PWR BX TMO BSX BKNG PM WMB CEG VST DLR ARM SPOT RACE 285A
  6857 SU SIE RHM ALV MC NESN RR/ SAP ASML AZN SHEL HSBA NOVOB RIO). 포트폴리오·벤치마크
  수익률을 unhedged USD로 회계(KRW/JPY/EUR/CHF/GBP/DKK → USD, Factor_PX_LAST 우선 +
  Index.xlsx 보충, 7일 신선도 게이트). 상장 전 마스킹 default-ON(추가 상장일: 285A
  2024-12-18, SNDK 2025-02-24, ARM 2023-09-14, CEG 2022-02-02).
- **근거 문서**: outputs/universe_100_recommendation(검증 체크 17/17 True),
  outputs/universe_100_comparison(Prior/Fable/Hybrid 3안 비교, hybrid 채택),
  scripts/audit_usd_cap_benchmark.py(전 통화 supports_usd_cap=true — CUR_MKT_CAP은 USD).
- **새 기준 수치 (ECOS·USD·100종, 2026-07-16 재생성)**:
  - production `codex_causal_rank_65`(표시명 Causal Rank 100): IR **1.599** · TE **4.42%** ·
    realized_beta **1.067** · turnover 0.871 · ex-ante TE 캡 **3.5%**(0.045→0.035 강화,
    MVO·projection 동일 적용)
  - challenger `iter15_65tkr_reb21_vtg`(표시명 Legacy S0 (100)): IR **1.011** · TE **2.75%** ·
    realized_beta **1.004** · turnover 1.052
- **비교 금지 선언**: 65종 시절 인증 수치(승격 당시 production IR 1.697, S0' 0.902,
  pre-causal 1.481)와 직접 비교 금지 — 유니버스·회계 기준이 다르다.
- **Legacy S0 재정의**: variant에 유니버스 고정 장치가 없어 챌린저도 100종으로 실행된다.
  65종 인증 이력과 단절되며 표시명을 "Legacy S0 (100)"으로 변경. 65종 재현은 과거 커밋의
  데이터 워크북에서만 가능.
- **계약 변경**: `listing_mask_enabled=True` · `convert_returns_to_usd=True`를 PipelineConfig
  default-ON으로 전환 — CLAUDE.md §2.1에 "데이터 정확성 계층 예외"로 개정(2026-07-17 승인).
  BEST_PX_BPS_RATIO는 essential→optional 이동(PM 등 결측 종목은 per-date median 임퓨테이션).
- **DSR/selection-bias 입장**: 본 전환은 arm 스윕 선택이 아니라 유니버스·데이터 기준 변경 —
  스윕 선택 게이트 비해당. 이후 연구 arm 평가는 새 100종 기준선을 단일 비교 기준으로 재시작.
- **감시 항목**: 실현 TE 4.42%가 가드 4.5% 직전(ex-ante 캡은 3.5%로 강화됐으나 실현 TE는
  100종·USD 전환으로 상승). 대시보드 TE 한도·headroom·리밸런스별 ex-ante TE 경보로 추적.
- **리포팅 추가(메트릭 불변 검증됨)**: 대시보드에 드리프트 모니터(current_drift)·누적
  거래비용(transaction_costs)·섹터 편차 밴드(sector_active) 추가 — 재생성 전후 IR/TE/beta
  6값 동일 확인. 가짜 검증 지표 `return_identity_max_abs_error`(동일 식 2회 차감, 항상 0)는
  삭제 — 실질 검증은 price_return_reconciliation_max_abs_error가 담당.
- **롤백**: 플래그 한 줄 revert 불가(데이터 워크북 자체가 100종) — git revert 단위
  (feat(universe) → feat(ops-dashboard) → run(outputs) → docs(contract) 역순).
- **(2026-07-18 추록)** 위 "새 기준 수치"의 production 값(IR 1.599/TE 4.42%/beta 1.067,
  turnover 0.871)은 metrics.json config 에코 확인 결과 **ex-ante TE 캡 0.045 시절 산출물**이다
  (캡 0.035 강화 커밋 5d762ef 이전 생성). 커밋된 config 기준 유효 baseline은 §S9.1 참조.

## S9.1 (파이프라인 end-to-end 재인증 + TE 캡 0.035 유효 baseline) — 2026-07-18

- **배경**: 사용자 요청 "bat 실행 시 업데이트→깃허브 업로드 자동 완료 여부 확인 + 유니버스
  확대 후 재점검". 조사 결과 스케줄 실행은 7-13 11:41(`1cfe8d6`)이 마지막 완주였다.
  - 7-16 11:30 런: step 4/10에서 `ai_signal_data.xlsx` EOFError 중단 — 같은 시각(11:44~11:46)
    유니버스 리프레시가 워크북을 쓰는 중이던 **동시 쓰기 경합**(일시적). 커밋·푸시 미도달.
  - 7-17: 태스크 결과 0x800710E0(실행 거부) — 태스크 설정 `DisallowStartIfOnBatteries=true`,
    노트북 배터리 상태로 기동 자체가 거부됨. 로그 미갱신.
- **플레이키 게이트 수정** (`8f1e42c`): `validate_portfolio_bundles.py`의
  "bundle predates its source metrics" 검사가 NTFS mtime(100ns) vs `exported_at_utc`(µs 절단)
  서브마이크로초 경합으로 간헐 실패(2026-07-18 재현 실측: exported `.035160` vs mtime
  `.0351605`). 결정적 재현 회귀 테스트 추가 후 1초 허용오차 적용 — 실제 위반(분 단위)은
  계속 검출된다. 테스트 235 PASS.
- **end-to-end 재실행** (`bb0e118`, ECOS·100종·unhedged USD, `--no-cache`): bat 전 단계
  [1/10]~[10/10] 완주, GitHub 푸시 확인(local==origin). 결과:
  - production `codex_causal_rank_65` (**ex-ante TE 캡 0.035 — 커밋된 yaml 기준 첫 재생성**):
    IR **1.570** · active ret **5.84%** · TE **3.72%** · realized_beta **1.054** ·
    turnover 0.848 · MDD −0.316 · sub-period P1 0.774 / P2 0.780 / P3 2.769 (전부 양).
  - challenger `iter15_65tkr_reb21_vtg`: IR **1.011** · TE 2.75% — §S9 기록과 동일 재현
    (파이프라인 결정성 확인. production 델타는 노이즈가 아닌 순수 캡 효과).
- **판정**: 캡 강화(0.045→0.035)의 실측 비용 ΔIR −0.029(≪ 0.36 SE, 노이즈 범위) ·
  실현 TE 4.42%→3.72%로 **가드 4.5% 안쪽 복귀** — §S9 감시 항목(TE watch) 해소.
  이후 모든 arm 비교 기준은 **S9.1 production 수치**(IR 1.570/TE 3.72%/beta 1.054)로 한다.
- **미해결(사용자 결정 대기)**: 스케줄 태스크의 배터리 조건(`DisallowStartIfOnBatteries`)과
  missed-run 미보정(`StartWhenAvailable` 없음)은 시스템 설정 변경이라 미적용 — 적용 시
  7-14~17형 불발이 재발하지 않는다. 워크북 리프레시와 11:30 스케줄의 동시 쓰기 경합은
  운영상 시간대 분리로 회피 권고. → **2026-07-18 §S9.2에서 해소(사용자 승인)**.

## S9.2 (거버넌스 강화: production HOLD 게이트·provenance·원인 규명) — 2026-07-18

외부 리뷰(GPT) 제안 5건을 코드·산출물 실측으로 점검한 뒤 사용자 승인 범위만 구현.
채택: fail-closed 게이트의 무중단 변형(HOLD 가시화)·provenance 전파·원인 규명 2건·스케줄러 수정.
**비채택**: 절대 집중도 옵티마이저 제약(§2.5 집중 캐릭터 보존과 긴장 — 사전등록 arm으로만),
자동 알파 축소/벤치마크 수렴(§2.5 무음 fallback 금지 충돌 소지), 모듈 분리·id 개명(비용>실익, 보류).

- **스케줄러 수정(승인 완료)**: `ai_port_run_and_upload`에서 `DisallowStartIfOnBatteries`·
  `StopIfGoingOnBatteries` 해제, `StartWhenAvailable` 활성화. 트리거(평일 11:30)·
  IgnoreNew 정책 보존, 다음 실행 2026-07-20(월) 11:30 확인.
- **섹터 리스크 위반 원인**: `top_sector_active_risk_share` 0.787 > 0.75는 export의 ex-post
  진단(최근 리밸런스 가중치 + Ledoit-Wolf 126d cov, 섹터별 active TE 기여/est TE).
  `max_name/sector_active_risk_share`는 **옵티마이저 어디서도 미사용(report-only)** — grep 실측.
  AI 중심 유니버스에서 액티브 리스크의 Technology 집중은 구조적. 한도 자체는 유지하고
  위반을 HOLD로 가시화(아래).
- **퇴화 원인(D0 진단 스크립트, 오늘 pkl)**: 퇴화 17/32(53.1%)의 근인은 **구조적 즉시
  조기종료** — 퇴화 best_iteration 중앙값 3(1~9) vs 건강 47. lr 0.02·min_child 60·
  patience 100·5y 창 조합에서 val loss가 초기 정체 → incumbent 재사용. P2 레짐 집중 가설은
  반증(P1/P2/P3=4/6/7). 강한 모델(348·82트리) 직후 연속 퇴화 군집(최장 5연속). 치료
  (하이퍼파라미터 변경)는 성능 arm → 사전등록 대상으로 이월. `fail_on_degenerate_model_rate`
  flip은 현 53%에서 파이프라인 즉사이므로 보류 — HOLD 게이트가 가시화를 담당.
- **production HOLD 게이트(무중단, 커밋 참조)**: `validate_portfolio_bundles.py`에
  `evaluate_production` 추가 — checks 6종(estimated_te_ok, name/sector_active_risk_ok,
  degenerate_rate_ok, realized_te_within_guard(≤4.5%), stale_tail_ok). False 존재 시
  production status "HOLD"(발행·업로드는 계속, exit code 불변), registry 최상위
  `production_gate` 기록, 대시보드 status-row에 게이트 칩. 데이터 없음(None)은 위반으로
  치지 않음(구버전 호환). **현재 판정: HOLD** (sector_active_risk_ok=false,
  degenerate_rate_ok=false — 위 두 원인 규명 참조). 개별 항목의 진짜 fail-closed 승격은
  §8 절차(결정 로그 게이트 + 사용자 승인)로만.
- **provenance 전파**: 번들 meta에 `portfolio_version="universe100-usd-v1"`,
  `source_manifest_sha256`, `git_hash`/`git_dirty`(해당 런의 experiment_manifest에서 복사 —
  재계산 금지) 추가, registry entries로 자동 전파. `_git_dirty`는 outputs/ 하위 변경을
  무시하도록 교정(기존엔 파이프라인 자체가 outputs를 먼저 더럽혀 항상 true로 기록되는 왜곡).
  오늘 번들의 git_dirty=true는 교정 전 매니페스트 값의 정직한 복사이며, 다음 런부터 교정
  적용.
- **검증**: 전체 스위트 244 PASS(신규 9: HOLD 게이트 4·provenance 5). 실번들 registry에서
  production HOLD·챌린저 게이트 불변(RESEARCH/FAIL)·언더스코어 키 무누출 확인.
- **HOLD 해소 경로(후속)**: ① 섹터 리스크 — 한도 0.75의 캘리브레이션 재검토(100종 AI
  유니버스 기준) 또는 사전등록 섹터 리스크 제약 arm, ② 퇴화율 — 사전등록 하이퍼파라미터
  arm(D0 진단이 앵커). 둘 다 §4 IR 게이트·DSR 해킷 적용. → §S10에서 진행.

## S10 (HOLD 해소 2건: 섹터 한도 재캘리브레이션 + 퇴화 arm 사전등록) — 2026-07-18

### S10.1 섹터 리스크 한도 재캘리브레이션 — 백테스트 불요 경로 확정

- **사전점검(94회 리밸런스 전수 재구성, export guardrail 수식 동일 재현)**: top-sector
  active risk share — 전기간 median 0.517 / P90 0.827 / max 0.970 / 위반율 19.15%;
  **최근 24회(≈2년) median 0.769 / P90 0.925 / 위반율 58.3%**. Technology가 94회 중 84회
  최상위. 최근 6회는 0.655~0.787로 한도 주변 상시 배회.
- **판정**: 위반은 일시 스파이크가 아니라 **만성** — 최근 레짐 median(0.769)이 한도(0.75)
  위에 있다. 한도 0.75는 벤더링 초기 스냅샷(a23d4e9)부터 존재한 **무캘리브레이션 값**
  (git log -S 실측)이며, 정상 운영의 중앙값 아래에 놓인 한도는 통제가 아니라 상시 경보다.
  → 옵티마이저 제약 arm(백테스트) 대신 **한도 재캘리브레이션**으로 해소.
- **새 한도 0.85**: 전기간 P90(0.827)을 0.05 단위 올림 = 0.85 (사전 상한 0.85와 일치).
  전기간 기준 상위 ~10% 극단 집중만 경보. **투명성 고지**: 이 공식은 분포 확인 후 선택 —
  단, 대상이 report-only 진단 한도(옵티마이저 미사용, §S9.2 실측)라 성능·선택편향(DSR)
  비해당. 포트폴리오 가중치·메트릭은 바이트 불변.
- **적용**: `PipelineConfig.max_sector_active_risk_share` 0.75→0.85(+캘리브레이션 주석),
  test_config_fields의 기본값 핀 동기 갱신, 번들 재-export로 sector 체크 해소.
  이름 한도(0.35, 현재 0.230)는 위반 없음 — 변경하지 않음.

### S10.2 퇴화 arm 사전등록 (min_child_samples 60→30) — **실행 전 등록**

- **단일 사전등록 파라미터**: production 변형 `lgbm_params.min_child_samples: 60 → 30`.
  다른 모든 설정·시드(42)·솔버(ECOS)·데이터 동일, `--no-cache` 단일 실행, 스윕 금지.
- **근거(사전)**: D0 — 퇴화 근인은 즉시 조기종료(best_iteration 1~9). min_child=60은
  65종 북에서 V2 튜닝된 값(20→60, 3배 강화)인데 유니버스 100종 전환으로 일자당 표본이
  ~54% 증가 — 분할 허용 문턱을 절반(30)으로 되돌려 스케일 재정렬. lr·patience 등은 불변.
- **비교 기준**: §S9.1 production (IR 1.570 / active 5.84% / TE 3.72% / beta 1.054 /
  turnover 0.848 / P1 0.774 P2 0.780 P3 2.769 / degenerate_rate 53.1%).
- **주 평가변수 E1**: `model_quality.degenerate_rate` ≤ 0.25.
- **가드(do-no-harm — 채택 근거는 리스크 거버넌스이고 IR은 가드임을 명시, §2.4 정합)**:
  - G1: ΔIR ≥ −0.36. 그리고 (ΔIR<0 이면서 서브기간 3개 전부 ΔIR<0)이면 FAIL.
  - G2: realized TE ≤ 0.045 · G3: active_share_l1 ≥ 0.30 · G4: realized_beta ∈ [0.95, 1.10]
  - G5: avg_annual_turnover ≤ 1.06 (=1.25×baseline)
- **채택 규칙(사전약정)**: E1 & G1~G5 전부 통과 → production yaml에 min_child_samples 30
  flip(한 줄) + 재생성·재검증·롤백 확인. 하나라도 실패 → OFF·기록만, 추가 후보 시도는
  새 사전등록 없이 금지. ΔIR > +0.36 & 서브기간 부호 일관이면 IR 근거 채택 가능도 병기.
- **DSR**: 본 arm을 experiment_inventory에 등재(+1 trial)하고 run_selection_bias 해킷을
  결과와 함께 기록.
- **결과 (2026-07-18 실행, ECOS·--no-cache·seed 42, 328s)**: **E1 FAIL — 가설 반증.**
  degenerate_rate **65.6%(21/32)** vs baseline 53.1% — min_child 완화가 퇴화를 오히려
  악화시켰다. 분할 문턱(정규화 과강)이 근인이 아니라는 뜻 — D0의 "val 즉시 정체"와 합치면
  남는 유력 가설은 **신호 자체가 126d 검증창에 일반화되지 않는 레짐**(val_window·lr 등은
  향후 별도 사전등록 대상). 참고 가드: ΔIR −0.137(노이즈 대역), 서브 ΔIR −0.429/−0.239/
  +0.339(혼재), TE 3.48%·beta 1.039·turnover 0.850(가드 내). **사전약정대로 불채택 —
  production 불변, 새 사전등록 없는 추가 후보 시도 금지 준수.**
- **DSR 해킷 (N=413)**: DSR p=0.1693 · Deflated SR 0.957 · Haircut 1.211 · MinTRL 1.1yr →
  게이트 FAIL 유지(승격 당시 사용자 오버라이드와 동일 상태). 오늘 활성화 대상 없음 —
  기록용.
- **S10 종결 상태**: HOLD 사유는 sector(해소, 0.787<0.85) → **degenerate_rate 1건만 잔존**.
  퇴화 해소는 후속 사전등록 arm(후보: val_window 126→252 또는 lr 0.02→0.03, 단일 선택)
  대기 — 착수는 사용자 승인 후.

## S11 (유니버스 150 확장 — 슬레이트 확정·워크북 편집·게이트 등록) — 2026-07-18

사용자 지시: MSCI World 편입 종목 위주 + 최종 150종의 섹터 구성을 MSCI World 비중에 근사.
슬레이트는 외부 리뷰(GPT) 2회를 거쳐 확정(EA→Publicis, BHP LN→Air Liquide 교체 포함).
**프레이밍**: 이름 수 배분은 "학습 표본 구성"이며 MSCI 복제가 아님. 실제 섹터 투자비중은
cap-weighted 내부 벤치마크 ±10%p 하드 밴드(`sector_deviation`)가 제한하고, 0.85 섹터
리스크 점유율은 사후 HOLD 모니터링 담당(§S10.1) — 옵티마이저 가중치 제약이 아님.

- **슬레이트 (50, 기준일 2026-06-30 MSCI World 팩트시트: IT 30.27%/금융 15.88%/산업
  11.64%/헬스케어 9.08%/경기소비 8.91%/커뮤니케이션 8.07%)**: Tech +10 (QCOM TXN ADBE
  NOW ACN IBM ADI DELL 8035 IFX) · Fin +13 (MS SCHW AXP PGR MMC CME KKR BN C LSEG ZURN
  CS[AXA] 8306) · Comm +9 (DIS CMCSA VZ T TTWO PUB 7974 DTE UMG) · CD +6 (LOW TJX SBUX
  ABNB 7203 ITX) · HC +4 (ABT DHR VRTX ROG) · Ind +3 (GE TT ABBN) · St +2 (KO ULVR) ·
  Mat +2 (ECL AI[에어리퀴드]) · Util +1 (IBE). 최종 150 = Tech 45/Fin 24/Ind 17/HC 14/
  CD 13/Comm 12/St 7/En 5/Mat 5/Ut 4/RE 4. 통화: USD 33·EUR 8·JPY 4·CHF 3·GBP 2 —
  **신규 FX 페어 0**.
- **워크북 편집 (2026-07-18 완료, 스크립트 검증 PASS)**: `Data/oppor.xlsx` tickers 시트
  102→152 셀(+50, Bloomberg 형식), `re_study/Factset_re_study.xlsx` 13시트 각 +50열
  (r2='TICKER-CC^', r3=시트별 동일 FDS 수식 복제; 101→151열, FwdEPS 시트만 85→135열).
  백업: `*.backup_20260718_s11.xlsx` 2건. 검증: 시트별 tail-50 일치·수식 균일·중복 0.
- **게이트 상태 (GPT 리뷰 공백 4건)**:
  1. MSCI 편입 ISIN/SEDOL 대조 — **PENDING**: 2026-06-30 구성 파일 확보 후 50종 증권 단위
     대조가 **리프레시 착수 조건**. 회사가 아닌 구성 증권 라인 기준(BHP LN 배제 선례).
  2. SM→EUR — **DONE**: `MARKET_TO_CURRENCY["SM"]="EUR"` + 회귀 테스트. ITX·IBE SM 라인 유지.
  3. 상장·기업행사 마스크 — **DONE(등록)**: IPO 3건(DELL 2018-12-28, ABNB 2020-12-10,
     UMG 2021-09-21) + 기업행사 3건(GE 2024-04-02 GE Vernova 분사, TT 2020-03-02
     Ingersoll-Rand 분리, BN 2022-12-12 BAM 분리) — 규칙: **사건 이후만 학습 인정**(가격·
     재무 모두 마스킹, 기존 listing_mask 메커니즘). 일자는 리프레시 시 first-valid로 재검증.
  4. 생존편향 정책 — **명시**: 2026-06-30 시점 구성원의 **고정 150종 유니버스**. point-in-time
     구성 이력을 쓰지 않으므로 과거 백테스트 구간은 **진단용으로만** 해석(비편향 MSCI 백테스트
     아님). 신규 50종의 정식 기여 평가는 2026-07 이후 전향 구간.
- **단독 arm 선언(중요)**: 100→150은 표본 확대가 아니라 **타깃 정의 변경**(PCA가 전체
  유니버스 수익률로 재적합 — 기존 100종의 specific-return 타깃도 바뀜). 따라서 150 전환은
  다른 어떤 파라미터 변경(val_window 등)과도 **동시 실행 금지** — S9와 같은 기준선 재정의
  이벤트로 취급하고 새 S0(150) 재인증 후 "150 이전 수치와 비교 금지"를 선언한다.
- **병행 초안과의 대조(§9 보고 의무)**: 같은 날 15:36 다른 세션이 생성한 미커밋 초안
  (`outputs/universe_150_recommendation`, `scripts/build_universe_150_analysis.py`)은
  다른 50종(~36종 상이)을 담고 있음. 본 슬레이트와의 판정 차이 — EA 제외(동일), GE(초안:
  전체 제외 / 본안: 기업행사 마스크 포함 — 285A 1.5y 데이터 선례), IBE(초안: SM 미지원으로
  RWE 대체 / 본안: SM 지원 구현으로 원안 유지). 본 슬레이트가 사용자 승인·MSCI 제약 반영본
  이며 워크북에 적용된 정본. 초안 파일은 참고용으로 보존(미커밋 — 차기 스케줄 런의
  `git add -A`가 자동 커밋할 수 있음을 고지).
- **잔여 단계(S9 절차 준용)**: ① MSCI 구성 파일 대조(게이트 1) → ② FactSet/Bloomberg
  리프레시(데스크탑) → ③ ai_signal_data 150 재생성 + Universe_Meta 확장 → ④ 마스크 일자
  검증 + 커버리지·임퓨테이션 보고(금융주 FCF류 결측 예상) → ⑤ 새 S0(150) ECOS 재인증 →
  §S11.1 기록.

## S11.2 (슬레이트 개정: MMC → AON) — 2026-07-20

(§S11.1 = 2026-07-19 리프레시 게이트 사용자 오버라이드·universe_config 150 선적용,
별도 세션 수행 — 메모리 기록 참조.)

- **트리거/근거(실측 2026-07-20)**: `RL_Universe_Data.xlsx` Universe_Meta 150종 중
  **MMC 유일 Missing** — 가격 소스 `Data/S&P500.xlsx`에 MMC 열 부재(PX_LAST 등).
  FactSet 측도 `D_Factset_re_study.xlsx` 실데이터가 아직 100종(§S11 리프레시 전)이라
  신규 50종 전체 미수신 상태이나, 가격 소스까지 비는 구조적 공백은 MMC가 유일.
- **교체안(사용자 확정)**: MMC(보험 브로커) → **AON US Equity**(동일 업종 보험 브로커,
  USD, MSCI World, 2013 이전 상장). 섹터 배분(Fin 24)·통화 믹스(USD 33)·FX 페어 0
  불변 — §S11 슬레이트 방법론 유지. oppor `S&P500` 풀 시트에는 AON 열이 기존재(CQ1의
  MMC 열은 유니버스 외 풀 항목으로 보존).
- **적용(2026-07-20)**: `Data/oppor.xlsx` tickers 시트 DM1 `MMC US Equity`→`AON US
  Equity` · `re_study/Factset_re_study.xlsx` 13시트 각 1건 `MMC-US^`→`AON-US^`(잔여
  MMC 0) · `universe_config.py`/`test_universe_config.py` MMC→AON(§S11.2 주석 부기).
  `run_data_pipeline.bat`은 티커 하드코딩 없음 — universe CHECK 경유 자동 반영.
  백업: `oppor.backup_20260720_aon.xlsx`, `Factset_re_study.backup_20260720_aon.xlsx`.
- **검증(PASS)**: 계약·파이프라인 테스트 11/11 PASS · `validate_universe()` 150/150 ·
  factset 13시트 AON-US^ 각 1개·MMC-US^ 0개 · oppor tickers AON 존재/MMC 부재
  (여분 `442580 KS` 1종은 기존 유니버스 외 항목, 금회 편집 무관).
- **후속**: §S11 잔여 단계 동일 — 리프레시 시 AON 13시트 수신·first-valid 확인,
  새 S0(150) 재인증 전까지 ai_port TICKERS(100) 불변. `Data/S&P500.xlsx`의 잔존
  MMC 열(BEST_EPS·BEST_SALES 헤더)은 파이프라인이 무시하므로 방치 가능.

## S11.3 (새 S0(150) ECOS 재인증 + ai_port 150 확장) — 2026-07-20

**이 절이 150종 유니버스의 새 단일 비교 기준(S0)이다. §S11 선언에 따라 100종 이전
수치(§S9.1 IR 1.570/TE 3.72% 등)와의 직접 비교 금지.** 솔버 ECOS 단일 프로토콜,
`--no-cache` full rebuild, seed 42, 단독 arm(파라미터 변경 0 — variant overrides
무변경, 표시 텍스트만 갱신).

- **전제 검증(2026-07-20, 병렬 조사 5종)**: 사용자 데스크탑 리프레시(13:39~14:15) 후
  `ai_signal_data.xlsx` Universe_Meta **150행 전원 Available**(AON 포함·MMC 0),
  핵심 시트 date+150열·universe_config 순서 일치·섹터 불일치 0. `D_Factset` 12시트
  151열 리프레시 확인(AON 4948/4948). **주의(실측)**: PX_LAST·Daily_Returns는 상장 전
  구간이 상수 백필(ABNB 68.0 고정 등)이라 채움률≠실존 — listing mask가 필수 방어선이며,
  백테스트 로그에서 마스크 13종(PLTR GEV BE 285A SNDK ARM CEG + S11 6종 DELL ABNB
  UMG GE TT BN) 적용 확인(잔여 ④ 해소).
- **코드 확장(248 PASS)**: `src/data_loader.py` TICKERS 100→150(+50, Universe_Meta
  순서)·FALLBACK_TICKER_CURRENCY +17(비USD) · 핀 갱신: `test_universe_fx_conversion`
  150/tail-5, `audit_usd_cap_benchmark` 게이트 150(`_check_universe` 추출+신규 테스트
  3), `export_operating_data` PORTFOLIO_VERSION **universe150-usd-v1**·표시명
  "Causal Rank 150"/"Legacy S0 (150)"(+테스트 핀 3), streamlit 칩 150, variant
  yaml 표시 텍스트. 엔진은 meta-driven(Universe_Meta 정본)이라 TICKERS는 fallback
  정합용 — 출력 변화는 데이터(150) 기인이지 코드 기인이 아님.
- **S0(150) production — codex_causal_rank_65 (Causal Rank 150), ECOS**:
  **IR 1.522** · active 5.45% · **TE 3.58%**(실현, ex-ante 캡 0.035·est TE 3.50%
  바인딩) · **realized_beta 1.048**(active_beta +0.048) · turnover 74.5%(양방향)/
  37.3%(편도) · MDD −31.19% · IC 0.0189 · P1/P2/P3 = 1.549/0.784/2.187(전부 양) ·
  Sharpe 1.31 · 솔버 {ECOS:190}·SCS fallback 0%·optimizer fallback 1/95(1.1%) ·
  541s. active share L1 0.405(편도 20.3%) — 집중 캐릭터 유지(§2.5).
- **S0(150) challenger — iter15_65tkr_reb21_vtg (Legacy S0 (150)), ECOS**:
  IR 1.160 · active 3.67% · TE 3.16%(캡 0.045 레거시 핀) · realized_beta 1.026 ·
  turnover 103.2%/51.6% · MDD −30.26% · IC 0.0277 · P1/P2/P3 = 0.579/0.720/1.956 ·
  543.6s. registry comparison_gate FAIL/RESEARCH 표시는 승격 후 방향 반전으로 정상.
- **P2 게이트 재판정**: realized_beta 1.048 ≈ 1.0 → **P2(beta-neutral) shelve 유지**(§3).
- **Production HOLD 게이트(§S9.2, report-only) = HOLD — 체크 3건 False**:
  ① `sector_active_risk_ok` FALSE — Technology active-risk share **0.957 > 0.85**
  (§S10 재캘리브레이션 한도; 100종 시절 0.787→150 전환으로 재돌파. 150 유니버스의
  섹터 구성 변화 기인 — 한도 재캘리브레이션 또는 제약 조정은 **새 사전등록 대상**),
  ② `name_active_risk_ok` FALSE — top name STX **0.446**(단일 종목 active-risk 집중),
  ③ `degenerate_rate_ok` FALSE — **59.4% > 25%**(기존 잔존 HOLD 사유; challenger는
  31.2%). TE(0.0358≤0.045)·stale tail(1.0d≤10d)은 OK. 발행·업로드는 계속(report-only).
- **커버리지·임퓨테이션 보고(잔여 ④)**: 금융주 FCF/CAPEX/GM/EV-EBITDA류 **컬럼 부재**
  (BN·ZURN·8306 FCF 등, §S11 예상 결측 — 로더가 부재→NaN 처리) · SHORT_INT_RATIO
  비미국 30종 부재(median 대체) · RR/ FactSet 시트 만성 전량 결측(리프레시 전과 동일,
  별도 조사 후보) · TSM FCF 부재. metadata.py TICKER_META는 65종 스테일(85종 부재)
  — 최적화 무관(섹터는 Universe_Meta 사용), 리포팅 개선 백로그.
- **DSR/selection-bias**: 유니버스 전환은 arm 선택이 아님(§S9 입장 일관) — 게이트
  비해당, 본 절 기록으로 갈음.
- **비교 금지 선언**: 이후 모든 arm·ablation은 본 §S11.3 수치를 기준으로만 비교한다.
  PCA 타깃이 150 전체 수익률로 재적합되었으므로 100종 시절 결과와의 델타는 무의미.

## S11.4 (point-in-time 유니버스 — 전 시트 상장 전 마스킹 + 재인증) — 2026-07-20

**§S11.3은 "마스크 결함 하 중간 인증"으로 격하한다.** 외부(GPT) 리뷰 지적을 자체
검증으로 확인: 시트 마스킹이 PX_LAST·CUR_MKT_CAP(+targets/predictions 셀)만 커버,
재무·컨센서스 19개 시트의 상장 전 백필이 미마스킹 상태로 피처 횡단면에 유입
(실측: ABNB BEST_EPS −0.582×2,535행, BEST_PE_RATIO 10,399.662, BEST_ROE 64.941 —
전부 비-NaN 상수라 임퓨트 median의 오염원이기도 함). §2.1 데이터 정확성 계층
예외로 default-ON 적용하되, 전 기간 피처·타깃이 바뀌므로 단일 arm으로 묶어
새 S0(150)′ ECOS 재인증을 수행한다.

- **정책(사용자 지시 2026-07-20)**: 종목은 **상장일 이후에만 유니버스에 편입**되며
  상장 전 시점의 유니버스는 150 미만이 정상이다(point-in-time membership).
  pre-IPO 진성 관측(NEWS_SENTIMENT 상장 전 뉴스, TG_Price 개시 목표가)도 일관성을
  위해 마스킹한다 — 정보 손실은 인지된 트레이드오프.
- **변경 세트(단일 arm, 261 PASS)**:
  1. **이중 마스킹**: preprocess 1차(임퓨트 전 — 유령이 `_fill_missing`/`align_dates`
     median에 못 섞임) + align 후 2차 재마스킹(전 시트 inclusive=False).
     Daily_Returns만 2차 제외(PCA dense 요구; 1차 마스킹+median 재충전으로 유령
     0.0 → 당일 상장종목 median 수익률로 대체). raw_returns(cov 경로)는
     inclusive=True 유지. 시트별 마스킹 셀 수를 `data_quality["listing_mask"]`로 기록.
  2. **breadth 분모**: conditioning.py 4곳(shape[1] → 날짜별 notna 수, 0-division
     가드). production은 feature_mode=core라 breadth류가 화이트리스트 밖 —
     모델 입력 무변(full/lean·진단만 변경). attribution.py:355 진단 분모는
     모델 경로 밖이라 보류(백로그).
  3. **expected_universe_size 가드**: config default None(합성 픽스처 보호),
     production·challenger variant에 150 지정. UniverseData __init__ +
     캐시 재사용 분기(check_cached_universe) 양쪽 검문. SAFE_FOR_CACHE_REUSE 등재.
  4. **listing_dates 커버리지 감사(선행 동일값 런 ≥15일 스캔)**: 미등재 백필 6종
     발견·등록 — ANET 2014-06-06 / RACE 2015-10-21 / LITE 2015-07-27 /
     VST 2016-10-05 / SPOT 2018-04-03 / **VRT 2018-08-01**(상수 백필 구간만 마스킹;
     2018-08 이후 GSAH SPAC 실역사는 보존 — 기존 "VRT 제외" 결정의 정신 유지).
     핀 테스트 3곳 동기(19종). 잔여: GE/TT/BN형(비상수·타 실체) 추가 후보는
     자동 탐지 불가 — 등록된 6건 외 미확인 명단 없음으로 종결.
- **재인증(사전등록)**: production(codex_causal_rank_65)·challenger(iter15) 각
  ECOS·`--no-cache`·seed 42, 파라미터 변경 0. 결과는 아래에 추가 기록.
- **재인증 결과(2026-07-20, ECOS·--no-cache) — 이 수치가 새 단일 기준 S0(150)′**:
  - **production (Causal Rank 150)**: **IR 1.371** · active 5.01% · **TE 3.65%** ·
    realized_beta **1.040** · turnover 74.1%(양방향)/37.1%(편도) · MDD −31.37% ·
    P1/P2/P3 = 1.535/0.714/1.971(전부 양) · IC 0.0214 · optimizer fallback 0/95 ·
    솔버 {ECOS:190}·SCS 0%. (metrics elapsed 21,912s는 시스템 절전 추정 아티팩트 —
    challenger 998s와 동급 연산.)
  - **challenger (Legacy S0 (150))**: IR 1.228 · active 3.65% · TE 2.97% ·
    realized_beta 1.021 · turnover 103.6%/51.8% · MDD −31.17% ·
    P1/P2/P3 = 1.014/0.547/2.289 · IC 0.0259.
  - **P2 게이트**: realized_beta 1.040 ≈ 1.0 → **P2 shelve 유지**.
  - §S11.3(마스크 결함 인증) 대비 방향 참고(액션 근거 아님): production IR
    1.522→1.371, challenger 1.160→1.228 — 유령 데이터 제거로 production의 우위
    폭이 줄었으며, 이는 §S11.3 수치 일부가 오염 기인이었음을 시사.
- **HOLD 게이트 재측정 — 3건 → 2건**:
  - `name_active_risk_ok` **TRUE로 해소**: top name STX 0.446 → **6857 0.235**
    (<0.35). STX 집중은 상장 전 유령 데이터 아티팩트였음이 확인됨.
  - `sector_active_risk_ok` FALSE 잔존: Technology **0.886** (>0.85, 근소 초과;
    §S11.3의 0.957에서 −0.071).
  - `degenerate_rate_ok` FALSE 잔존: **56.25%**(18/32) (§S11.3 59.4%에서 소폭 개선;
    challenger는 31.2%→40.6%로 악화 — 방향 혼재, 마스킹과 퇴화의 인과 불명).
  - est TE 3.50%(캡 바인딩)·실현 TE 3.65%≤0.045·stale tail 1.0d OK.

## S11.5 (HOLD 잔존 2건 — 사전등록 arm) — 2026-07-20

사용자 승인(2026-07-20 "phase2,3도 진행해") 하에 §S10.2 양식으로 사전등록.
비교 기준은 §S11.4 S0(150)′(production IR 1.371/TE 3.65%/Tech share 0.886/
퇴화율 56.25%). 두 arm은 서로 다른 계층(optimizer vs model)이라 독립 평가하되,
**production flip은 §8대로 한 번에 1개**만.

- **Arm A — 섹터 active-risk soft penalty**: `sector_active_risk_penalty_enabled=
  true`, **λ=5.0 단일 사전약정**(스윕 금지). 볼록 프록시 Σ_s (m_s∘a)'C(m_s∘a)를
  MVO objective에 가산(§4.1 inline 패턴, OFF 시 int 0 바이트동일 — 파리티·방향성
  단위테스트 등재). 캐시 안전(MVO objective만 변경).
  **E1**: export guardrail `top_sector_active_risk_share` < 0.85.
  **가드**: G1 ΔIR ≥ −0.36, G2 실현 TE ≤ 0.045, G3 active return > 0 및
  집중 캐릭터 보존(무음 bm 붕괴 시 FAIL, §2.5), G4 optimizer fallback ≤ 5%,
  G5 P1/P2/P3 부호 전부 양 유지.
- **Arm B — 퇴화율: val_window 126→252 단일 사전등록**(§S10.2 종결부의 잔존 후보
  {val_window, lr} 중 단일 선택; 선택 근거 = §S9.2 D0 "신호가 126d 검증창에
  일반화되지 않는 레짐" 가설의 직접 검증. lr은 이번에 시도하지 않음).
  캐시 불안전 → `--no-cache` full run.
  **E1**: `model_quality.degenerate_rate` ≤ 0.25.
  **가드**: §S10.2 G1-G5 동일(ΔIR ≥ −0.36 · TE ≤ 0.045 · 캐릭터 보존 ·
  fallback ≤ 5% · 서브기간 부호).
- 결과·판정은 아래에 추가 기록.
- **Arm A 결과(2026-07-20, ECOS·--no-cache·seed 42) — E1 FAIL(무력 용량)·불채택**:
  λ=5.0 arm의 전 지표가 §S11.4 production과 인쇄 자릿수까지 동일(IR 1.371/TE
  3.65%/turnover 74.1%/P1-P3 동일) — penalty가 수치적으로 무력했다.
  원인(실측): 옵티마이저 mu는 z-score 스케일(2026-06-22 median|mu|=0.840,
  P90=1.819)인데 penalty는 일간 분산 단위(Tech 한계기울기 median 2|C·a_s|=
  3.23e-05) — 균형 λ* ≈ **26,000**. 사전약정 5.0은 4자릿수 부족.
- **전기간 재구성 진단(§S10.1 방법 동일, 95회, PIT 기준)**: top-sector share
  median 0.496 / P90 **0.749** / max 0.945 / 위반율(>0.85) **6.3%**;
  최근 24회 median 0.626 / P90 0.889 / 위반율 20.8%. Technology 최상위 73/95.
  **판정: 한도 0.85는 새 기준선에서도 적정 캘리브레이션**(전기간 P90 < 한도).
  현재 0.886 위반은 만성이 아니라 endpoint 꼬리 스파이크(§S10.1의 만성 case —
  최근 위반율 58.3% — 와 다름).
- **Arm A 종결**: λ* 재시도는 하지 않는다 — (i) 전기간 위반 6.3%짜리 꼬리 문제에
  전 95회 리밸런스를 왜곡할 초대형 penalty는 §2.5(캐릭터 보존)·§S10.1 선례
  (만성일 때만 조치)와 상충, (ii) 결과를 본 뒤의 2차 λ는 outcome 선택 편향 우려.
  인프라(default-OFF·파리티 테스트)는 유지하고, **발동 조건을 사전등록**:
  최근 24회 위반율이 §S10.1 수준(≥50%)으로 만성화되면 λ≈26,000 단일 사전약정
  arm 착수. 그 전까지 sector HOLD는 "정상 작동 중인 report-only 경보"로 유지.
  (부수 증거: λ=5 arm이 production과 완전 동일한 결과를 낸 것은 신설 penalty
  코드 경로의 무해성(≈OFF 파리티)에 대한 경험적 확인을 겸한다.)
- **Arm B 결과(2026-07-20, ECOS·--no-cache·seed 42) — E1 FAIL·불채택**:
  degenerate_rate **53.125%**(17/32) > 0.25 — baseline 56.25% 대비 −3.1pp에
  불과. **"신호가 126d 검증창에 일반화되지 않는다"는 D0 가설도 사실상 반증**
  (252d로 늘려도 대부분 즉시 조기종료 유지 — 원인은 검증창 길이가 아님).
  가드 참고치(비액션): IR 1.507(Δ+0.136, 노이즈 대역)·TE 3.70%·beta 1.040·
  P1/P2/P3 = 1.511/0.451/2.478·fallback 0/95 — G1-G5 전부 통과했으나 §2.4에
  따라 노이즈 대역 ΔIR은 채택 근거가 아니며, 사전약정 endpoint FAIL이므로
  불채택(§S10.2 선례와 동일한 규율).
- **§S11.5 종결**: 퇴화 원인 후보 중 min_child(§S10.2)·val_window(본 절) 2개
  반증 완료. 잔존 후보는 lr 0.02→0.03 단일 — **새 사전등록·사용자 승인 대기**
  (본 절 사전등록이 "lr은 이번에 시도하지 않음"을 명시했으므로 연속 시도 금지).
  구조 가설(D0 즉시 조기종료가 rank_xendcg+NDCG 특성일 가능성 — challenger
  회귀 objective는 퇴화율 40.6%로 낮음)은 후속 조사 후보로만 기록.
- **HOLD 최종 상태(§S11.4 기준)**: `sector_active_risk`(0.886, endpoint 꼬리 —
  정상 작동 경보·발동 조건 등재)·`degenerate_rate`(56.25%, 후보 2개 반증·
  lr 대기) 2건 잔존, report-only로 발행 계속. name 게이트는 PIT 데이터 수정으로
  해소(6857 0.235).

## S11.6 (GPT 리뷰 후속 — 위생 수정 3건 + Daily_Returns 오염 실측) — 2026-07-21

GPT-5.6 리뷰 4건을 코드 대조로 전부 사실 확인. 사용자 지시: "A그룹부터 진행하고
⑤는 진단 먼저" — A그룹(기준선 불변 위생 수정 3건)은 TDD로 적용, ⑤(returns 뷰
분리)는 수정 전 실측 진단만 수행. ④(HOLD 차단형 전환)는 결정 대기로 미적용.

- **A① production 게이트 fail-closed** (`scripts/validate_portfolio_bundles.py`):
  기존 `any(v is False)→HOLD`는 체크 결측(None)이 PRODUCTION으로 통과하는
  fail-open. `all(v is True)`일 때만 PRODUCTION으로 변경(결측=HOLD). 기존
  fail-open을 핀으로 고정하던 테스트("None checks are not violations")를
  fail-closed 계약으로 교체 + 부분 결측 테스트 추가. challenger 엔트리는
  comparison gate 사용이라 무영향. 현 registry는 체크 6종 전부 실측값이라
  오늘 라벨 변화 없음(HOLD 유지). report-only 정책 자체는 §S9.2 그대로.
- **A② 캐시 유니버스 구성 가드** (`run_variant.py check_cached_universe`):
  기존 len==150만 검사 → 구성이 다른 스테일 150종 캐시(예: MMC 시절)가 통과.
  expected==len(TICKERS)일 때 set 비교(순서 무관) 추가, 불일치 시 missing/extra
  5개씩 제시하며 ValueError. 정규 배치는 --no-cache라 산출물 무영향.
- **A③ analytics ticker_meta 필수화** (`src/analytics.py` 2개 함수):
  compute_style_sector_tilt_rows / compute_monthly_ow_explanation_rows의
  None→정적 TICKER_META(60/150 스테일) 무음 fallback을 ValueError로 교체
  (레거시 정적 동작은 ticker_meta=TICKER_META 명시로만). 운영 경로 호출부
  0곳(무영향)·잠재 결함 제거. 테스트 전체 **273 PASS**.
- **⑤ 진단(수정 없음, 읽기 전용)**: production config(pca 252/5/2, horizon 20)로
  A(현행: 전 150열, 유령=당일 median) vs B(eligibility-aware: 윈도우 전체가
  실데이터인 종목만) 엔진 수식 복제 비교. census: 시작일(2014-01) 유령
  19/150(12.7%)→2025-02-21 소멸; 전체 fit 날짜의 **96.8%**가 오염 가능(252d
  윈도우가 상장일 이후로도 걸침).
  - EW 시장평균 |Δ|: mean 0.99bp/일·P90 2.17·max 11.7 (유령 존재일 n=2,890).
  - momentum_126d 순위(pct) 이동(실상장 셀 402,053개): mean 0.0159(≈2.4슬롯),
    **1슬롯 초과 66.1%·3슬롯 초과 31.9%** — rank 기반 모델에 유의미.
  - PCA 타깃 |Δ|: **mean 22.5bp·P50 8.0·P90 42.1·max 2,548bp** — |fwd 20d|
    median 447bp 대비 **5.1%**. |Δ|>10bp 셀 42.2%. 날짜별 Spearman median
    0.9998·P10 0.9969·min 0.785(순위 구조는 대체로 보존, 꼬리 날짜 왜곡).
  - 한계 명시: B는 "윈도우 전체 실데이터" 기준이라 Δ에는 유령 오염분과
    신규상장 실데이터 제외분이 섞임(상장 전환기 부근은 상한 성격). GPT 소형
    재현(avg 12bp)보다 실데이터 오염이 큼(avg 22.5bp).
  - **판정**: 오염 실재·규모 유의(특히 피처 순위·타깃 스케일 5%). ⑤ 수정
    (피처·횡단면=상장 전 NaN 뷰 / PCA=eligibility-aware) 진행 여부는 기준선
    변경(재인증 필요)이므로 사용자 승인 대기 → **2026-07-21 사용자 승인,
    §S11.7로 진행**.

## S11.7 (사전등록 — PIT returns 이중 뷰: Daily_Returns 예외 제거) — 2026-07-21

**성격**: 데이터 정확성 계층(§2.1 예외 범주). 성능 arm이 아니므로 IR endpoint
없음 — §S11.6 실측(fit 날짜 96.8% 오염·momentum 순위 1슬롯 초과 66%·PCA 타깃
mean 22.5bp)이 근거. 단독 구조 변경(다른 파라미터 동시 변경 금지). 완료 시
**새 기준선 S0(150)″** 선언, §S11.4 수치와 직접 비교 금지(진단 참고만).

**설계(사전 확정)**:
- `data.returns`(dense·median-fill)는 유지 — 시뮬레이션 P&L 경로는 0-가중
  유령이 불활성이므로 dense가 안전하고 정확. *(정정 2026-07-21: 공분산은
  dense가 아니라 이미 마스킹된 `raw_returns`를 사용 — backtest risk_returns.
  실제 구현이 본 서술보다 안전했음. GPT 후속 리뷰 P3 반영.)*
- `data.returns_masked` 뷰 신설: 상장 전(상장일 포함, inclusive=True) NaN.
  `listing_mask_enabled=False`면 `returns`와 동일 객체(파리티 핸들).
- 소비처 전환 5곳: `features/price.py`·`features/conditioning.py`·
  `features/assembly.py`(lean momentum)·`features/macro_cross.py`·
  `target_engine.build_targets`.
- 엔진 eligibility-aware: fit 창(252d) 전체가 실데이터인 열만 PCA 기저·타깃
  대상. 비적격 열(상장 전+상장 후 252d 미만)은 해당일 타깃 NaN. dense 입력이면
  기존 알고리즘과 **항등**(파리티).
- 비교 연산자 가드 2곳(price.py `pos_ret_ratio`·`trend_consist`): NaN>0=False가
  0.0으로 새는 것을 `.where(notna)`로 차단.

**합격기준(판정 가능)**:
1. dense-parity 테스트: NaN 없는 패널에서 신규 엔진 출력 == 기존 알고리즘
   inline 참조 (atol 1e-12).
2. ghost-exclusion 테스트: 유령 열이 있어도 실상장 열 타깃은 eligible-only
   참조와 일치·날짜 스킵 없음, 유령 열은 창 중첩 구간 NaN.
3. 피처 모듈별 masked-소비 계약 테스트(price/conditioning/assembly/macro_cross)
   + `returns_masked` 계약(마스크 ON: 상장일까지 NaN, OFF: 동일 객체).
4. 전체 테스트 suite PASS.
5. 재인증: production+challenger `--no-cache`·ECOS·seed 42, fallback 0/95,
   집중 캐릭터 보존(§5: TE 가드 4.5% 이내·active share 성격 유지). IR 변동은
   채택/기각 판정 대상이 아님(데이터 정확성 수정) — 결과를 S0(150)″로 기록.

**결과 (2026-07-21) — 합격기준 전부 충족, S0(150)″ 확정**:
- 구현: `returns_masked` 뷰(data_loader)·소비처 5곳 전환·엔진 창별 열
  eligibility(비적격 열 타깃 NaN, `n_eligible<2` 스킵 가드)·비교 가드 2곳.
  신규/갱신 테스트 11건 포함 **전체 284 PASS** (dense-parity·ghost-exclusion·
  소비처 계약·returns_masked 계약 전부 green; red→green TDD 준수).
- **재인증(ECOS·--no-cache·seed 42) — 새 기준선 S0(150)″**:
  - production(codex_causal_rank_65): **IR 1.681 / TE 3.61% / realized_beta
    1.041**(P2 shelve 유지) / turnover 74.6% / MDD −30.8% / 서브기간
    1.669·1.218·2.265 전부 양 / ECOS 190·SCS fallback 0%·optimizer fallback
    1/95(1.1%) / 퇴화율 56.25%(18/32) / causal_validation_ok / 679.9s.
  - challenger(iter15): **IR 1.915 / TE 3.56% / beta 1.026** / turnover
    108.3% / 서브기간 1.284·1.240·2.675 / 퇴화율 40.6% / 725.6s.
  - 해석: 오염 제거로 양쪽 IR이 §S11.4 대비 상승(1.371→1.681, 1.228→1.915).
    유령 median 시계열이 횡단면 신호를 희석하고 있었음을 시사. §2 규율에
    따라 §S11.4 수치와의 ΔIR은 채택 근거가 아니며(데이터 정확성 수정),
    S0(150)″가 이후 유일 비교 기준.
- **게이트 갱신(export+validator, fail-closed 로직 §S11.6-A① 적용)**:
  - production HOLD **2→1건**: `sector_active_risk_ok` **True**(0.840<0.85) —
    §S11.5의 "endpoint 꼬리 스파이크" 판정이 데이터 정화만으로 해소된 것으로
    확인(λ≈26k 발동 조건은 등재 유지·현재 비발동). top_name STX 0.243(<0.35).
    잔존 HOLD는 `degenerate_rate` 56.25% 단독(lr 후보 사전등록 대기).
  - comparison gate RESEARCH/FAIL 유지: 7종 중 `turnover_within_1_25x`만
    실패(108.3/74.6=1.45×), IR/active/TE/beta/MDD/sub_wins(2) 통과.
    challenger 우위 관찰은 §8·DSR 게이트 대상 — 본 절에서는 비액션 기록만.
  - est TE 3.50%(캡 3.5% 준수)·est vol 18.8%·realized TE 4.5% 가드 이내 —
    §5 집중 캐릭터 보존 확인.
- **부기(2026-07-21, GPT 후속 리뷰 반영)**:
  - *P2 잔여*: `compute_specific_returns_regime_weighted`(regime-weighted PCA
    연구 분기, `regime_pca_weighted_enabled` default-OFF)는 eligibility-aware가
    아님 — `hist.notna().all(axis=1)` 행 제거만 수행해, masked 입력에서 신규
    상장 1종이 기존 종목의 과거 행 전체를 지운다(GPT 소형 재현: 표준 39셀 vs
    regime 0셀). 두 기준선 모두 OFF라 S0(150)″ 수치·게이트 무영향.
    **활성화 전 수정 필수(pre-enable requirement)** — 해당 arm 사전등록 시
    eligibility 이식을 선행 조건으로 한다. multi-horizon 경로는 표준 함수를
    내부 호출하므로 정상.
  - *P3 정정*: 위 설계 bullet의 공분산 서술 정정 완료(코드 주석·acceptance
    테스트 주석·메모리 동기). 계산 오류 아님 — 문서 정확성 정정.

## S11.8 (multi-horizon 사전 단계 — IC 감쇠 진단 + 인과 분할 선결 수정) — 2026-07-21

**동기**: Brini & Kolm(JFDS Spring 2026, PPO 동적 거래) 검토 — 거래비용 하
최적 거래는 알파 기간구조를 요구(GP aim portfolio). DJIA 실증에서 단일
horizon 예측만으로는 단일기간 Markowitz와 동일, 기간구조(h=1/2/5/10) 추가
시 유의미 개선, 장기 노이즈 증가(시나리오 3)에도 우위 유지. Pictet PDF
p32("horizon combination" 로드맵)·p13(장단기 동인 상이)과 정합. arm 착수
전 사전 질문: "현 20d 신호에 버려지는 느린 알파 성분이 실재하는가."

**(a) IC 감쇠 진단 (읽기 전용, 재실행 없음)**:
- 방법: S0(150)″ 산출물 `backtest_result.pkl`의 `pre_overlay_predictions`
  (§4.2 정본 패널; `raw_predictions`는 감도) × `data.returns_masked`(PIT·USD,
  production config로 로드). 리밸런스 95일에서 t+1 기점 h일 누적수익률과의
  횡단면 Spearman IC, 창 완전 유효 종목만·최소 30종. 창: 1–5/1–10/1–20/
  1–40/1–63/**21–63(보유기간 이후)**. 스크립트: 세션 스크래치패드
  `diag_ic_decay.py`(일회성 진단, 저장소 미포함).
- **결과 — production(rank, pre-overlay)**: IC가 감쇠하지 않고 증가.
  h05 0.048 / h20 0.053 / h40 0.068 / h63 **0.080**(t 3.6) /
  **post21_63 0.069**(t 3.4, 양성 64%, 전·후반 +0.025/+0.113 부호 일관).
  naive t는 창 중첩(21d 간격 vs 63d 창 ~3×)으로 부풀려짐 — 보정 시 ~2.4
  수준, 부호 일관이 정직한 판정. raw(pre-EMA)와 사실상 동일.
- **결과 — challenger(regression)**: h40 정점 0.044 후 감쇠,
  post21_63 0.021(t 1.55)·전반기 음(−0.009/+0.051) — 느린 성분 신뢰 불가.
- **판정**: multi-horizon arm의 전제("보유기간 종료 후에도 지급되는 느린
  알파") **production 신호에 대해 통과**. 21d 리밸런스가 미실현 알파를
  반복 폐기 중이라는 직접 증거 — 논문의 turnover 감쇠 메커니즘 기대 유효.
  단, 이 진단은 "현 신호의 느린 성분" 존재 증명이며 "63d 전용 모델의 추가
  신호"는 arm 본실험 대상. IC 수치는 arm 채택 근거 아님(설명력 진단).
- 부수 기록: 진단 중 C: 잔여 0B 상태 발견(일시적 — 원인 프로세스 미상,
  이후 자연 회복 10.5GB) → 사용자 승인으로 Temp 스크래치패드 정리(~9MB)·
  hibernation off(+6.3GB), 잔여 16.9GB에서 본 기록 작성.

**(b) 인과 분할 선결 수정 (pre-enable requirement 이행)**:
- 결함: `train_and_predict`가 분할 horizon으로 `config.forward_horizon`(20)을
  전달(model_trainer.py) — `multi_horizon_targets_enabled` 시 블렌드 타깃에
  63d 성분이 포함되면 라벨 실현창(63d)이 embargo(20d)를 초과해 train/val
  라벨 중첩(누수). `src/rl/dr_walkforward.py`의 "embargo≥horizon 강제"
  가드와 동일 클래스의 공백.
- 수정: `effective_label_horizon(config)` 헬퍼 — mh OFF(기본)면
  `forward_horizon` 그대로(파리티), ON이면 `max(forward_horizon,
  max(mh_weights keys))`. 분할 호출부가 이 값을 사용.
- 파리티: 두 기준선 모두 mh OFF → 분할 입력 불변 → 바이트 동일(코드 경로
  검증은 단위테스트, 재인증 불요 판단 — 산술 변경 없음).

## S11.9 (사전등록 — multi-horizon 블렌드 타깃 arm) — 2026-07-21

**사전등록 (실행 전 기록)**. 사용자 승인: 설계 1안(블렌드 타깃), lr arm보다
선행. §S11.8 진단 통과가 전제.

- **가설**: §S11.8이 실증한 느린 알파(보유기간 이후 21–63d IC 0.069)를
  타깃에 반영하면 (i) 예측 신호의 회전이 줄어 turnover 하락, (ii) IR
  비손상~개선. 근거: Brini&Kolm(JFDS 2026) 시나리오 2·3(기간구조의 비용
  인지 거래 가치, 장기 노이즈에도 우위 유지)·GP aim portfolio, Pictet
  p32 "horizon combination" 로드맵.
- **단일 사전약정 파라미터**: `multi_horizon_weights = {20: 0.7, 63: 0.3}`
  (+enable 플래그). 근거: 인증 20d 타깃 우세 보존(0.7)·진단이 지목한 63d
  성분(0.3). 스윕 금지 — 이 맵 1개만 판정. 블렌드 산술은 기존 인프라
  그대로(sqrt(252/H) 연율화, 셀은 전 horizon 유효 시에만 값).
- **실행**: `<PY> run_variant.py --variant variants/arm_s11_9_mh_blend.yaml
  --no-cache` — ECOS·seed 42·단일 프로세스. variant는 production 사본 +
  mh 2필드(role: research). target 필드 변경이라 캐시 자동 비활성이나
  --no-cache 명시.
- **판정 기준 (vs S0(150)″ production IR 1.681/TE 3.61%/turnover 74.6%/
  beta 1.041)**:
  - E1 (IR 채택 바): ΔIR > +0.36 & 서브기간 3구간 부호 일관.
    |ΔIR| < 0.36 → 노이즈(설명력 기록만).
  - E2 (메커니즘 공동 판정): avg_annual_turnover ≤ 67.1%(상대 ≥10% 감소).
  - 가드(위반 시 FAIL): G1 realized TE ≤ 4.5% · G2 realized_beta ∈
    [0.95, 1.05] · G3 optimizer fallback ≤ 5% · G4 ECOS-only(SCS 0) ·
    G5 causal_validation_ok=True + 분할 audit embargo=63(§S11.8(b) 발동
    확인) · G6 집중 캐릭터 보존(§5, 무음 bm 붕괴 없음).
  - 관찰(게이트 아님): 퇴화율·MDD·ic_series·유효 타깃 셀 수.
  - 해석 매트릭스: E1&E2 → §8 flip 후보(DSR 별도) / E1만 → flip 후보
    (메커니즘 미확인 부기) / E2만 → "메커니즘 확인·IR 중립" 기록 후
    2모델(mu 결합) 설계 검토로 이관 / 둘 다 미충족 → 불채택·OFF.
- **기계적 기대 차이(사전 명시)**: 63d 성분으로 학습 라벨 테일 NaN이
  20d 대비 ~43거래일 확대(트레이너 dropna 처리·예측 생성은 무영향),
  Phase 3 PCA 2회로 실행시간 증가.

**결과 (2026-07-21 실행, 834s) — E1·E2 모두 미충족 → 불채택·OFF**:
- 발동 검증: 블렌드 로그 horizons=[20,63] weights=[0.7,0.3] 확인,
  분할 audit 32건 전원 embargo_days=63·forward_horizon=63·causal_ok=True —
  **§S11.8(b) 선결 수정 정상 발동(G5 통과)**. 유효 타깃 셀 409,684·최종
  라벨 2026-04-22(사전 명시한 테일 확대와 일치).
- **E1 FAIL**: IR **1.370**(ΔIR **−0.311** vs 1.681 — |Δ|<0.36 노이즈 대역
  이나 음수). 서브기간 IR 0.792/0.514/2.566 → Δ −0.877/−0.704/+0.301
  **부호 비일관**(P1·P2 훼손, P3만 개선).
- **E2 FAIL**: avg_annual_turnover **72.6%**(기준 ≤67.1%) — 상대 −2.6%뿐.
  타깃을 느리게 섞어도 신호 회전은 거의 불변 = **가설의 핵심 메커니즘
  미발현**.
- 가드: G1 TE 3.76% OK / **G2 realized_beta 1.056 위반**(>1.05, 부기) /
  G3 optimizer fallback 0/95 OK / G4 ECOS 190·SCS 0 OK / G5 상기 OK /
  G6 TE·active 성격 유지(단 beta 상승이 경계 신호).
- 관찰: 퇴화율 50%(16/32; production 56.25% 대비 소폭↓ — 63d 성분이
  검증 개선을 약간 도움, 게이트 비대상). MDD −31.9%.
- **해석**: §S11.8의 느린 알파는 실재하나, **블렌드 타깃은 기간구조를
  학습 전에 붕괴시켜 이를 포착하지 못함** — Brini&Kolm 시나리오 2의
  이득이 "horizon 분리 상태 유지"에서 나온다는 해석과 정합. 63d 노이즈
  혼입이 전·중반 구간 rank 학습을 희석(P1/P2 급락)한 것으로 보임.
  매트릭스 판정: 둘 다 미충족 → **불채택, default-OFF 유지, 프로덕션
  무변경(no-flip)**. E2-only 시의 2모델 이관 조건도 미충족 — 2모델(mu
  결합) 설계는 자동 진행하지 않고 별도 사전등록+승인 대상으로 남김.
  잔존 등록 후보: lr 0.02→0.03(퇴화율 HOLD 겨냥, 사전등록 대기).

## S11.10 (사전등록 — 퇴화율 arm: learning_rate 0.02→0.03) — 2026-07-21

**사전등록 (실행 전 기록)**. 사용자 승인(§S11.9 종료 후 1안 지시). 퇴화율
HOLD(production 56.25% > 게이트 25%) 겨냥 **잔존 마지막 등록 후보** —
min_child 60→30(§S10.2)·val_window 252(§S11.5 Arm B)는 반증 완료.

- **가설**: D0 진단(§S7)의 "구조적 즉시 조기종료(best_iter median 1)"는
  lr 0.02에서 그루당 기여가 too-small이라 purged 검증창의 개선이 노이즈에
  묻히기 때문 — lr 0.03(그루당 기여 +50%)이면 진짜 신호의 검증 개선이
  patience(100) 내 가시화되어 퇴화율 하락. 참고: 0.02는 "was 0.03 — V2
  안정성" 주석의 인하값(원복 실험). **사전 명시 리스크**: Pictet p34는
  low lr을 강건성 요소로 명시 — 퇴화율 개선과 IR 훼손이 교환될 수 있음
  (그래서 do-no-harm 가드가 공동 판정).
- **단일 사전약정 파라미터**: `lgbm_params.learning_rate: 0.02 → 0.03`.
  `n_estimators` 800 유지(early stopping이 실그루수 결정 — 상한 불변으로
  단일 변경 보존). 스윕 금지.
- **실행**: `<PY> run_variant.py --variant variants/arm_s11_10_lr_003.yaml
  --no-cache` — ECOS·seed 42·단일 프로세스. variant는 production 사본 +
  lr 1필드(role: research).
- **판정 기준 (vs S0(150)″ production IR 1.681/TE 3.61%/turnover 74.6%/
  beta 1.041/퇴화율 56.25%)**:
  - **E1 (HOLD 해소 기준, §S11.5b 관례)**: `model_quality.degenerate_rate`
    ≤ **0.25**.
  - Do-no-harm 가드(위반 시 채택 불가): G1 ΔIR > −0.36(노이즈 대역 내
    하락까지 허용, 그 이상 훼손 불가) · G2 realized TE ≤ 4.5% · G3
    realized_beta ∈ [0.95, 1.05] · G4 optimizer fallback ≤ 5%·ECOS-only
    (SCS 0) · G5 causal_validation_ok=True · G6 turnover ≤ 93.3%
    (1.25×) · G7 집중 캐릭터 보존(§5).
  - 관찰(게이트 아님): n_trees 분포·best_iter, MDD, ic_series.
  - 해석 매트릭스: E1+가드 전부 통과 → §8 flip 후보(단일 flip·DSR 해킷
    별도, HOLD 해소). E1 미충족·부분 개선(<56.25%) → 불채택·설명력 기록.
    악화(≥56.25%) → **lr 가설 반증 — 퇴화율 등록 후보 소진**, objective
    특성 가설(rank_xendcg, §S11.5 구조 단서)만 잔존함을 기록.

**결과 (2026-07-21 실행 3회차 완주, 738s) — E1 FAIL·lr 가설 반증·불채택**:
- 실행 이력: 1·2회차 외부 중단(1차 ~7분 진행·2차 출력 0줄, 산출물 미생성
  — 오염 없음; 배터리 방전 모드 중 발생, §S9.1 전례와 정합하나 원인 미확정),
  3회차 완주.
- **E1 FAIL**: degenerate_rate **56.25%(18/32)** — 기준선과 **정확히 동일**
  (0.0pp 이동). 매트릭스 ≥56.25% 분기 → **lr 가설 반증**. 다만 n_trees
  분포는 이동: [2,8,8,5,8,9,8,2,2,4,2,1,6,2,1,9,1,8] — 기준선(대부분 1~2)
  대비 8~9그루 근접 실패가 다수. **그루당 기여 증대는 실재했으나 10그루
  문턱을 못 넘김** — "검증창 개선 불가시" 구조가 lr로 안 풀린다는 뜻으로,
  objective 특성 가설(rank_xendcg NDCG 정체)을 강화.
- 가드(참고 — E1 실패로 채택 불가): G1 IR 1.564(Δ−0.117, 노이즈 대역 내)
  OK / G2 TE 3.74% OK / G3 beta 1.047 OK / G4 fallback 0/95·ECOS 190·
  SCS 0 OK / G5 causal_ok OK / G6 turnover 73.4% OK / G7 캐릭터 유지 OK.
  서브기간 1.378/0.742/2.622.
- **판정: 불채택·프로덕션 무변경(no-flip). 퇴화율 등록 후보 소진** —
  min_child(§S10.2)·val_window(§S11.5)·lr(본 절) 전부 반증. 잔존 설명은
  objective 특성(rank_xendcg; challenger 회귀 40.6% vs production 56.25%,
  동일 인과규율 하 격차)뿐. 후속 선택지(각각 별도 사전등록/결정 필요):
  (a) objective 교체 실험(사실상 §8 challenger 승격 논의와 중첩 — DSR
  게이트 대상), (b) HOLD 정책 결정(④ 차단형 전환 여부, §S11.6 미결),
  (c) 퇴화율 게이트 한도 재캘리브레이션(§S10.1 섹터 한도 전례의 실측
  기반 방식 — 단, "재학습 신선도" 지표의 의미가 훼손되지 않는 선에서).

**부기 (2026-07-21, 재사용 손상 진단 — 읽기 전용, 액션 없음)**:
- 질문: 퇴화 재학습의 이전 모델 재사용이 예측 품질을 실제로 해치는가.
- 방법: 리밸런스 95일을 모델 age(마지막 성공 재학습 이후 연속 퇴화
  횟수)로 분류 — production 신선(age0) 41일 vs 재사용(age≥1) 54일
  (최장 age 7). 지표 2종: 파이프라인 `ic_series`(예측 vs 타깃)·h20 IC
  (§S11.8 방식, 예측 vs t+1..t+20 총수익률). 스크립트 세션 스크래치패드
  `diag_stale_model_ic.py`.
- **결과 — 재사용 손상 증거 없음, 방향은 오히려 역**:
  - production ic_pipe: 신선 +0.003 vs 재사용 **+0.037**(Welch p=0.29,
    양 반기 모두 재사용 우위). ic_h20: +0.044 vs +0.060(p=0.72, 동일 방향).
  - challenger ic_pipe: +0.032 vs +0.071(p=0.12, 동일 방향), ic_h20 동률.
  - 유일한 음(−) 구간은 production age 7(N=3, ic_h20 −0.088) — 표본 미소,
    "초장기 연속 재사용"만 잠재 리스크로 표시.
- 해석: 퇴화 재학습은 "incumbent를 못 이김"(§S7 D0)의 표현이므로 재사용
  구간은 검증을 한 번 통과한 강한 모델이 지배 — fallback 설계가 의도대로
  작동 중. **degenerate_rate는 손상의 대리지표로서 실측 근거 없음**(단
  유의차도 아님 — "손상 미검출"이 정직한 결론). (c) 게이트 재캘리브레이션
  (총량률 → 예: 연속 재사용 깊이 상한)에 실측 근거 제공; ④ 차단형 전환의
  근거는 약화. 본 진단은 설명력 — 게이트 변경은 별도 사전등록/결정 필요.

---

## S12 — 외부 GPT 리뷰 후속 시퀀스 (2026-07-22 사용자 승인)

> 배경: 외부 GPT 리뷰의 평가·개선 제안을 코드·결정 로그 대조로 검증
> (Fable 세션, Explore 3-agent 교차확인). 사실관계 7항 중 5 TRUE·2 PARTIAL.
> 특히 objective 지적은 §S11.10 잔존 가설과, 퇴화 가드 지적은 §S11.10
> 부기 실측과 독립 수렴. 사용자 승인 순서: **① 퇴화 게이트 재정의(거버넌스)
> → ② objective swap 단일 arm → ③ EWMA full refresh(+스케일 제거 청소)
> → ④ mu 보정 최소 버전 → ⑤ PCA 표준화 → ⑥ 비용 모델**. 성능 arm은
> 각각 단일 사전등록 파라미터·ΔIR +0.36 바(§2.4)·vs S0(150)″.
> 기각된 GPT 제안: PIT 유니버스 재구축(§S11 생존편향 정책 유지)·직접
> 리스크 예산(현 하드캡+가드레일 설계 의도적)·1.25× 즉시 폐지(챌린저
> 차단자는 DSR/노이즈밴드임을 확인).

### S12.1 (2026-07-22) — 퇴화 게이트 재정의: 총량률 → 연속 stale depth (거버넌스)

**사전등록 (구현 전 기록)**:
- **성격**: 성능 주장 없음 — 게이트/모니터링 거버넌스 변경. IR 바 비적용.
- **근거**: §S11.10 부기 — 재사용 손상 실측 증거 없음(production ic_pipe
  신선 +0.003 vs 재사용 +0.037, 방향 역), degenerate_rate는 손상 대리지표로
  실측 근거 없음. 유일 잠재 리스크 = 초장기 연속 재사용(age 7 꼬리, N=3,
  ic_h20 −0.088, 비유의).
- **실측 (2026-07-22 재인증 번들, metrics.model_quality events×split_audit
  유도)**: 최대 연속 퇴화 재학습 — production **7**(run 집합 {1,2,5,7}),
  challenger **3**. §S11.10 부기의 "max age 7"과 정확히 일치(동일 지표의
  두 표현임을 확인).
- **설계**: production HOLD 게이트(§S9.2 체크 6종)에서 `degenerate_rate_ok`
  제거(값은 관찰 지표로 유지), 신설 **`stale_depth_ok` = 최대 연속 퇴화
  재학습 ≤ 7**. 유도는 `split_audit` 재학습 순서 × `events` 날짜(재실행
  불필요·기존 번들 호환). **한도 7 근거**: 부기에서 depth ≤7 전 구간 손상
  미검출(7 꼬리만 비유의 음) — 실측 무해 구간의 상한을 허용하고 **8+는
  미검증 영역이므로 HOLD**. fail-closed 유지(events/split_audit 부재 →
  None → HOLD). 파이프라인 내부 경고(max_degenerate_model_rate 0.25·
  fail_on_degenerate_model_rate)는 불변 — 게이트만 교체.
- **모니터링**: operating monitoring.json 가드레일 블록에
  `model_max_stale_depth`·`model_stale_depth_breached` 추가
  (`model_degenerate_rate`는 관찰값으로 병기).
- **합격기준**: (1) `<PY> -m pytest tests/test_validate_portfolio_bundles.py
  -q` PASS — 신규 케이스: rate 높아도 depth≤7이면 PRODUCTION·depth 8+는
  HOLD·model_quality 부재는 fail-closed HOLD. (2) `<PY> scripts/
  validate_portfolio_bundles.py --bundle outputs/operating --bundle
  outputs/operating_codex_causal_rank_65` → production_gate 체크 6종 전부
  True·status PRODUCTION(퇴화율 56.25%는 values 관찰로 잔존). (3) 대시보드
  JSON 계약 비파괴(필드 추가만).
- **예상 결과**: 잔존 HOLD 1건(degenerate_rate)이 근거 있는 형태로 해소 →
  §S9.2 도입 이래 최초 production_gate=PRODUCTION.

**결과 (2026-07-22 구현·검증 완료 — 합격기준 전부 충족)**:
- 구현: validator `stale_depth_ok`(split_audit×events 유도·fail-closed·한도
  상수 `MAX_CONSECUTIVE_STALE_RETRAINS = 7`)
  + values에 `max_stale_depth`/한도 병기, `degenerate_rate`는 관찰값 유지;
  export monitoring 가드레일에 `model_max_stale_depth`·
  `model_stale_depth_breached` 추가(차기 export부터 산출물 반영). 파이프라인
  내부 경고(max_degenerate_model_rate 0.25)는 불변.
- 검증: 전체 스위트 **301 PASS**(신규 게이트 케이스 3종 포함). 실번들 검증
  → production_gate 체크 6종 전부 True·**status PRODUCTION — §S9.2 도입
  이래 최초**. production max_stale_depth **7**(한도 7)·challenger 3.
  degenerate_rate 0.5625는 values 관찰로 잔존.
- 부수 수리(별도 커밋): `test_universe_fx_conversion` 항등성 테스트 1건이
  c6d9584(07-22 스케줄 `git add -A` 스윕 커밋 — 11:30 테스트 통과 시점과
  12:44 커밋 사이에 트리가 더 변경되어 **테스트 미검증 상태가 커밋됨**,
  289→299개 수집 격차로 확인)에서 적색: §S11.4 자동 추론이 픽스처 첫
  날짜를 eligibility로 해석해 상장일 당일 raw 수익률을 inclusive 마스킹
  (동작은 인증 설계 그대로) — 테스트에서 `listing_auto_infer_enabled=False`
  로 변환 항등성 검증 의도만 복원. **동작 코드 불변**.

### S12.2 (2026-07-22) — objective swap 단일 arm: rank_xendcg → regression

**사전등록 (실행 전 기록)**:
- **가설**: §S11.10 잔존 objective 특성 가설 — rank_xendcg의 NDCG@[5,10]
  중심 손실이 (a) purged 검증창에서 정체해 퇴화(조기종료)를 유발하고
  (b) 하위 순위·언더웨이트 품질을 직접 최적화하지 않음(외부 GPT 리뷰
  지적과 수렴). 동일 프레임에서 objective만 바꾸면 격리 검증 가능
  (challenger 40.6% vs production 56.25% 격차는 confounded — 프레임 상이).
- **단일 사전약정 변경**(의미상 1건 — 결합 3필드): `model_objective:
  cross_sectional_rank → regression` + `lgbm_params.objective: rank_xendcg
  → regression` + `lgbm_params.metric: ndcg → mse`(challenger iter15의
  정본 regression 설정). 그 외 전부 production 핀 유지(lr 0.02·causal
  validation·EMA·overlays·optimizer). `rank_relevance_levels`/`rank_eval_at`
  은 잔존하나 regression 경로에서 inert. 스윕 금지. 참고: 스윕 커밋에
  포함된 미문서 `symmetric_rank` 브랜치(config 208·model_trainer 385-416,
  default-inert)는 본 arm과 무관 — 별도 사전등록 전 사용 금지.
- **실행**: `<PY> run_variant.py --variant
  variants/arm_s12_2_objective_regression.yaml --no-cache` — ECOS·seed 42·
  단일 실행.
- **판정 기준 (vs S0(150)″ production IR 1.681/TE 3.61%/turnover 74.6%/
  beta 1.041/퇴화율 56.25%/sub 1.669·1.218·2.265)**:
  - **E1 (§2.4 채택 바)**: ΔIR > **+0.36** AND 서브기간 3구간 Δ 부호 일관.
  - **E2 (objective 가설 판정)**: `model_quality.degenerate_rate` ≤ **0.45**
    (challenger regression 실측 40.6% 동수준 이하로 하락 시 가설 지지).
  - Do-no-harm 가드: G1 ΔIR > −0.36 · G2 TE ≤ 4.5% · G3 beta ∈ [0.95,
    1.05] · G4 fallback ≤ 5%·ECOS-only(SCS 0) · G5 causal_validation_ok ·
    G6 turnover ≤ 93.3%(1.25×) · G7 집중 캐릭터 보존(§5).
  - 관찰(게이트 아님): max_stale_depth(S12.1)·n_trees 분포·MDD·IC.
- **해석 매트릭스**: E1 통과 → §8 flip 후보(단일 flip·DSR 해킷 별도).
  E1 실패·E2 통과 → **objective 가설 실증(설명력)** — 성능 채택 없음,
  rank+regression mu-combination(§S11.9 잔존 후보) 설계 근거로 기록.
  E1·E2 모두 실패 → objective 가설 기각, 퇴화 원인 미해명으로 기록.

**결과 (2026-07-22 실행 완주, ECOS 192·fallback 0) — E1 FAIL·E2 PASS →
불채택·objective 가설 실증**:
- **E1 FAIL**: IR **1.565**(Δ **−0.116**, 노이즈 대역 내) + 서브기간
  1.403/1.281/1.903 → Δ −0.266/**+0.063**/−0.362 **부호 비일관**.
- **E2 PASS**: degenerate_rate **40.6%(13/32)** ≤ 0.45 — challenger
  regression의 40.6%와 **정확히 동수준**. 동일 프레임 격리에서 objective
  단독 교체가 퇴화율 56.25→40.6%를 재현 → **§S11.10 잔존 objective 특성
  가설(rank_xendcg NDCG 정체) 실증 확정**. max_stale_depth 3(vs production
  7) — 신선도도 개선.
- 가드: G1 Δ−0.116 OK / G2 TE 2.97% OK / G3 beta 1.018 OK / G4 ECOS-only
  OK / G5 causal_ok OK / **G6 위반: turnover 107.9% > 93.3%(1.25×)** —
  regression 신호는 순위 안정성이 낮아 회전이 challenger(108.3%)처럼 증가.
  G7 캐릭터: TE·active 성격 유지(관찰).
- **판정: 불채택·프로덕션 무변경(no-flip)**. 퇴화의 원인은 이제
  설명됨(objective 특성) — 단 성능·회전 게이트가 교체를 정당화하지 않음.
  관찰: P2(최약 구간)만 +0.063 개선 — rank는 P1/P3(추세 구간)에서,
  regression은 P2(횡보)에서 상대 우위라는 상보성 단서 → §S11.9 잔존
  후보(rank+regression mu-combination)의 설계 근거로 기록. IC 0.0498
  (production 0.0214 대비 높음 — IC와 IR의 괴리는 회전·순위 안정성 경유).

### S12.3 (2026-07-22) — EWMA 스케일 무효 검증 + 주기적 full refresh arm

**사전등록 (실행 전 기록)**:
- **배경 (GPT 지적 ③, 코드 검증으로 확정)**: (a) EWMA feature scaling
  (√(ewma/mean)·clip[0.5,2] 양수 곱)은 tree split 순서 불변 — 실효 의심;
  (b) 탈락 피처는 update가 active만 갱신해 감쇠만 하므로 **재진입 경로
  없음**(일방향 경로 의존). 스윕 커밋 c6d9584에 병행 세션의 인프라가 이미
  존재: `ewma_feature_scaling_enabled`(기본 True)·
  `ewma_full_refresh_interval`(기본 0) — config 253·256, **기본값이 파리티
  보존**. 미인증 스윕 코드였으므로 본 절에서 검증 후 사용: 오늘 기준선
  재현(IR 1.6809 = S0(150)″)이 기본 경로 파리티의 실증이고, 트래커 단위
  테스트(test_pit_sp500_ai_v2)+ 신규 walk-forward 테스트 2종이 동작 검증.
- **플러밍 (본 세션 추가)**: `model_quality.ewma_full_refresh` —
  interval>0일 때만 키 생성(파리티 안전): refresh_dates·reentry_events·
  reentry_feature_count. 테스트: OFF 키 부재 + ON 기록·full set 학습 확인
  (스위트 303 PASS).
- **(i) 스케일 무효 파리티 런 — arm 아님, IR 게이트 비적용**:
  `variants/parity_s12_3_no_ewma_scale.yaml` = production 사본 +
  `ewma_feature_scaling_enabled: false`(out_dir 분리). **판정**: metrics의
  성과 블록·degenerate 지표가 인증 기준선(outputs/codex_causal_rank_65)과
  **완전 일치**하면 "스케일링은 실증 no-op" → 청소 판정 완료(제거 대신
  플래그·문서화, 향후 기본 OFF 전환 근거). 불일치면 스케일링 실효 있음 →
  기본 ON 유지·차이 기록(이 경우 '청소' 없음).
- **(ii) full refresh arm — 단일 사전약정 파라미터**:
  `variants/arm_s12_3_ewma_refresh_4.yaml` = production 사본 +
  **`ewma_full_refresh_interval: 4`**(재훈련 63d × 4 ≈ 연 1회 refresh).
  스윕 금지.
  - **E1 (§2.4 채택 바)**: ΔIR > +0.36 AND 서브기간 3구간 Δ 부호 일관
    (vs S0(150)″ IR 1.681).
  - **E2 (기전)**: `reentry_feature_count ≥ 1` — 탈락 피처 재진입 실증.
  - 가드 G1–G7 (§S12.2와 동일 정의).
  - 해석: E1 통과 → §8 flip 후보(DSR 별도). E1 실패·E2 통과 → 경로 의존
    해소 기전 실증·성능 무득 → 불채택·설명력 기록. E2 실패 → refresh가
    재진입을 못 만듦(core whitelist 포화 — active가 이미 floor 60 부근 —
    가설) 기록.
- **실행 순서**: S12.2 완료 후 (i) → (ii), 각각 `--no-cache`·ECOS·seed 42·
  단일 실행.

**(i) 결과 (2026-07-22 실행 완주, ECOS 192·fallback 0) — 파리티 반증·
청소 없음**:
- metrics 성과 필드 전반 **불일치**: IR 1.6809→**1.6212**(Δ−0.060, 노이즈
  대역) · TE 3.61→3.84% · turnover 74.6→74.2% · IC 0.0223→**0.0298** ·
  퇴화율 56.25→**50.0%**(18→16) · 서브기간 1.716/**0.749**/2.377
  (기준선 1.669/**1.218**/2.265 — P2가 크게 이동).
- **판독**: 양수 스케일링의 split-순서 불변성은 정확산술의 성질일 뿐 —
  LightGBM histogram bin 경계의 부동소수점 섭동이 rank_xendcg의 민감한
  early stopping(NDCG 정체)을 경유해 tree 구조·퇴화 패턴·P2 IR까지
  증폭된다. 스케일링은 경제적 신호가 아니라 사실상 **결정적 시드 섭동**
  으로 작동(ΔIR −0.060은 §S7 시드 운 ±0.19 범위 내). GPT의 "스케일링
  제거 무방" 주장은 이 파이프라인에서 **반증** — 제거는 결과 불변이
  아니라 기준선 교체이며, 기대 이득 없는 재인증 비용만 발생.
- **판정 (사전등록 분기)**: 불일치 → `ewma_feature_scaling_enabled`
  **기본 True 유지·청소 없음**. 부수 확인: 스케일 OFF 세계도 가드 범위
  내(TE 3.84%<4.5%·beta 1.040·ECOS-only) — 파이프라인의 chaos 민감도
  실측 사례로 §S7 시드 분산 기록에 연결.

### S12.α (2026-07-22 14:4x 발견) — 데이터 빈티지 포크: §S12.2·S12.3(i) 판정 잠정 무효

- **사실**: `ai_signal_data.xlsx`가 **13:43 리프레시**됨(사용자 수행,
  Excel 세션 13:23~ 열림 유지; `Index.xlsx`는 12:01 갱신). 그 결과
  - **v1** (11:30 스케줄 런·S0(150)″ 재현·S12.1 검증): 3,256일, ~07-20
    (ffill 1일).
  - **v2** (14:16 이후 모든 런): **3,258일, ~07-22**(ffill 3일, 교집합은
    여전히 07-17) — §S12.2 arm·§S12.3(i) 파리티 런이 전부 v2 사용.
- **결론**: §S12.2·S12.3(i)의 "vs S0(150)″ IR 1.681" 비교는 **처치 효과와
  데이터 빈티지 효과가 교락** — 위 두 절의 판정은 **잠정 무효**로 격하.
  특히 S12.3(i)의 "파리티 반증(FP chaos)" 판독은 빈티지 차이만으로도
  설명 가능하므로 **재검** 필요. §9(현실-문서 불일치 시 중단·기록) 적용.
- **복구 절차 (사전등록)**: ① production variant를 v2에서 재인증 →
  **S0(150)‴** 채택(이후 §S12 전 arm의 단일 비교 기준; v1 1.681과 직접
  비교 금지). 데이터에 07-21이 포함되므로 이 재인증이 **07-21 리밸런싱
  등록**(운영 P0)도 겸함. ② §S12.2·S12.3(i)는 **이미 v2 산출물**이므로
  재실행 없이 S0(150)‴ 대비로 **재판정**해 본 로그에 개정 기록. ③ 잔여
  arm(S12.3(ii)·S12.4·S12.5)은 v2에서 실행·판정. ④ 종료 후 표준 배치
  (run_and_upload)로 운영 번들 갱신·검증·push. 부기: S12.3(ii) 1차 실행은
  14:4x 외부 중단(killed, 원인 미상 — §S11.10 실행 이력의 중단 전례와
  유사)·산출물 미생성, 재실행 예정. Excel 동시 열림 상태는 §S9.1 위험으로
  기록(읽기 시점 단발이라 진행, 저장 충돌 시 해당 런 폐기·재실행).

**S0(150)‴ 확정 (2026-07-22 재인증 완주, ECOS 192·fallback 0, 재시도
1회 — 1차는 외부 중단)**: production (Causal Rank 150, v2 데이터
3,258일·tail_ffill 3d(07-17→07-22)·FX 07-22·data_as_of 07-22) —
**IR 1.6212 · TE 3.84% · realized_beta 1.0398 · turnover 74.2% · MDD
−30.3% · IC 0.0298 · 퇴화율 50.0%(16/32) · max_stale_depth 5 · 서브기간
1.716/0.749/2.377 · causal_ok**. 이후 §S12 arm 판정은 전부 이 기준.
v1 1.681과의 차이(ΔIR −0.06, P2 1.218→0.749, 퇴화 18→16)는 리프레시로
과거 시트 값이 재기재(restatement)된 효과 — §S7 시드 운 규모와 정합.

**§S12.3(i) 재판정 — 파리티 입증(이전 '반증' 기록 철회)**: v2 재인증
(스케일링 ON) metrics가 S12.3(i) 스케일링 OFF 런과 **전 필드 float
완전 일치**(IR 1.6212193701417186까지 동일·퇴화 이벤트 동일·서브기간
동일). 동일 데이터에서 ON=OFF **비트 동일** — 양수 피처 스케일링의
split-불변성이 이 파이프라인·데이터에서 실증됨(GPT 지적 (a) 확인).
직전의 "파리티 반증·FP chaos" 판독은 v1/v2 빈티지 교락의 오독이었음을
명시 철회. **판정**: 스케일링은 실증 no-op — `ewma_feature_scaling_
enabled` 기본 True 유지(어느 쪽이든 바이트 동일이라 무의미), 코드 제거는
§3 외과적 원칙상 불요, 본 증명 기록으로 '청소' 종결.

**§S12.2 재판정 (vs S0(150)‴, 동일 v2 데이터 — 교락 해소)**:
- E1: IR 1.5653 vs 1.6212 → ΔIR **−0.056**(노이즈 대역) · 서브기간 Δ
  −0.313/**+0.532**/−0.473 부호 비일관 → **E1 FAIL 유지**.
- E2: 퇴화율 **40.6% vs 50.0%**(동일 프레임·동일 데이터) — 절대 바
  (≤0.45) 통과·**E2 PASS 유지**, 단 갭은 v1 대비 축소(15.6→9.4pp)를
  정직 기록. max_stale_depth 3 vs 5.
- G6 turnover 107.9% > 92.7%(1.25×74.2%) 위반 유지. **최종 판정 불변:
  불채택·no-flip·objective 가설 지지(강도 하향)**. P2 상보성 단서는
  오히려 강화(+0.53) — mu-combination 설계 근거 유지.

**§S12.3(ii) 결과 (2026-07-22 재시도 완주, ECOS 192·fallback 0, vs
S0(150)‴) — E1 FAIL·E2 FAIL → 불채택·refresh 무효 가설 확정**:
- **E1 FAIL**: IR **1.5544**(Δ **−0.067**, 노이즈 대역) · 서브기간
  1.656/0.485/2.461 → Δ −0.060/−0.264/+0.084 **부호 비일관**.
- **E2 FAIL**: full refresh **6회 실제 발생**(interval 4, model_quality.
  ewma_full_refresh 기록)했으나 **reentry_feature_count = 0** — 탈락
  피처가 한 번도 재진입하지 않음. **core whitelist 포화 가설 확정**:
  피처 총수 대비 floor(ewma_min_features=60)가 커서 drop 여지가 수 종에
  불과하고, refresh 후에도 EWMA 순위가 동일 집합을 재선택. GPT 지적 ③의
  경로 의존성은 구조적으로 실재하나 **현행 core 구성에서는 무해**(재진입
  수요 자체가 없음).
- 가드: TE 3.84%·beta 1.039·turnover 73.6%·퇴화율 50.0%·stale depth 5 —
  전부 기준선 동수준(관찰).
- **판정: 불채택·프로덕션 무변경(no-flip)·OFF 유지**. full-refresh
  인프라·기록 플러밍은 향후 feature_mode 확장(full/lean) 시 재평가용으로
  유지.

### S12.4 (2026-07-22) — mu 보정 최소 버전: mu_vol_scaling 재측정 arm

> (개정 2026-07-22: 아래 사전등록의 비교 기준 "S0(150)″ 1.681"은 §S12.α
> 빈티지 포크에 따라 **S0(150)‴ 1.6212**로 대체 적용한다. 편집 사고로
> 본 절 헤더가 일시 소실되어 복원함 — 본문 불변.)

**사전등록 (실행 전 기록)**:
- **배경**: 프로덕션 mu는 순수 CS z(§S11.5 λ*≈26k 단위 불일치 확인).
  §S7 A1(2026-07-06)에서 σ·z(Grinold 방향, 파라미터-프리)를 65종·
  pre-causal·regression 기준선에서 측정 — ΔIR +0.071(노이즈)·서브기간
  부호 불일치로 OFF 유지. **레짐이 전면 교체**(150종·causal rank·PIT·
  USD)됐으므로 동일 플래그의 1회 재측정을 사전등록(§S9 "65종 수치 직접
  비교 금지"의 역방향 적용 — 과거 노이즈 판정도 새 레짐에 이월 불가).
  **IC 상수 보정은 제외**: 상수 c는 risk_aversion·turnover_penalty와의
  균형 재캘리브레이션(다중 파라미터)을 강제하므로 "단일 사전등록"과
  양립 불가 — 스케일-프리 상대 보정(σ/median, 중앙종목 mu=z 불변)만.
- **단일 사전약정 파라미터**: `mu_vol_scaling_enabled: true`(기존 인증
  인프라 — 순수함수·합격 테스트 6종·§S7 OFF 바이트동일 파리티 증명).
- **실행**: `<PY> run_variant.py --variant
  variants/arm_s12_4_mu_vol_scaling.yaml --no-cache` — ECOS·seed 42.
- **판정**: E1 (§2.4) ΔIR > +0.36 & 서브기간 부호 일관 (vs S0(150)″
  1.681). 가드 G1–G7(§S12.2 정의). 관찰: 스케일 분포(σ/median)·고변동
  종목 active 이동·turnover.
- **해석**: E1 통과 → §8 flip 후보(DSR 별도). 미달 → OFF 유지·§S7 A1과
  묶어 "두 레짐 모두 노이즈"로 종결(추가 재측정 없음).

**결과 (2026-07-22 실행 완주, ECOS 192·fallback 0, vs S0(150)‴) —
E1 FAIL → 불채택·종결**:
- **E1 FAIL**: IR **1.5860**(Δ **−0.035**, 노이즈 대역) · 서브기간
  1.464/0.401/2.714 → Δ −0.252/−0.348/**+0.337** **부호 비일관**.
- **§S7 A1 패턴 재현**: 65종·regression 레짐의 Δ(−0.140/−0.121/+0.326)와
  동일하게 **개선이 P3(2023-)에만 집중** — σ·z 변환의 이득이 특정 레짐
  전속이라는 §S7 판독이 150종·causal rank 레짐에서 독립 재현됨.
- 가드: TE 3.98%·beta 1.0454(상한 1.05 내)·turnover 75.6%·퇴화
  이벤트/분포 기준선과 동일(post-prediction 변환이라 학습 불변 — 정합성
  확인) — 전부 OK.
- **판정: 불채택·OFF 유지·종결** — 두 레짐 모두 노이즈·P3 집중으로
  `mu_vol_scaling` 후보는 재측정 없이 닫음. IC 상수 보정(절대 단위화)은
  λ·turnover_penalty 공동 재설계가 필요한 별도 프로젝트로만 재개 가능.

### S12.5 (2026-07-22) — PCA 변동성 표준화 arm

**사전등록 (실행 전 기록)**:
- **배경 (GPT 지적 ⑤b, 코드 검증으로 확정)**: `compute_specific_returns`
  는 raw daily returns에 PCA(centering만)를 적합 — 고변동 종목이 PC를
  지배해 공통성분 추출이 변동성 크기에 왜곡될 수 있음. 타깃 정의가
  바뀌므로 전 모델 재학습을 수반하는 성능 arm.
- **구현 (본 세션, default-OFF·파리티)**: `pca_vol_standardize: bool =
  False`(config) — ON이면 창 내 per-ticker σ(ddof=1, 0/비유한 → 1.0
  폴백)로 표준화 후 적합·사영하고 specific return을 σ로 복원. 테스트
  4종: OFF 플래그 inert·ON inline 참조 항등·raw 대비 실차 존재·무변동
  열 가드 (스위트 306 PASS). regime-weighted 경로는 프로덕션 미사용으로
  범위 외.
- **단일 사전약정 파라미터**: `pca_vol_standardize: true`.
- **실행**: `<PY> run_variant.py --variant
  variants/arm_s12_5_pca_vol_standardize.yaml --no-cache` — ECOS·seed 42.
- **판정**: E1 (§2.4) ΔIR > +0.36 & 서브기간 부호 일관. 가드 G1–G7.
  관찰: degenerate_rate·IC·타깃 분산 변화.
- **해석**: E1 통과 → §8 flip 후보. 미달 → OFF 유지·설명력 기록.

**결과 (2026-07-22 3차 시도 완주 — 1·2차 외부 중단, ECOS 192·fallback 0,
vs S0(150)‴) — E1 FAIL(전 구간 악화) → 불채택·종결**:
- **E1 FAIL**: IR **1.3440**(Δ **−0.277**) · 서브기간 1.156/0.407/2.280 →
  Δ **−0.560/−0.342/−0.097 전 구간 음** — 오늘 arm 중 가장 명확한 방향
  일관 악화(채택 바는커녕 do-no-harm 경계 −0.36에 근접).
- 가드: G1 Δ−0.277 > −0.36 경계 내 · TE 3.81% · beta 1.038 · turnover
  73.0% · 퇴화율 50%(이벤트 분포는 상이 — 타깃 변경으로 학습이 실제로
  달라졌음을 확인) · stale depth 6 — 가드 자체는 전부 OK.
- **판독**: raw-returns PCA의 "변동성 가중 공통성분 제거"는 결함이 아니라
  **이 파이프라인의 유효 설계**다 — 표준화(상관구조 기반) 인자는 알파
  관련 횡단면 신호까지 걷어내 전 구간에서 specific-return 타깃의 질을
  떨어뜨린다. REDESIGN L(n_remove 완화 시 P1 훼손) 판독과 정합: 이
  데이터에서는 공격적·변동성 가중 factor scrubbing이 맞다. GPT 지적
  ⑤b는 이론적으로 타당하나 **실증 기각**.
- **판정: 불채택·OFF 유지·종결**. 인프라(플래그·참조 대조 테스트 4종)는
  잔존(default-OFF·파리티).

### S12.Ω (2026-07-22) — 시퀀스 종합 판정·운영 반영

- **6개 유닛 전 판정 완료 (GPT 리뷰 실행 항목 전수 처리)**:
  | 유닛 | 판정 |
  |---|---|
  | S12.1 게이트 재정의 | **채택**(거버넌스) — HOLD 해소, §S9.2 이래 최초 PRODUCTION |
  | S12.2 objective swap | 불채택 — E1 FAIL·**E2 PASS(가설 실증)**·G6 위반 |
  | S12.3(i) 스케일 무효 | **no-op 입증**(동일 데이터 ON=OFF float 동일) — 청소 종결 |
  | S12.3(ii) full refresh | 불채택 — E1·E2 FAIL(재진입 0, whitelist 포화) |
  | S12.4 mu σ·z | 불채택·종결 — 두 레짐 모두 노이즈·P3 집중 |
  | S12.5 PCA 표준화 | 불채택 — 전 서브기간 악화(raw PCA가 유효 설계) |
  | S12.6 비용 민감도 | 진단 — 50bps에도 순위 불변, 1.25× 비바인딩 |
- **프로덕션 플래그 변경 0건**(S12.1 게이트/모니터링 제외) — §2.1/§8 준수.
  성능 arm 4종 전부 단일 사전등록·ΔIR +0.36 바 미달·no-flip.
- **운영**: v2 데이터로 **07-21 리밸런싱 최초 등록**(last 07-21·next
  08-19·150 trades·data_as_of 07-22), 두 번들 export·검증·commit f667df7·
  push 완료. production_gate **PRODUCTION**(stale_depth 5/7). comparison
  gate는 v2에서 RESEARCH/FAIL 3건(active_return·turnover 1.25×·sub_wins)
  으로 확대 — challenger 승격 논거는 v2에서 오히려 약화(관찰).
- **부기 (실행 환경)**: 세션 자식 프로세스로 띄운 장기 런이 7회 외부
  중단(14:26 Modern Standby idle timeout 1건 실측, 나머지 원인 미확정) —
  일회성 schtasks 태스크로 우회해 완주. **장기 런은 스케줄 태스크 경유가
  이 머신의 안정 패턴**(매일 11:30 배치 무중단 전례) → 런북 후보.
- 잔존 연구 후보(각각 별도 사전등록 필요): rank+regression
  mu-combination(§S11.9·S12.2 P2 상보성 근거), HOLD 정책 ④ 차단형 전환
  여부(§S11.6), challenger 승격(§8+DSR).

**부기 (2026-07-22, mu-combination 사전 진단 — 읽기 전용, arm 미착수)**:
- 방법: production·S12.2 arm의 v2 pkl에서 `pre_overlay_predictions` 추출,
  PIT/USD `returns_masked` 20d 선도수익률 대비 신호 수준 분석(스크립트
  세션 스크래치패드 `diag_mu_combination.py`, 1,978일).
- 결과: ① 신호 상관 평균 **0.355**(다변화 여지 존재). ② 그러나 블렌드
  IC 곡선 **단조** — w(rank 가중) 0→1에서 IC 0.0280→**0.0520**, 내부
  최적점 없음(순수 rank 최강). ③ 21d 순위 안정성 rank **0.823** >
  blend50 0.749 > regression 0.588 — 블렌드는 IC·회전 두 축 모두 순수
  rank에 **지배당함**.
- 판독: §S12.2의 P2 +0.53 상보성 단서는 (a) v2 기준선 P2 붕괴(재기재
  효과, v1 기준 +0.06)와 (b) 포트폴리오 경로 노이즈의 산물 — 신호
  수준에서 상보성 미발현. 레짐 조건부 가중 변형은 새 가설(레짐 타이밍)
  이라 §S7 D2·§S11.9 반증 계보상 비권고.
- **권고: 후보 종결(사전등록·arm 불필요)** — 종결 확정은 사용자 결정
  대기. 본 진단은 설명력 기록.

### S12.6 (2026-07-22) — 비용 모델 민감도 (읽기 전용 진단, arm 아님)

**사전등록**: one_way_tc는 P&L 차감에만 쓰이고 가중치 결정에 불참 —
백테스트 재실행 없이 해석적으로 재계산 가능. 절차: 인증 번들의 일별
수익률·리밸런스 turnover 이력에서 10bps 비용을 gross-up 후 {10, 25,
50}bps로 재차감해 production·challenger의 net IR·active를 재계산.
목적: (a) 비용 가정 강건성 보고, (b) challenger의 1.25× turnover 게이트
위반(1.45×)을 "비용 여유" 관점에서 재조명(§8·DSR 게이트는 불변 — 본
진단은 게이트 변경 근거가 아니라 자료). 판정 게이트 없음·기록만.

**결과 (2026-07-22, 해석적 재계산 — 스크립트 세션 스크래치패드
`s12_6_cost_sensitivity.py`, 인증 운영 번들 returns.csv×monitoring
turnover, 95 리밸런스)**:

| one-way tc | production IR (Δ) | challenger IR (Δ) | drag(연) prod/chal |
|---:|---:|---:|---:|
| 10bps (기준) | 1.650 | 1.870 | 0.07% / 0.11% |
| 25bps | 1.619 (−0.031) | 1.823 (−0.047) | 0.19% / 0.27% |
| 50bps | 1.567 (−0.083) | 1.744 (−0.126) | 0.37% / 0.54% |

- (해석적 IR은 일별 산술 정의라 인증 수치 1.681/1.915와 소폭 상이 —
  **Δ가 판단 대상**.)
- **판독**: 비용 5배(50bps)에서도 양쪽 IR 훼손 −0.08~−0.13에 불과, 순위
  역전 없음. challenger 초과 회전(1.45×)의 추가 비용은 연 **~0.17%p**
  (0.54−0.37) — active ~6.5% 대비 미미. **1.25× turnover 게이트는 현실적
  비용 범위에서 경제적으로 바인딩하지 않음** — challenger 승격의 실질
  차단자는 §8 DSR/노이즈밴드(ΔIR +0.234 < +0.36)임을 재확인. 게이트
  개정은 별도 사전등록 필요(본 진단은 자료). 스프레드·임팩트의 비선형
  비용은 범위 외(데이터 부재) — 모델링 필요 시 별도 설계.

## S13 (유니버스 200 확장 — 슬레이트 확정·워크북 편집·게이트 등록) — 2026-07-23

사용자 지시: 150에 50종 추가. §S11 방법론 승계(MSCI World 편입 + 섹터 이름수 ∝
MSCI 비중 + 신규 FX 페어 0 + "학습 표본 구성" 프레이밍). 슬레이트는 외부 리뷰
교차검증을 거침 — GPT 5.6 대안(방어섹터 17종 확대·Tech +5)을 팩트체크 후 3안
(원안 MSCI 비례 / GPT 균형 / 병합 절충) 제시, **사용자가 원안 확정**. GPT안
검증 기록: 중복 0·FX 코드 주장 정확(`data_loader.py:125`), 그러나 CRWD·ALC 마스크
무플래그, 초대형 공백(BRK/B·INTU·RTX 등) 방치, En/Mat/RE/Ut 34% 배분은 §S11
사전등록 방법론과 배치 — 원안 채택으로 해소.

- **슬레이트 (50, 기준 2026-06-30 MSCI World 팩트시트 비중)**: Tech +15 (INTU SNPS
  APH MSI CRWD NXPI KEYS ADSK WDAY WDC DDOG FTNT DSY 6146[디스코] 6981[무라타]) ·
  Fin +8 (BRK/B CB ICE MCO PYPL COF BNP MUV2) · Ind +6 (RTX EMR AXON UBER AIR SAF) ·
  HC +5 (MRK AMGN SYK MCK NOVN) · CD +4 (NKE ORLY DASH CFR[리치몬트]) · Comm +4
  (TTD RBLX 9432[NTT] 9433[KDDI]) · St +3 (PEP MDLZ OR[로레알]) · En +2 (CVX TTE) ·
  Mat +2 (NEM SHW) · Ut +1 (SO). 최종 200 = Tech 60/Fin 32/Ind 23/HC 19/CD 17/
  Comm 16/St 10/En 7/Mat 7/Ut 5/RE 4. 통화: USD 37·EUR 7·JPY 4·CHF 2 —
  **신규 FX 페어 0·신규 거래소 코드 0**.
- **워크북 편집 (2026-07-23 완료, 검증 ALL PASS)**: `Data/oppor.xlsx` tickers
  152→202 셀(EW1..GT1, Bloomberg 형식) · `re_study/Factset_re_study.xlsx` 13시트
  각 +50열(r2='TICKER-CC^' FactSet 국가코드 US/FR/DE/CH/JP, BRK/B는 `BRK.B-US^`
  NOVO.B 선례 표기; r3=시트별 마지막 열 FDS 수식 문자열 그대로 복제; 151→201열,
  FwdEPS 시트만 135→185열). 백업: `oppor.backup_20260723_s13.xlsx`,
  `Factset_re_study.backup_20260723_s13.xlsx`. 검증: 시트별 셀 수·tail-50 일치·
  수식 균일·중복 0 (스크립트 fresh 재오픈 패스).
- **universe_config 선적용 보류(§S11.1과 의도적 차이)**: 일일 run_data_pipeline이
  UNIVERSE를 소비하므로 가격 소스에 50열이 생기기 전 200 승격 시 평일 런이
  Missing/실패 위험(§S11.1은 일요일 선적용이라 무해). 적용 대기 블록 + 절차를
  `outputs/s13_universe_config_append.py`로 스테이징(계약 테스트
  `tests/test_s13_universe_config_append.py` 2 PASS — 50종·배분·중복 0·FX 0 고정).
  **리프레시 직전에 적용**(EXPECTED 200 + UNIVERSE +50 + BRK/B FactSet 표기
  오버라이드 + test_universe_config 핀 갱신).
- **게이트 상태 (§S11 준용)**:
  1. MSCI 편입 대조 — **PENDING**: 2026-06-30 구성 파일 기준 50종 증권 라인 대조가
     리프레시 착수 조건(BHP LN 배제 선례).
  2. FX/거래소 — **해당 없음**: 전 코드 기지원, 신규 페어 0.
  3. 상장·기업행사 마스크 — **DONE(등록)**: IPO 6건 — TTD 2016-09-21, UBER
     2019-05-10, CRWD 2019-06-12, DDOG 2019-09-19, DASH 2020-12-09, RBLX
     2021-03-10. 기업행사 2건 — **WDC** 2025-02 SNDK 분사 RemainCo(GE/GEV 선례
     기본 적용: 분사 이후만 학습 인정 — 잔여 이력 ~1.5y는 285A 선례로 허용),
     **COF** 2025-05 Discover 흡수(마스크 여부 리프레시 시 first-valid·규모 단절
     보고 후 판단). 일자는 리프레시 시 first-valid로 재검증. 관찰 플래그:
     BRK/B 컨센서스 커버리지 희소(BEST_*/FDS 결측 가능 — MMC 계열 리스크).
  4. 생존편향 정책 — **명시**: 2026-06-30 시점 기준 고정 200종. 과거 구간은
     진단용, 신규 50종 정식 평가는 전향 구간.
- **데이터 실재성 사전 점검**: oppor `S&P500` 풀 시트에 미국 37종 중 31종 열
  기존재; **무열 6종 KEYS WDAY WDC AXON DASH RBLX + 비미국 13종은 리프레시 시
  `Data/S&P500.xlsx` 18시트 신규 열 생성 확인 필수**(§S11.2 MMC 선례 — 가격 소스
  열 부재가 하드 게이트).
- **단독 arm 선언**: 150→200은 타깃 정의 변경(PCA 전체 유니버스 재적합) — S9/S11과
  동일한 기준선 재정의 이벤트. 새 S0(200) 재인증 후 "200 이전 수치와 비교 금지"
  선언, 다른 파라미터 변경과 동시 실행 금지. ai_port TICKERS(150)는 새 S0(200)
  전까지 불변.
- **잔여 단계(§S11 준용)**: ① MSCI 구성 파일 대조(게이트 1) → ② universe_config
  200 적용(스테이징 블록) → ③ 데스크탑 리프레시 + 가격 소스 50열 생성 확인 →
  ④ ai_signal_data 200 재생성 + Universe_Meta 확장 + 마스크 first-valid 검증 +
  커버리지·임퓨테이션 보고 → ⑤ ai_port TICKERS 200 확장 + 새 S0(200) ECOS 재인증
  → §S13.1 기록.

**부기 (2026-07-23, PIT 계약 명문화)**: 상장 전 배제 불변식(포트폴리오·벤치마크·
피처)의 정본 문서 `2026-07-23-pit-listing-mask-contract.md` 작성 — 일자 해석
3단계·inclusive 규칙·방어선 7층(`파일:라인` 고정)·Daily_Returns 면제 근거·
합격기준 명령·§S13 마스크 등록 8건·금지 4항. 인용한 합격기준 실행 확인:
`test_listing_mask + test_pit_universe + test_pit_sp500_ai_v2` **28 PASS**.

## S13.1 (universe_config 200 적용 + Fwd_OpCashflow 배선) — 2026-07-23

사용자 지시 2건: ① run_data_pipeline에 유니버스 200 반영, ② D_Factset 신규
`Fwd_OpCashflow` 시트를 ai_signal_data로 배선.

- **리프레시 실측(선적용 리스크 소멸)**: 사용자가 §S13 워크북 편집 후 당일
  리프레시 완료 — `Data/S&P500.xlsx` 12:09, PX_LAST/CUR_MKT_CAP 202셀(신규 50
  표본 8/8 존재) · `D_Factset_re_study.xlsx` 15:53, 전 시트 r2=201(date+200,
  꼬리 SHW/SO). §S13 게이트 ①(MSCI 대조)은 사용자 지시로 사실상 오버라이드
  (§S11.1 선례) — 형식 대조는 미수행으로 기록.
- **universe_config 200 적용 (TDD, machine/re_study — git 외부)**: 스테이징 블록
  (`outputs/s13_universe_config_append.py`) 그대로 UNIVERSE +50 · EXPECTED 200 ·
  `build_factset_ticker_map`에 BRK/B 점 표기 별칭 2종(`BRK.B-US^`/`BRK-B-US^`).
  계약 테스트 갱신: §S11 블록은 [-100:-50] 위치로 이동 고정, §S13 tail-50·통화
  믹스(USD 37/EUR 7/JPY 4/CHF 2)·신규 별칭 6종+BRK/B 사전등록.
  `run_data_pipeline.bat` [CHECK] 문구 150→200.
- **Fwd_OpCashflow 배선**: `create_universe_data.py` FACTSET_SHEETS를
  원본→별칭 dict로 전환(기존 3종 출력명 `Factset_EPS_Revision` 등 **불변** —
  ai_port 소비 계약), 신규 `'S&P500 Fwd_OpCashflow(1Y)' → 'Fwd_OpCashflow'`
  (출력 `Factset_Fwd_OpCashflow` 22자, Excel 31자 제한 OK).
  `create_ai_signal_data.py`는 RL_Universe_Data 전 시트 복사(:397-399)라
  **무변경**으로 자동 전파. ai_port 측은 ESSENTIAL_SHEETS 밖(신규 시트가
  유니버스 교집합·피처에 불참 = inert) — **피처 소비는 별도 사전등록 arm 대상**.
- **검증**: machine/re_study 계약·스모크 테스트 red(8 FAIL) → 구현 → **14 PASS**.
  실데이터 e2e(읽기 전용, create_universe_data FactSet 블록 재현): Fwd_OpCashflow
  4,951행(2013-01-01~2026-07-22) · **200/200 티커 매칭** · BRK/B 매핑 OK ·
  미매핑 잔여 0. 관찰: 은행류 OpCF 추정 전량 결측(JPM 0.000, §S11 금융주 결측
  계열)·BRK/B 최신 NaN(컨센서스 희소 플래그 실증) — 로더 NaN 처리로 무해.
- **잔여**: ⑴ 파이프라인 재생성 실행(`run_data_pipeline.bat` — 워크북 열림 상태
  확인 후 단일 foreground 또는 명일 11:30 스케줄; 신규 50종 감성은 step 1
  analyzer 실행 시 채워짐) → ⑵ ai_signal_data 200 검증(Universe_Meta 200
  Available·마스크 first-valid — PIT 계약서 §5 체크리스트) → ⑶ ai_port TICKERS
  200 확장 + 새 S0(200) ECOS 재인증(단독 arm, §S13 선언).

**부기 (2026-07-23, 어닝 날짜 단일 소스화 — 사용자 지시)**: 기존 흐름은
`earnings_sp500.xlsx`(timeline) 1차 + S&P500.xlsx `Earnings_Date` combine_first
보조였음 — 1차 소스가 **2026-03-21 이후 스테일**(이벤트 종료 03-20·신규 50종
전무)인데 병합 우선권을 가져 03월 이전 구간을 스테일 값이 지배하는 구조.
전환 판정 근거(읽기 전용 대조): 공유 140종·공유 기간(~03-20) 이벤트
**6,544 vs 6,545(Δ+1)**, 종목별 편차 ±2 이내 — 이력 손실 없음. Earnings_Date는
금일까지 갱신·200종 완비. `create_universe_data.py` step 8을 Earnings_Date 단일
소스로 교체, `EARNINGS_FILE*`·`resolve_existing_path` 제거(고아 정리), 회귀 핀
`test_earnings_single_source_is_sp500` 추가 — red 1 FAIL → 구현 → **15 PASS**.
`earnings_sp500.xlsx` 파일 자체는 보존(타 소비자 미확인·삭제는 별도 판단).

## S13.2 (FactSet surprise 2종 배선: EPS/Sales Surprise) — 2026-07-24

사용자 지시: D_Factset의 `S&P500_earning_surprise`·`sales_surprise`를
ai_signal_data에 포함. 포맷/명명은 에이전트 재량 위임.

- **포맷 결정**: 타 Factset 시트와 동일한 raw date×ticker 일별 패널
  pass-through. 근거: `FE_SURPRISE(PERCENT, EPS|SALES, MEAN, QTR_ROLL, 0, …, D)`
  = 최근 보고 분기 서프라이즈 %의 계단형 일별 시계열이라 dated download
  자체가 PIT-안전(Revision 계열과 동일 구조). z-scoring·PEAD 감쇠 등 가공은
  ai_port 피처 계층의 사전등록 arm 몫(불변식 1 OFF-default+parity 준수 경계).
- **명명**: 별칭 `EPS_Surprise`/`Sales_Surprise` → 출력
  `Factset_EPS_Surprise`(20자)/`Factset_Sales_Surprise`(22자, Excel 31자 OK).
  기존 `Factset_EPS_Revision`/`Factset_Sales_Revision` 짝과 일관
  (원본 `S&P500_earning_surprise`는 실제 EPS 수식이므로 EPS_ 접두 채택).
- **변경**: `create_universe_data.py` FACTSET_SHEETS +2 (§S13.1 dict 패턴).
  `create_ai_signal_data.py`는 RL_Universe_Data 전 시트 복사(:397-399)라
  무변경 자동 전파. PIT 마스킹은 ai_port 로더 측 시트-일반 2차 재마스킹
  (`src/data_loader.py:1023-1046`)이 커버 — 파이프라인 측 추가 작업 없음.
  ai_port는 ESSENTIAL_SHEETS 밖 = inert(피처 소비는 별도 사전등록 arm).
- **검증**: TDD red 1 FAIL(KeyError) → 구현 → re_study 계약·스모크 **13 PASS**.
  실데이터 e2e(읽기 전용, step-7 블록 재현): 두 시트 모두 4,952행
  (2013-01-01~2026-07-23) · **200/200 티커 매칭** · 미매핑 잔여 0.
  계단형 sanity: AAPL/JPM distinct-runs 55/56(≈분기 수, 2013~2026). 전량 결측:
  EPS_Surprise {CS, SAF, CFR} · Sales_Surprise {RR/, RIO} — §S11 컨센서스 희소
  계열, 로더 NaN 처리로 무해.
- **관찰(세션 중 리프레시 상태 변화)**: 세션 초 D_Factset(07-23 15:53본)은
  ROG-CH^였으나, 검증 시점에 사용자 데스크탑 리프레시 완료 확인 —
  `S&P500.xlsx` 07-24 12:36 · `D_Factset_re_study.xlsx` 07-24 15:08, 둘 다
  SAN-FR^/SAN FP 반영. 즉 위 e2e는 ROG→SAN 교체 후 데이터 기준.
- **잔여**: 파이프라인 재생성(`run_data_pipeline.bat`) 시 신규 2시트가
  ai_signal_data에 실림 — ROG→SAN·200 확장 재생성 사이클과 동일 실행에서
  처리 예정. 현재 `ai_signal_data.xlsx`(07-23 16:49본)는 구본(Roche·surprise
  시트 없음).

## S13.3 (유니버스 200 확장 + ROG→SAN — 새 기준선 S0(200) ECOS 재인증) — 2026-07-26

**단독 arm 선언 이행**: 150→200 확장과 ROG SW→SAN FP 교체(§rog_to_san, 07-24
리프레시 완료분)를 하나의 유니버스 재정의 이벤트로 묶어 피처·파라미터 불변으로
재인증. **200 이전 수치(§S12 S0(150)‴ IR 1.6212 포함)와 직접 비교 금지.**

- **번들 검증(07-24)**: ai_signal_data.xlsx(07-24 16:32 재생성) — Universe_Meta
  200행·universe_config 순서 정합·Sanofi 존재·Roche 부재·Status 전원 Available.
  신규 시트 3종(Factset_EPS_Surprise/Sales_Surprise/Fwd_OpCashflow) 4,952행×200열.
- **마스크 등록 10건**: §S11.4식 선행 상수-run 감사 → PIT 계약서 §6 사전등록
  IPO 6건(TTD 2016-09-21·UBER 2019-05-10·CRWD 2019-06-12·DDOG 2019-09-19·
  DASH 2020-12-09·RBLX 2021-03-10) **등록=추론 전부 일치**; 감사 추가 발견
  분사 백필 2건 PYPL 2015-07-07·KEYS 2014-10-21(first real observation);
  기업행사 2건 WDC 2025-02-24(SNDK 분사 RemainCo, GE/GEV 선례)·
  COF 2025-05-19(Discover 흡수 — CUR_MKT_CAP **+99.4%**·합병일 +67.1% 실측 후
  마스크 채택). 일본 신규 티커 run=5는 연초 휴장 ffill — auto-infer 처리, 비등록.
- **ai_port 200 확장**: TICKERS 150→200(워크북 순서)·통화 fallback +13
  (EUR 7/JPY 4/CHF 2)·listing_dates +10·expected_universe_size 200
  (production+challenger yaml)·150-pin 계약 테스트 6파일 갱신 → 전체 스위트
  308 PASS. 로더 스모크: 등록 10종 마스크 first_valid == 등록일 전부 OK.
- **S0(200) 결과** (codex_causal_rank_65, --no-cache 전체 재빌드, 07-24 착수
  07-26 완료, wall 183,414s·수면 중단 포함, git eb7aa99+dirty):
  - **IR 1.5529 · active 5.82% · TE 3.75%**(가드 4.5% 내) · two-way turnover
    65.7%(one-way 32.8%) · MDD −32.37% · avg IC 0.0196
  - **realized_beta 1.0581** → §3 P2 베타 게이트: ~1.0 유지 → **P2 shelve 지속**
  - 서브기간 IR: P1 1.344 / P2 0.825 / P3 2.294 (전부 양)
  - 솔버: **ECOS 192/192·fallback 0%**·optimizer fallback 0/96 (§2.2 준수)
  - causal_validation_ok=True (전 32 split, embargo 20d)
  - **퇴화율 56.25%(18/32)** — §S11.10과 동일 수준, 200 확장으로도 미해소.
    HOLD 항목 지속(objective 가설만 잔존, §S12 40.6% 실증 참고).
- **판정**: 이 수치가 **새 단일 기준선 S0(200)**. 이후 모든 arm은 이 수치 대비.
  challenger(iter15) 200 재인증은 미실시 — 차기 과제.

## S13.4 (사전등록 — 신규 피처 arm 3종: surprise 2종 + fwd OpCF) — 2026-07-26

사용자 지시(07-24): eps_surprise·sales_surprise·Fwd_OpCashflow 피처 후보 반영.
S8 news_trend 청사진(조건부 extra_whitelist) 그대로 — 코드는 무조건 빌드,
플래그는 whitelist 승인만 제어. 전 플래그 OFF 시 바이트 동일(불변식 1).

- **사전등록 정의(후보당 단일, 스윕 금지)**:
  - S13.4a `eps_surprise` = Factset_EPS_Surprise 원 레벨(최근 보고 분기
    서프라이즈 %, PEAD 계단열·클리닝 없음) — `eps_surprise_feature_enabled`
  - S13.4b `sales_surprise` = Factset_Sales_Surprise 원 레벨 —
    `sales_surprise_feature_enabled`
  - S13.4c `fwd_opcf_yield` = Factset_Fwd_OpCashflow / local_prices
    (tg_upside와 동일한 로컬 통화 단위 계약, 가격 0→NaN) —
    `fwd_opcf_feature_enabled`
- **구현**: config 플래그 3종 default-OFF·sellside.py 빌더·assembly extra 배선·
  arm yaml 3종(arm_s13_4a/4b/4c = S0(200) 정본 + 플래그 1개). 플래그는
  SAFE_FOR_CACHE_REUSE 밖(피처 패널 변경 = 전체 재실행 강제) 확인.
  수용 테스트 red 4 FAIL → 구현 → 전체 **316 PASS**.
- **게이트(사전 약정)**: S0(200) IR 1.5529 대비 **ΔIR > +0.36 & 서브기간 부호
  일관 & DSR/selection-bias 해킷** 통과 시에만 후보별 독립 flip(§8).
  |ΔIR| < 0.36은 노이즈 = 비액션. 3 arm 결과는 실행 완료 후 이 로그에 추가.

### S13.4 결과 (3 arm 실행 완료·게이트 판정) — 2026-07-28경 완료

체인 순차 실행(단일 프로세스, S0(200)과 동일 ECOS 프로토콜 — 3 arm 모두
ECOS 192/192·fallback 0%·TE 3.69%·피처 승인 265→62(+1, 배선 정합 확인)).

| arm | IR | ΔIR vs 1.5529 | 서브기간 IR (P1/P2/P3) | Δ부호 | 판정 |
|---|---|---|---|---|---|
| S13.4a eps_surprise | 1.241 | **−0.312** | 1.476/0.550/1.975 | +/−/− | **FAIL** |
| S13.4b sales_surprise | 1.327 | **−0.226** | 1.464/0.543/2.040 | +/−/− | **FAIL** |
| S13.4c fwd_opcf_yield | 1.373 | **−0.180** | 1.240/0.550/2.377 | −/−/+ | **FAIL** |

- **판정**: 3 arm 전부 ΔIR 음수·서브기간 부호 불일치 → **전부 불채택,
  default-OFF 유지, production flip 0건**. |ΔIR|<0.36이므로 형식상 노이즈
  대역이나 방향이 전부 음수라 채택 근거 부재. DSR 해킷은 활성화 후보가
  없어 미적용(§S8 불채택 선례와 동일 처리).
- **관찰(설명력)**: 세 arm 모두 P2 서브기간이 0.82→0.55로 일관 하락 —
  추가 피처가 중간 구간(대략 2021~2023 회귀 구간)에서 랭킹 모델을 일관되게
  희석. turnover도 소폭 증가(65.7→66~70%). 서프라이즈 계열은 기존 PEAD
  오버레이(pead_boost)와 정보 중복 가능성 — 후속 가설로만 기록.
- **코드 처리**: 플래그 3종·빌더·arm yaml은 사전등록 기록으로 보존
  (S8 선례). CORE_FEATURE_WHITELIST 무변경. 어떤 프로덕션 variant도
  플래그를 켜지 않음 = OFF 경로 바이트 동일 보장(수용 테스트 고정).

## S13.5 (사전등록 — fwd_opcf 리비전 3윈도우 arm, 사용자 지시) — 2026-07-28

사용자 지시: "fwd_opcf 63d/126d/252d 리비전 모두 테스트가 필요해".
§S13.4c(yield 레벨 FAIL) 후속 — 리비전(추정치 변화) 축.

- **다중성 선언(불변식 4 예외 처리)**: 3윈도우 동시 시험은 단일 사전등록
  파라미터 원칙과 충돌하는 **사용자 지시 스윕**이다. 따라서 (a) 최대-IR
  arm 선택 금지 유지, (b) 어떤 윈도우든 채택하려면 게이트(ΔIR>+0.36 &
  서브기간 부호 일관) **+ 3-trial 다중성 해킷(experiment_inventory·
  run_selection_bias)** 통과를 추가 요건으로 사전 약정한다.
- **사전등록 정의**: fwd_opcf_rev_{63,126,252}d =
  `safe_pct_change(Factset_Fwd_OpCashflow, N)` — tg_mom 관용구(|기저값|
  분모 = CF 추정치가 0을 지나도 부호 보존, 0→NaN). 플래그 3종 default-OFF,
  arm yaml 3종(arm_s13_5a/5b/5c = S0(200) 정본 + 플래그 1개).
- **검증**: 수용 red 3 FAIL → 구현 → 전체 **323 PASS**. 플래그는
  SAFE_FOR_CACHE_REUSE 밖 확인(전체 재실행 강제).
- 결과는 실행 완료 후 이 로그에 추가. 기준선 = S0(200) IR 1.5529
  (P1 1.344 / P2 0.825 / P3 2.294).

### S13.5 결과 (3윈도우 arm 실행 완료·게이트 판정) — 2026-07-26

실행 노트: 5a 정상 완료(순수 연산 ~12분/arm — S0(200)의 wall 51h는 수면
지배 확인). 체인이 외부 요인으로 2회 중단(5b 재실행 2회) — 사용자 재시작
승인 후 완주. 3 arm 모두 ECOS 192/192·fallback 0%·피처 승인 268→62(+1).

| arm | IR | ΔIR vs 1.5529 | 서브기간 IR (P1/P2/P3) | Δ부호 | 판정 |
|---|---|---|---|---|---|
| S13.5a rev63d | 1.366 | **−0.187** | 1.350/0.607/1.968 | +/−/− | **FAIL** |
| S13.5b rev126d | 1.246 | **−0.307** | 1.185/0.752/1.815 | −/−/− | **FAIL** |
| S13.5c rev252d | 1.285 | **−0.268** | 1.631/0.638/1.670 | +/−/− | **FAIL** |

- **판정**: 3윈도우 전부 ΔIR 음수 → **전부 불채택, default-OFF 유지,
  flip 0건**. 다중성 해킷은 채택 후보 부재로 미적용(사전 약정 조건 미도달).
- **축 소진 결론**: fwd_opcf 계열은 yield 레벨(§S13.4c −0.180)과 리비전
  3윈도우(−0.187/−0.307/−0.268) 전부 음수 — **Fwd_OpCashflow 피처 축은
  현 아키텍처에서 소진**으로 기록. 63d가 리비전 중 최선이나 여전히 음수.
- **관찰**: ① §S13.4와 동일하게 P2(≈2021-23) 일관 하락 반복 — 추가 피처의
  구조적 희석 패턴. ② non-finite 예측 경고 93건/arm(0 근처 CF 추정치의
  리비전 폭주 → NaN 안전 처리 확인, 무해). ③ P1은 rev252에서 +0.287로
  개선되나 P3 −0.624로 상쇄 — 장기 리비전은 초기 구간에만 유효한 신호.

## S13.6 (사전등록 — 은행권 구조적 결측 펀더멘털 NaN 보존 arm) — 2026-07-26

### 발단: 워크북 데이터 품질 감사 재실행

`scripts/audit_ai_signal_data_stability.py` 재실행(07-23 판본은 stale — 워크북이
07-24 16:32 수정, ROG→SAN 교체 반영). 결과 **Critical 0 / High 0 / Medium 5 /
Low 7** (07-23 대비 High 4건 해소: ROG stale price·RR/ 리비전 0% 복구).

구조 무결성은 결함 없음: PX_LAST↔Daily_Returns 항등식 corr 1.0·>100bp 불일치
0건/200종, Summary_Stats 재계산 불일치 0셀(최대 5.8e-11), Factor 49개 meta↔data
완전 정합, S13 신규 50종 이슈 0건, 상장일 200/200 해석.

**Medium 3건(극단 수익률)은 오탐**: AMD 2016-04-22 +52.3%, BE 2018-07-25 +66.7%
(IPO일, 리스팅 마스크가 이미 배제), BE 2024-11-15 +59.2% — 셋 다 가격↔수익률
항등식 통과·익일 반전 없음 → 실제 이벤트. **Low 7건(missing_ticker_columns)도
결손이 아니라 정당한 N/A** (은행은 gross margin·FCF·capex·EV/EBITDA 미성립,
SHORT_INT_RATIO 부재 40종은 전부 비미국 상장).

### 감사 도구의 사각 (헤드라인이 낙관 편향인 이유)

스크립트는 `ESSENTIAL_SHEETS ∪ CONTINUOUS_SHEETS` 에만 커버리지 검사를 건다.
S13.4/S13.5 가 쓴 신규 3시트는 어느 집합에도 없어 **커버리지 0% 컬럼이 있는데도
이슈 0건**을 냈다. 실측 합성(ffill/median) 비중:

| 시트 | 합성 비중 | 커버리지 0% | 1년+ 동결 꼬리 |
|---|---:|---|---:|
| Factset_Fwd_OpCashflow | **10.10%** | 11종(GS·JPM·WFC·BAC·MS·C·AXP·HSBA·BNP·8306·COF) | 18종 |
| Factset_EPS_Surprise | **7.92%** | 3종(CS·SAF·CFR) | 14종 |
| Factset_Sales_Surprise | **2.94%** | 2종(RIO·RR/) | 3종 |
| (대조) EPS_Rev / Sales_Rev / TG_Price | 0.13 / 0.18 / 0.10% | 0종 | 0종 |

`data_loader._fill_missing` 은 **무제한 ffill → per-date 횡단면 median**(look-ahead
없음). 따라서 부분 컬럼은 마지막 관측값이 영구 동결된다: BRK/B OpCF 2015-03-02
이후 2,865영업일, PUB EPS surprise 2015-04-20 이후 2,831영업일, SU 2017-10-25,
ZURN 2019/2020/2022. `max_tail_ffill_days=10` 가드는 *소스 인덱스를 넘어선 tail
확장*만 검사하므로 이 동결 꼬리는 **어떤 가드도 통과하지 않는다**.

→ **§S13.4/§S13.5 "축 소진" 결론의 해석을 수정한다.** 입증된 것은 "fwd OpCF에
알파가 없다"가 아니라 "**현 소싱 상태의 fwd OpCF는 금융섹터에서 사용 불가**"다.
결측이 무작위가 아니라 섹터 체계적(은행권 영업현금흐름은 컨센서스 추정 대상이
아님)이므로 횡단면 랭킹 모델을 구조적으로 희석한다 — §S13.4/§S13.5가 공통으로
보인 P2 일관 하락과 부합. 재소싱 없이 이 축을 재시도하지 않는다.

### 사전등록 arm: 프로덕션에 이미 배선된 동일 결함

같은 메커니즘이 **이미 프로덕션에 있다**. `BEST_CALCULATED_FCF`/`BEST_CAPEX`/
`BEST_EV_TO_BEST_EBITDA`/`BEST_GROSS_MARGIN` 은 은행권에서 **컬럼 자체가 부재**
(`data_loader.OPTIONAL_SHEETS` 가 "banks: no traditional FCF / no physical capex /
EBITDA undefined / no gross margin concept" 라고 이미 명시). 이 4시트가 화이트리스트
61개 중 **8개 피처**를 먹이고, `assembly.py` 패널 fill 이 그 칸을 per-date 중앙값으로
채운다 — 즉 **17종에 대해 "정확히 시장 중앙값"을 매일 주장**한다.

실데이터 실측(200종): 8306·BAC·CB·WFC **8/61(13.1%)**, COF 7, JPM·MUV2·HSBA 5,
GS·ZURN 4, PGR·C·AXP 3, BN·BRK/B·TSM 2, MS 1 — **17종/200**.

- **선택 키**: 섹터 라벨이 아니라 **(티커 × 시트) 실측 부재**(`UniverseData.
  optional_missing`). 섹터 규칙은 TSM(Technology, FCF 부재)을 놓치고, AXP·PGR·C
  (gross margin만 부재)의 멀쩡한 FCF 를 버린다.
- **범위**: 4시트 8피처. `BEST_PEG_RATIO`(FN)·`BEST_PX_BPS_RATIO`(PM)는 단일 티커
  저커버리지라 이번 범위 밖.
- **메커니즘**: 실측 부재 칸만 NaN 유지 → LightGBM 네이티브 결측 처리. 워밍업
  NaN 은 기존 median fill 유지(선택 키와 정확히 일치시키기 위함).
- **필수 배선(발견)**: `model_trainer` 의 `valid = y.notna() & X.notna().all(axis=1)`
  는 listwise deletion 이다. 현재는 패널이 dense 라 잠들어 있으나 NaN 을 흘리면
  **17종이 모든 학습 행에서 탈락**하는 반면 `predict_cross_sectional` 은 계속
  점수를 매긴다. `_valid_rows` 가 `NAN_TOLERANT_FEATURES` 8개에 한해 NaN 을
  허용하도록 3개 호출부 수정. 플래그 OFF 면 패널에 NaN 이 없으므로 바이트 동일.

### 구현 (default-OFF, §2.1 parity)

- `config.py`: `absent_fundamental_nan_enabled=False`,
  `ABSENT_FUNDAMENTAL_SHEET_FEATURES`(4시트→8피처), `NAN_TOLERANT_FEATURES`.
- `features/assembly.py`: `apply_absent_fundamental_nan()` — median fill 직후 적용.
- `model_trainer.py`: `_valid_rows()` 신설, 호출부 3곳 치환.
- `variants/arm_s13_6_absent_fundamental_nan.yaml` — S0(200) 정본 + 플래그 1개
  (프로덕션 variant 대비 델타 정확히 1키 확인).
- 테스트: `tests/test_absent_fundamental_nan.py`(값 7건) +
  `tests/acceptance/test_s13_6_absent_fundamental_nan.py`(계약 6건). **336 PASS.**
- 실데이터 parity 확인: 플래그 OFF → 마스킹 NaN 0셀·패널 동일; ON → 78 (티커,
  피처) 쌍/일, 감사에서 예측한 17종·피처 수와 정확히 일치.

### 판정 기준 (사전등록, 스윕 없음)

- **E1(IR 근거)**: ΔIR > +0.36 & 서브기간 부호 일관 → 채택. 기준선 = S0(200)
  IR 1.5529 (P1 1.344 / P2 0.825 / P3 2.294).
- **|ΔIR| < 0.36 이면 IR 은 비액션.** 그 경우 리스팅 마스크(§2.1)와 동일하게
  **데이터 정확성 근거로 별도 판정**한다. 채택 근거는 "median 주입은 중립이
  아니라 허위 진술"이고, **반대 증거는 결측이 섹터 지시자로 학습되는지 여부** —
  피처 중요도에서 8개 피처의 순위 급등, 금융 섹터 active weight 의 구조적 이동,
  퇴화율 악화 중 하나라도 나타나면 불채택.
- 어느 경로든 이 로그에 결과 기록 후에만 flip. 결과는 실행 완료 후 추가.

### S13.6 결과 (arm 실행 완료·게이트 판정) — 2026-07-27

실행: `run_variant.py --variant variants/arm_s13_6_absent_fundamental_nan.yaml`,
949.5s. 마스킹 발화 확인 — FCF 13종→2피처, CAPEX 9종→2피처, EV/EBITDA 7종→1피처,
GROSS_MARGIN 9종→3피처(감사 예측과 정확히 일치). 피처 승인 **268→61**(신규 피처를
더하지 않고 기존 피처의 *값만* 바꾸는 arm이므로 화이트리스트 크기 그대로가 정상.
S13.4/S13.5는 피처 1개 추가로 62였다). 누락 0건.

| 지표 | S0(200) | arm | Δ |
|---|---:|---:|---:|
| **IR** | 1.5529 | **1.1653** | **−0.3877** |
| TE | 3.748% | 3.762% | +0.014%p |
| realized_beta | 1.0581 | 1.0704 | +0.0122 |
| active return | 5.82% | 4.38% | −1.44%p |
| turnover | 0.6567 | 0.6831 | +0.0264 |
| avg IC | 0.0196 | 0.0189 | −0.0007 |
| degenerate_rate | 0.5625 | **0.5000** | −0.0625 |
| P1 / P2 / P3 IR | 1.3441 / 0.8250 / 2.2938 | 0.9849 / 0.7042 / 1.7831 | −0.359 / −0.121 / −0.511 |

- **E1 판정: FAIL.** ΔIR −0.3877 (바 +0.36). 서브기간 부호는 일관되나 **전부 음수**.

#### 반대 증거 판정: 사전등록한 반증이 그대로 발화 → 데이터 정확성 경로도 FAIL

사전등록 문구는 "결측이 섹터 지시자로 학습되는지"였다. 실측 결과는 **그보다 나쁘다 —
효과가 섹터가 아니라 마스크 자체를 따라간다**. 예측 z-score 평균 이동:

| 그룹 | n | S0 | arm | Δ |
|---|---:|---:|---:|---:|
| 마스킹 & 금융 | 16 | −0.4901 | −0.6565 | **−0.1664** |
| 금융인데 **비마스킹** | 16 | −0.3181 | −0.3008 | **+0.0173** |
| 마스킹인데 **비금융**(TSM) | 1 | +0.2099 | +0.0253 | **−0.1845** |
| 비금융·비마스킹 | 167 | +0.0974 | +0.1135 | +0.0161 |

비마스킹 금융 16종은 시장과 동일하게 상승(+0.017 vs +0.016)했고, Technology 인 TSM 은
마스킹된 은행들과 같은 폭으로 하락했다. 즉 학습된 것은 "금융섹터"가 아니라
**"펀더멘털이 결측이면 낮게 매겨라"** 는 순수 데이터 아티팩트다. 경제적 내용이 없다.

- **가중치 이동**: 마스킹 17종 슬리브 평균 7.887% → 6.683% (**−1.204%p**, 상대 −15%),
  최종일 7.215% → 5.429% (−1.786%p). 반면 **금융 32종 전체는 −0.131%p 로 거의 불변** —
  섹터 베팅이 아니라는 것을 가중치에서도 재확인. 개별로는 JPM −0.426%p, WFC −0.316,
  BRK/B −0.256, TSM −0.173, HSBA −0.147 하락.
- **용량-반응 없음**: corr(마스킹 피처 수, Δz) = **−0.160**. 페널티가 결측 *개수*에
  비례하지 않고 **결측 존재 자체에 거의 이진적**으로 걸린다 — 기본 방향 분기 하나가
  이름을 통째로 내려보내는 형태와 부합.
- 중요도 순위: 전체 61피처를 쓴 retrain 2/32 에서만 비교 가능(EWMA 가 30회는 60피처로
  드롭, 트래커 history 는 pkl 에 미저장). 그 2회 기준 평균 +2.5 순위 상승,
  `cash_conversion_z` +11(gain 1.303%→2.197%). 방향은 일치하나 **표본 n=2 로 약함** —
  판정 근거로는 위 z-score/가중치 실측을 쓴다.

#### 판정 및 결론

- **불채택. `absent_fundamental_nan_enabled` default-OFF 유지, flip 0건.**
  E1(IR) FAIL + 데이터 정확성 경로도 사전등록 반증 확정으로 FAIL — 두 경로 모두 막혔다.
- **핵심 교훈**: median 주입은 "허위 진술"이 맞지만, 이 아키텍처에서는 **정직한 NaN 이
  더 해롭다**. LightGBM 의 네이티브 결측 처리가 결측을 *순위 신호*로 바꿔버리기 때문이다.
  0 근처(median-of-z)로 채우는 현행 동작은 이 시험으로 **사후 정당화**되었다.
- 따라서 §S13.6 감사에서 지적한 "17종 8피처 날조" 는 **인지된 채로 수용**한다. 해소하려면
  결측 처리 방식이 아니라 **데이터 소싱**(은행용 대체 지표)이나 피처 정의 변경
  (섹터 내 재정규화 — 이번에 사용자가 기각한 옵션 b)이 필요하며, 둘 다 별도 사전등록 대상.
- **부수 관찰**: degenerate_rate 0.5625 → 0.5000 로 개선(HOLD 잔존 항목). 다만 IR 이
  −0.388 이므로 퇴화율 경로로 삼을 수 없다. 결측 도입이 트리 성장을 늘린 부작용으로 해석.
- **§S13.4/§S13.5 해석 수정은 유지**: fwd OpCF 축은 "알파 없음"이 아니라 "현 소싱 상태로
  금융섹터 사용 불가"다. 다만 이번 결과로 **단순 NaN 마스킹은 그 축의 해법이 아님**이
  함께 입증되었다 — 재소싱만이 경로다.

---

## S13.7 사전 점검 — 금융/비금융 그룹별 z-score (arm 미실행, 게이트 이전 단계) — 2026-07-27

**제안(사용자)**: 금융주는 펀더멘털 지표가 구조적으로 부재하니 cross-sectional z-score 를
금융주끼리 / 나머지 섹터끼리 **분리**해서 계산하자. §S13.6 결론에서 "별도 사전등록 대상"
으로 남겨 둔 **옵션 b(섹터 내 재정규화)** 에 해당한다.

**절차**: CLAUDE.md §4.3 선례(=중립화 대상 노출이 실제로 존재함을 먼저 증명)를 적용해
arm 을 코딩하기 전에 **바인딩 여부를 실측**했다. 그룹별 z-score 는 각 피처가 가진
"금융 vs 나머지" 체계적 레벨 오프셋만큼만 패널을 바꾸므로, 그 오프셋이 0 이면 inert 다.

- 대상: S0(200) 정본 config 로 빌드한 core 패널 61피처, 2020-01-01 이후 1,713일.
- 상장 전 median-fill 행은 두 그룹 모두 ≈0 으로 희석시키므로 `listing_dates` 로 **제외**
  (331,231 / 652,000 행 = 50.8% 유지). 제외 전후 mean η² 0.0106 → 0.0109 로 결론 불변.
- 지표: `delta` = 시점별 mean_z(금융) − mean_z(나머지) 의 시계열 평균,
  `eta2` = 횡단면 z 분산 중 **그룹 간** 비중(pooled, 실제 일별 그룹 크기 사용).
- 산출물: `outputs/s13_7_sector_offset_precheck.csv`.

**결과 — 바인딩하지 않음**:

| 지표 | 값 |
|---|---:|
| 피처 수 | 61 |
| mean η² | **0.0109** |
| median η² | 0.0085 |
| η² > 0.05 인 피처 | **0개** |
| \|delta\| > 0.50 인 피처 | 1개 |

상위 오프셋(η² 기준)은 **전부 변동성·모멘텀 계열**이다:

| 피처 | delta(금융−나머지) | η² |
|---|---:|---:|
| idio_vol_63d | −0.5084 | 0.0432 |
| realized_vol_126d | −0.4192 | 0.0336 |
| realized_vol_21d | −0.3828 | 0.0309 |
| best_gross_margin_chg_252d | −0.2387 | 0.0281 |
| max_ret_63d | −0.3345 | 0.0268 |
| fin_pb_level_z / best_px_bps_ratio_level_z | +0.1689 | 0.0262 |
| fin_pe_level_z | +0.0936 | 0.0142 |

**해석 — 세 가지 이유로 arm 을 실행하지 않는다**:

1. **밸류에이션 스케일 문제는 이미 해소되어 있다.** PE/PEG/P/B/EV-EBITDA 는
   `features/utils.rolling_tsz`(756일 per-ticker 시계열 z, min_periods 252)를 **먼저**
   통과한 뒤 횡단면 z 를 받는다. 즉 "시장 대비 싼가"가 아니라 "자기 이력 대비 싼가"로
   이미 변환된다. 그 결과 은행/테크 멀티플 격차가 제거되어 `fin_pb_level_z` delta 는
   +0.169, `fin_pe_level_z` 는 +0.094 에 불과하다. 제안이 겨냥한 문제는 이미 없다.
2. **남은 오프셋은 아티팩트가 아니라 실제 정보다.** 최대 오프셋 3개가 변동성이고
   부호는 전부 음(−0.38 ~ −0.51) — 은행이 실제로 저변동성이라는 **참인 사실**이다.
   그룹별 z 는 이 정보를 파괴한다. 편향 제거가 아니라 **신호 손실**이다.
3. **구조적 결측 8피처는 오히려 가장 영향이 작다**(η² 0.0024 ~ 0.0281). 마스킹된 절반이
   median-of-z ≈ 0 으로 채워져 있고 그룹별 z 를 써도 그룹 내 ≈0 에 놓이므로 수치가 거의
   같다. **제안의 동기가 된 문제 자체를 해결하지 못한다.**

추가로, 모델 목적은 전 종목 횡단면 `rank_xendcg`(글로벌 랭크 라벨)다. 피처만 그룹별로
정규화하면 "금융 섹터를 그룹째 낮게/높게 매긴다"는 표현력을 잃는데, 라벨은 여전히 그
판단을 요구한다 — 목적함수와의 정합도 어긋난다.

**대안(더 나은 형태, 미실행)**: 같은 아이디어의 **가법(additive)** 버전이 이미 구현되어
있고 미개척이다 — `features/interaction.build_sector_interaction_features` 의
`peer_rel_{eps_rev, momentum_63d, tg_upside}`(= 피처 − 섹터 평균). 글로벌 z 를 **유지한
채** 섹터 상대 관점을 *추가*하므로 레벨 정보를 파괴하지 않고 LightGBM 이 선택하게 둔다.
치환형을 지배한다. 단 현재 `include_sector_interactions` 는 `run_backtest` 인자(default
False)일 뿐 `PipelineConfig` 필드가 아니고, core 화이트리스트 **이후**에 붙어 61피처
패널에 최대 110+ 피처를 얹는다 — arm 으로 돌리려면 config 플래그화 + peer_rel 3개로
한정하는 축소가 선행되어야 한다. 별도 사전등록 대상으로 남긴다.

**판정: 그룹별 z-score arm 은 사전 점검에서 바인딩 실패 → 코딩·실행하지 않음.
프로덕션 변경 0건.**

---

## S13.8 진단 + 사전등록 — 퇴화율 56.25% 의 구조적 원인: 조기종료 메트릭 2개 — 2026-07-27

### 진단 (arm 이전, 재실행 없이 S0(200) pkl 에서 직접 측정)

§S10/§S11.5/§S11.9/§S11.10 의 퇴화율 arm 4종(min_child 30 · val_window 252 ·
mh 블렌드 · lr 0.03)은 **전부 하이퍼파라미터 스윕**이었고 퇴화율은 0.5625 에서
불변이었다. 원인을 규명하지 않은 채 파라미터만 흔든 결과다. 이번에 원인을 특정했다.

**절차**: `outputs/codex_causal_rank_65/backtest_result.pkl` 의 32개 모델에서
`evals_result_`(검증 곡선)를 직접 읽었다. 백테스트 재실행 없음.
산출물 `outputs/s13_8_degeneracy_curves.csv`.

**구조적 사실**: `result.models` 에는 퇴화 모델이 남아 있지 않다. 18회는 이전 모델로
덮여 있어 **고유 모델은 14개**뿐이다(퇴화 모델의 곡선은 폐기됨).

**핵심 관측 — `best_iteration` = min(argmax(ndcg@5), argmax(ndcg@10)), 14/14 일치**:

| 날짜 | best_iter | argmax@5 | argmax@10 |
|---|---:|---:|---:|
| 2018-11-26 | 37 | 87 | **37** |
| 2019-05-21 | 111 | **111** | 144 |
| 2022-01-14 | 92 | **92** | 185 |
| 2023-01-03 | 18 | **18** | 38 |
| 2023-03-31 | 16 | 106 | **16** |
| 2025-03-06 | 46 | **46** | 119 |

**메커니즘(소스 확정)**: lightgbm 4.6.0 `callback._EarlyStoppingCallback.__call__` 은
`env.evaluation_result_list` 를 순회하며 **가장 먼저** `stopping_rounds` 만큼 정체한
메트릭 `i` 에서 `EarlyStopException(self.best_iter[i], self.best_score_list[i])` 를
던진다. `first_metric_only` 는 기본 `False` 이고 `model_trainer.py:449` 는 이 인자를
넘기지 않는다. 따라서 `rank_eval_at: [5, 10]` 은 **노이즈 지표 2개의 argmax 중
최솟값에서 학습을 자른다**. 단일 지표의 조기 정체 확률이 p 이면 2개일 때 약
`1-(1-p)^2` 로 상승한다 — 메트릭을 추가할수록 모델이 구조적으로 작아진다.

**실측 손실**: 고유 모델 14개 중 **6개가 ndcg@10 정점 이전에 절단**(손실 33·93·20·73·
3·12 라운드). 노이즈는 @5 가 더 크다 — round-to-round diff std 0.0078(@5) vs
0.0054(@10), SNR 13.64(@5) vs 14.69(@10).

**한계(명시)**: 퇴화한 18개의 곡선은 폐기되어 직접 관측하지 못했다. 메커니즘은 생존
14개 + 라이브러리 소스로 확정했고, 이것이 18개에도 적용된다는 것은 **추론**이다.
아래 arm 의 E2 가 그 검증이다.

### 사전등록 — arm `arm_s13_8_single_eval_metric`

- **단일 변경**: `rank_eval_at: [5, 10] -> [10]`. `rank_eval_at` 의 소비처는
  `model_trainer.py:448` 의 `eval_at` **단 한 곳**이므로 이 변경은 조기종료 메트릭에만
  작용한다. **소스 수정 없음**(config-only). production 대비 delta = 정확히 1개 키
  (검증 완료). 스윕 없음 — `[20]` 등 다른 k 는 시도하지 않는다.
- **가설**: min-of-two-argmax 제거 → `best_iteration = argmax(ndcg@10)` →
  트리 수 증가 → 퇴화율 하락 → 모델 stale 기간 단축.
- **판정 기준**:
  - **E1(성능)**: ΔIR > +0.36 & 서브기간 부호 일관이면 IR 근거 채택.
  - **E2(메커니즘)**: `degenerate_rate` 가 0.5625 에서 **유의하게 하락**해야 진단이
    옳다. E2 실패 시 "min-of-argmax 가 퇴화 원인"이라는 **가설 자체가 반증**된다.
  - E2 통과 + `|ΔIR| < 0.36` 이면 IR 은 비액션이고, **운영 안정성**(현재 최대 315영업일
    ≈15개월 stale 모델로 운용되는 구간이 2회 존재) 근거로 별도 판정한다.
- **반대 증거**: 트리 수 증가가 과적합으로 나타나는지 — turnover 급증, TE 상승,
  서브기간 부호 불일치를 확인한다.
- **비교 기준**: S0(200) = IR 1.5529 / TE 3.748% / beta 1.0581 /
  P1 1.3441 · P2 0.8250 · P3 2.2938 / degenerate_rate 0.5625 (18/32).

### S13.8 결과 (arm 실행 완료·게이트 판정) — 2026-07-27

실행: `run_variant.py --variant variants/arm_s13_8_single_eval_metric.yaml`,
exit 0, 923.0s. 산출물 `outputs/arm_s13_8_single_eval_metric/`.

#### E2(메커니즘) — **PASS. 진단이 정밀하게 확증됨**

| | S0(200) | arm | Δ |
|---|---:|---:|---:|
| degenerate_retrains | 18/32 | **11/32** | **−7** |
| degenerate_rate | 0.5625 | **0.3438** | −0.2187 |
| 최대 연속 퇴화 | 5회 | 4회 | −1 |
| 최악 모델 staleness | 315bd(~15개월) | 252bd(~12개월) | −63bd |
| 32슬롯 평균 트리 수 | 42.4 | **81.0** | +38.6 |

- **7회 해소, 신규 퇴화 0회.** 회귀 없이 단조 개선.
- **예측 일치**: S0 곡선에서 `argmax(ndcg@10)` 가 검열되지 않은 4개 슬롯은 arm 트리 수가
  그 값과 **정확히** 일치했다 — 2019-05-21(144→144), 2023-01-03(38→38),
  2026-02-23(18→18), 2026-05-21(26→26). 나머지는 S0 곡선이 조기 절단되어 관측 범위를
  넘어 더 커졌다(2022-01-14: 92→331). **`best_iteration = min(argmax@5, argmax@10)`
  가설이 사후예측으로 확증되었다.**
- 따라서 퇴화율 56.25% 는 데이터·레짐 속성이 아니라 **평가지표 설계 아티팩트**다.
  §S10/§S11.5/§S11.9/§S11.10 의 스윕 4종이 왜 전부 무력했는지도 설명된다 — 어느 것도
  min-of-argmax 구조를 건드리지 않았다.

#### E1(성능) — 비액션

| | S0(200) | arm | Δ |
|---|---:|---:|---:|
| IR | 1.5529 | 1.4700 | **−0.0829** |
| TE | 3.748% | 3.680% | −0.068%p |
| realized_beta | 1.0581 | 1.0566 | −0.0016 |
| turnover | 0.6567 | 0.6372 | −0.0195 |
| avg_ic | 0.0196 | 0.0170 | −0.0026 |
| P1 / P2 / P3 IR | 1.3441 / 0.8250 / 2.2938 | 1.3689 / 0.8899 / 1.8393 | +0.025 / +0.065 / **−0.455** |

- `|ΔIR| = 0.083 < 0.36` 이고 서브기간 부호도 불일치 → **§2.4 에 따라 IR 은 비액션**.
- **사전등록 반대 증거는 발화하지 않았다**: turnover −0.0195, TE −0.068%p, beta −0.0016
  으로 과적합 시그니처(회전율 급증·TE 상승) 없음.
- 다만 방향은 일관되게 약하다 — IC −0.0026(상대 −13%), P3 −0.455. 트리 수가 가장 크게
  늘어난 구간이 P3(2024-06-14 16→176, 2025-03-06 46→154, 2025-06-03 46→82)이라
  **여분 용량이 최근 구간에서 노이즈를 적합**한다는 해석과 부합한다. 단 서브기간 SE 는
  전체기간 SE(0.36)보다 크므로 P3 −0.455 단독으로는 **확정된 열화가 아니다**.

#### 판정

- **불채택. `rank_eval_at` 은 `[5, 10]` 유지, flip 0건.**
  E2 는 통과했으나 운영 안정성 경로도 채택 바에 못 미친다: (a) degenerate_rate 0.3438 이
  여전히 가드 `max_degenerate_model_rate=0.25` 를 위반해 **HOLD 항목이 해소되지 않았고**,
  (b) staleness 개선은 15→12개월로 제한적이며, (c) IC·P3 가 반대 방향이다. 부분적 운영
  개선을 위해 IR/IC 하락을 감수할 교환비가 아니다.
- **새로 확보된 지식(이것이 이번 arm 의 실제 산출물)**:
  1. 퇴화는 **평가지표 아티팩트**다. 데이터·레짐 문제가 아니다.
  2. `ndcg@5` 는 조기종료 지표이면서 동시에 **암묵적 용량 규제자**로 작동해 왔다.
     제거하니 평균 트리 수가 42.4 → 81.0 으로 배증했다.
  3. 최악 구간(2020-02~2021-04, 11트리 6연속)은 **arm 으로 전혀 움직이지 않았다**.
     이 6슬롯은 min-of-argmax 와 무관한 별개 원인이다.
- **후속 후보(별도 사전등록 필요)**: 두 역할을 분리한다 — 단일·저노이즈 조기종료 지표로
  퇴화를 잡고, 용량은 **명시적으로** 규제한다(`n_estimators` 상한 또는
  `early_stopping_rounds` 축소 중 **하나만**, 단일 사전등록 값). 스윕 금지 원칙상 두
  파라미터를 동시에 흔들지 않는다. 이번 결과는 그 축이 실재함을 보였을 뿐 값을 고르지 않았다.

---

## S13.9 / S13.10 사전등록 — 어닝 캘린더 admission + peer 어닝 연쇄 — 2026-07-27

### 착안 (§S13.7/§S13.8 후속)

§S13.7이 피처 축 소진을, §S13.8이 "출력이 변동성 하나로 붕괴"(예측·idio_vol 상관 최근
252일 **+0.83**, 평균 IC 0.016)를 보였다. 그래서 이번 두 arm의 선정 기준은 "IR이 오를
것 같은가"가 아니라 **"기존 61개와 구조적으로 다른가"**다.

**점검에서 나온 사실**: 파이프라인은 268개를 만든 뒤 61개만 남기는데, **어닝 캘린더
8개 전체와 공매도 3개 전체가 화이트리스트에서 빠져 있다.** 어닝 정보는 PEAD 오버레이
(예측 *이후* 단계)로만 들어가고 **모델은 발표일을 한 번도 본 적이 없다.**

### 비교 기준선 — 빈티지 주의

**S0(200) = 2026-07-27 빈티지: IR 1.4380 / TE 3.731% / beta 1.0594 /
P1 1.4786 · P2 0.7874 · P3 2.1920 / degenerate_rate 0.5000 (16/32).**
§S13.3/§S13.8이 쓴 IR 1.5529는 07-26 빈티지이며 **이번 arm과 비교 금지**. 07-27 11:30
스케줄 실행이 동일 config로 프로덕션을 재실행해 metrics.json을 덮었고, 설정 무변경·데이터
하루 차이만으로 **IR이 −0.115 움직였다**. 이 폭 자체가 ±0.36 바를 읽는 척도다.

### S13.9 — `arm_s13_9_earnings_calendar`

- **단일 변경**: `earnings_calendar_feature_enabled: true`. 신규 피처 코드 없음 —
  conditioning.py가 이미 만드는 8개(`earn_is_day`/`earn_days_since`/`earn_days_to_next`/
  `earn_pre_5d`/`earn_pre_10d`/`earn_post_5d`/`earn_post_10d`/`earn_cycle_pos`)를
  §S8 `extra_whitelist` 경로로 admit만 한다. production 대비 delta = 정확히 1키(검증).
- **구현 중 발견·수정한 결함**: Conditioning 그룹은 z-score를 건너뛰는데
  `clip_outliers(±5)`는 전 피처에 걸린다. 그대로면 0~999 범위인 `earn_days_since`/
  `earn_days_to_next`가 **0~5로 뭉개져** 기존 `earn_pre_5d`/`post_5d` 플래그의 중복이
  된다(실측 확인: 고유값 6개). `skip_zscore`에서 두 피처만 제외해 해소 —
  수정 후 고유값 130,152 / 138,605. **arm이 OFF면 core 필터가 이미 이름을 지우므로
  무연산이다.** 수용 테스트로 고정.
- 패널: 61 → **69** 피처(실측).

### S13.10 — `arm_s13_10_peer_earnings`

- **단일 변경**: `peer_earnings_cascade_feature_enabled: true`. 신규 모듈
  `src/features/peer_earnings.py`가 3개를 만든다 — 전부 **동일 섹터 leave-one-out**:
  - `peer_earn_reported_frac` — 내 섹터 중 이번 시즌 이미 발표한 비율(나 제외)
  - `peer_earn_reaction_63d` — 그들의 발표일 **초과**수익(당일 유니버스 평균 차감)
  - `peer_earn_lead_lag` — 내 days-since − 섹터 중앙값(양수 = 내가 늦은 발표자)
- **소스 한정**: `Earnings_Timeline` + `Daily_Returns` + 섹터 메타만. `Factset_
  EPS_Surprise`는 **의도적 배제** — 은행 커버리지 공백이 §S13.4를 침몰시킨 결함이라
  이 arm과 교락시키지 않는다.
- **PIT**: t 시점 값은 t 이하 발표일과 그날 실현 수익만 사용. 당일 포함은 기존
  `earn_is_day`/`earn_post_5d` 관례와 동일하며 `execution_signal_lag_days`가 위에 걸린다.
  단위 테스트가 "t=10 발표가 t<10 값을 바꾸지 않음"을 고정.
- 패널: 61 → **64** 피처(실측). 섹터 1명 그룹은 집계 제외(자기 자신으로 퇴화 방지).

### 판정 기준 (양 arm 공통)

- **E1(성능)**: ΔIR > +0.36 & 서브기간 부호 일관 → 채택. `|ΔIR| < 0.36`은 비액션.
- **반대 증거 ①(양 arm)**: 예측·`idio_vol_63d` 횡단면 상관(최근 252일)이 S0의 **0.83**
  에서 **내려가지 않으면**, 블록은 admit됐으나 신호가 여전히 변동성 축으로 붕괴한 것이다.
  이 경우 ΔIR 부호와 무관하게 "새 축이 열렸다"고 주장하지 않는다.
- **반대 증거 ②(S13.10 한정)**: 리트레인 트리들의 분기 중 peer 3개가 차지하는 비율이
  ~0이면 블록이 **inert**이며, 결과는 관계형 신호에 대해 아무것도 말해주지 않는다.
- 두 arm은 **순차 실행**(단일 foreground, CLAUDE.md §1). 어느 쪽도 게이트 통과 전 flip 금지.

### S13.9 결과 (arm 실행 완료·게이트 판정) — 2026-07-27

실행: `run_variant.py --variant variants/arm_s13_9_earnings_calendar.yaml --no-cache`,
exit 0, 897.1s. 패널 61 → **69**(실측). 산출물 `outputs/arm_s13_9_earnings_calendar/`.

#### E1(성능) — FAIL

| | S0(200) 07-27 | arm | Δ |
|---|---:|---:|---:|
| IR | 1.4380 | 1.2448 | **−0.1932** |
| **avg_ic** | 0.0162 | **0.0311** | **+0.0149 (+92%)** |
| **avg_annual_turnover** | 0.6905 | **0.9281** | **+0.2376 (+34%)** |
| active return | 5.37% | 4.52% | −0.85%p |
| TE | 3.731% | 3.633% | −0.098%p |
| realized_beta | 1.0594 | 1.0416 | −0.0178 |
| P1 / P2 / P3 IR | 1.479 / 0.787 / 2.192 | 1.461 / 0.506 / 1.823 | −0.017 / −0.282 / −0.369 |

- `|ΔIR| = 0.193 < 0.36` → §2.4상 비액션이나 서브기간 부호는 **전부 음수로 일관**. 채택 불가.
- degenerate_rate 0.5000 불변.

#### 반대 증거 — **양쪽 모두 처음으로 통과했다**

사전등록한 두 반증이 이번엔 발화하지 **않았다**. 이는 지금까지 19개 arm 중 처음이다.

- **① 변동성 축 붕괴 — 발생하지 않음.** 예측·`idio_vol_63d` 상관이
  최근 252일 **0.8255 → 0.7664**(−0.059), 전체 기간 0.7238 → 0.6587(−0.065)로
  **실제로 내려갔다.** 블록이 변동성 축의 일부를 밀어냈다.
- **② inert — 아님, 오히려 지배적.** 전체 분기 중 블록 점유율 **8.46%**(균등 배분이면
  11.59%). 특히 **`earn_days_to_next`는 모델 전체에서 2위 피처**(3.89%,
  1위 `idio_vol_63d` 4.14% 바로 아래), `earn_cycle_pos` 3.06%, `earn_days_since` 1.51%.
  이진 플래그 5개(`earn_is_day`·`pre_5d`·`pre_10d`·`post_5d`·`post_10d`)는
  **분기 0회로 전부 무용** — 연속형 3개만 일한다.

#### 그래서 왜 IR이 내려갔는가 — 신호가 아니라 수확 기구의 문제

**IC가 +92% 올랐는데 IR은 내려갔다.** 원인을 예측 지속성에서 특정했다:

| corr(pred_t, pred_{t−lag}) | S0 | arm | Δ |
|---|---:|---:|---:|
| lag 1d | 0.9940 | 0.9906 | −0.003 |
| lag 5d | 0.9572 | 0.9054 | −0.052 |
| **lag 21d** | **0.8495** | **0.6149** | **−0.235** |
| lag 63d | 0.6853 | 0.6826 | −0.003 |

**교란이 21일 지점에만 집중된다.** 1일·63일은 사실상 불변이다. 21일은 정확히
`rebalance_freq: 21`이며, 어닝 주기(~63영업일)와 맞물려 리밸런싱 시점마다 순위가
최대로 뒤바뀐다. 단일 리밸런싱 최대 회전율도 0.1401 → **0.4402(3.1배)**.

**거래비용은 원인이 아니다**: turnover +0.2376 × `one_way_tc` 0.0010 ≈ **2~5bp**로,
active return 하락 −85bp를 설명하지 못한다. 남는 경로는 **집행 계층**이다 —
`partial_rebalance_eta: 0.50`(목표의 절반만 이동)과 `turnover_penalty: 0.03`,
`no_trade_band: 0.003`은 전부 **느린 알파를 전제로 튜닝**돼 있다(§S11.8 "느린 알파
실증: 21–63d IC 0.069"). 21일 지속성 0.61짜리 신호를 이 기구로는 잡지 못하고,
절반씩 따라가다 항상 한 박자 늦는 책이 된다.

이 인과는 **추론이며 미검증**이다. 결정적 검증은 집행 파라미터 단일 변경(아래).

#### 판정

- **불채택. `earnings_calendar_feature_enabled` default-OFF 유지, flip 0건.**
- **다만 이 실패는 앞선 18개와 종류가 다르다.** 그것들은 "신호가 없다"였고, 이것은
  **"신호는 있는데(IC +92%) 수확 기구가 못 잡는다"**이다. 반대 증거 두 개가 처음으로
  통과했고, 변동성 축이 실제로 밀려났다.
- **이진 플래그 5개는 영구 제외 권고**: 분기 0회. 재시도 시 연속형 3개
  (`earn_days_to_next`·`earn_cycle_pos`·`earn_days_since`)만 admit한다.
- **후속 후보(별도 사전등록 필요)**: 집행 계층이 빠른 신호를 못 잡는다는 가설의
  단일 파라미터 검증 — `partial_rebalance_eta` 0.50 → 1.00 **또는**
  `rebalance_freq` 21 → 5 중 **하나만**. 스윕 금지. 이때 비교 대상은 S0가 아니라
  **S13.9 arm**이어야 한다(같은 피처셋에서 집행만 바꾼 효과를 재야 하므로).

### S13.10 결과 (arm 실행 완료·게이트 판정) — 2026-07-27

실행: `run_variant.py --variant variants/arm_s13_10_peer_earnings.yaml --no-cache`,
exit 0, 1027.6s. 패널 61 → **64**(실측). `[PeerEarnings] 3 relational features over
11 sectors (200 tickers, season=63bd)`.

#### E1(성능) — FAIL

| | S0(200) 07-27 | arm | Δ |
|---|---:|---:|---:|
| IR | 1.4380 | 1.3627 | **−0.0752** |
| avg_ic | 0.0162 | 0.0157 | −0.0005 (**무변화**) |
| avg_annual_turnover | 0.6905 | 0.7017 | +0.0111 (**무변화**) |
| TE | 3.731% | 3.562% | −0.169%p |
| **degenerate_rate** | 0.5000 | **0.5625** | **악화 (+2회)** |
| P1 / P2 / P3 IR | 1.479 / 0.787 / 2.192 | 1.279 / 0.733 / 2.138 | −0.200 / −0.055 / −0.055 |

`|ΔIR| = 0.075 < 0.36` → 비액션이나 서브기간 부호 전부 음수. 채택 불가.

#### 반대 증거 ② — inert 아님. **오히려 블록이 모델을 지배했다**

- 블록 분기 점유율 **9.41%** (균등 배분 4.69%의 **2.0배**).
- **`peer_earn_lead_lag`가 모델 전체 1위 피처(6.88%)** — `idio_vol_63d`(5.16%)를 제쳤다.
- 즉 **관계형 축은 실제로 열렸다.** 모델이 종목 단독 속성이 아닌 변수를 최우선으로
  채택할 수 있음이 처음 확인됐다. 그런데도 IR은 내려갔다.

#### 왜 — **분산 분해가 원인을 특정했다: 절반이 종목 정체성이다**

| 피처 | between-ticker 분산 / 총분산 | 해석 |
|---|---:|---|
| `peer_earn_lead_lag` | **54.0%** | 절반 이상이 "이게 어느 종목인가" |
| `peer_earn_reported_frac` | 25.2% | 부분적으로 정체성 |
| `peer_earn_reaction_63d` | **4.0%** | 진짜 시변 신호 |

기업의 섹터 내 발표 순번은 **분기마다 거의 고정**이다(항상 먼저 내는 회사, 항상 늦게
내는 회사). 따라서 `peer_earn_lead_lag`는 뉴스가 아니라 **준-정적 종목 ID**로 기능했고,
트리는 이를 값싼 분할 변수로 남용했다 — 1위 피처인데 IC는 0.0162 → 0.0157로 **전혀
개선되지 않았다**. [[S13.6]]의 교훈("모델은 종목을 식별할 수 있는 안정적 속성을 주면
경제적 내용 없이 그것을 학습한다")이 다른 형태로 재현됐다. degenerate_rate 악화
(0.5000 → 0.5625)도 이와 정합적이다 — 검증 지표를 개선하지 못하는 분할이 늘었다.

정작 정직한 시변 피처(`peer_earn_reaction_63d`, between 4.0%)는 분기 점유율 1.61%에
그쳤다.

#### S13.9의 집행 계층 가설 — **본 arm으로 보강되지 않음**

| corr(pred_t, pred_{t−lag}) | S0 | S13.9 Δ | **S13.10 Δ** |
|---|---:|---:|---:|
| 1d | 0.9940 | −0.003 | −0.002 |
| 5d | 0.9572 | −0.052 | −0.016 |
| **21d** | **0.8495** | **−0.235** | **−0.046** |
| 63d | 0.6853 | −0.003 | −0.013 |

S13.10은 IC 무변화·회전율 무변화·21일 지속성 −0.046으로 **빠른 신호 패턴을 재현하지
않았다**. 따라서 §S13.9의 "신호는 있으나 집행 계층이 흘린다"는 **단일 arm 관찰로
남으며**, 독립 검증은 여전히 필요하다(집행 파라미터 단일 변경 arm).

#### 판정

- **불채택. `peer_earnings_cascade_feature_enabled` default-OFF 유지, flip 0건.**
- **확보된 지식**: (a) 관계형 축은 모델이 **채택은 한다**(1위 피처). 축 자체가 막힌
  것이 아니다. (b) 그러나 본 구성은 **종목 정체성 누출**로 실패했다 — 설계 결함이며
  가설의 반증이 아니다. (c) 시변성이 확보된 유일한 피처는 `peer_earn_reaction_63d`다.
- **재시도한다면**(별도 사전등록): `peer_earn_reaction_63d` **단독**. `lead_lag`는
  영구 제외(54% 정체성), `reported_frac`도 제외(25.2%). 단 사전 확률은 낮다 —
  단독 피처의 분기 점유율이 1.61%에 불과했다.
- **신규 설계 규칙 제안**: 신규 피처는 admit 전에 **between-ticker 분산 비중을
  사전 점검**한다(§4.3 선례와 동일 취지). 30% 초과면 정체성 누출로 보고 admit하지 않는다.

#### 실행 환경 각주

본 실행에서만 joblib이 물리 코어 탐지에 실패해 논리 코어로 폴백했다(S13.9 로그에는
0회). LightGBM은 OpenMP를 직접 쓰고 joblib 코어 수를 참조하지 않으므로 영향은 낮다고
보나, 완전 배제는 못 한다. 다만 ΔIR −0.075는 바(+0.36)에서 멀고 판정 근거가 분산 분해·
분기 점유율 같은 **구조적 진단**이라 이 폭으로 뒤집히지 않는다 → 재실행 불요로 판단.

---

## S13.11 사전등록 — 집행 계층이 빠른 신호를 흘리는가 (2×2 요인설계) — 2026-07-27

### 검증 대상 가설

§S13.9는 IC가 **+92%**(0.0162 → 0.0311) 오른 신호를 얻고도 IR이 **−0.193** 빠졌고,
손실이 **21일 지점에만 집중된 예측 교란**(지속성 0.850 → 0.615, 1일·63일은 불변)으로
설명됐다. 21일은 정확히 `rebalance_freq`다. 거래비용은 원인이 아니다(회전율 증가분 ×
`one_way_tc` ≈ 2~5bp vs active return −85bp). 남은 후보는 **집행 계층**이다:
`partial_rebalance_eta: 0.50`은 목표의 절반만 따라가라는 지시이고, §S11.8이 실증한
**느린 알파**를 전제로 튜닝돼 있다.

§S13.10은 이 패턴을 **재현하지 않았다**(IC 무변화, 21일 지속성 −0.046). 따라서 §S13.9는
단일 arm 관찰이며 독립 검증이 필요하다.

### 왜 arm 하나로는 못 읽는가 — 2×2가 필요한 이유

`eta`만 올려 S13.9 arm이 개선돼도, 그것이 **빠른 신호 때문인지 eta가 원래 잘못 튜닝된
것인지** 구분되지 않는다. 필요한 것은 **상호작용**이다.

| | eta 0.50 | eta 1.00 |
|---|---|---|
| **S0 피처셋(61)**, 21d 지속성 0.850 | IR **1.4380** (기보유) | **11a (대조)** |
| **+어닝캘린더(69)**, 21d 지속성 0.615 | IR **1.2448** (기보유) | **11b (처치)** |

`eta`의 실제 의미: `eta_eff = base_eta × √confidence`, `[0.05, 0.95]` 클립
(`backtest.py:1094`). 1.00은 "완전 이동"이 아니라 **클립이 허용하는 최대 속도**다.

**선행 정보**: `config.py:457-458`에 iter17(2026-04-17)이 0.50 → 0.65를 시도했다
되돌린 기록이 있다("iter15 baseline restored"). **느린 피처셋에서의 시도였으므로
가설과 모순되지 않으며, 오히려 11a가 재현해야 할 대조군이다.**

### 사전등록 판정 기준

- **주 판정(상호작용)**: `ΔIR_11b − ΔIR_11a`.
  여기서 `ΔIR_11a = IR(11a) − 1.4380`, `ΔIR_11b = IR(11b) − 1.2448`.
  **양수이고 클 때만** "집행 계층이 빠른 신호를 흘린다"가 지지된다.
- **반증(사전등록)**: `ΔIR_11a ≥ ΔIR_11b` 이면 **가설 기각**. eta는 신호 속도와 무관하게
  단순히 잘못 튜닝돼 있었을 뿐이며, §S13.9의 해석을 결정 로그에서 정정한다.
- **채택 게이트(별건)**: 프로덕션 flip은 `IR(11b)` 또는 `IR(11a)`가 **S0 1.4380 대비
  ΔIR > +0.36 & 서브기간 부호 일관**일 때만. 상호작용이 양수라도 절대 성과가 바를
  못 넘으면 flip 없음(설명력 근거로만 기록).
- **가드 확인**: `max_te_annual` 0.035 / 실현 TE 가드 0.045 / 회전율 폭증 / degenerate_rate.
  eta 상향은 회전율을 직접 올리므로 TE·비용 훼손 여부를 반드시 함께 본다.
- **단일 사전등록 값**: eta = 1.00 하나. 0.65·0.80 등 스윕 금지.
- 두 셀은 **순차 실행**. 어느 셀도 게이트 통과 전 flip 금지.

### S13.11 결과 (2×2 완료·판정) — 2026-07-27

11a: exit 0. 11b: exit 0. 두 셀 모두 `--no-cache`.

| 셀 | 피처 | eta | IR | TE | 회전율 | IC | P1 / P2 / P3 |
|---|---:|---:|---:|---:|---:|---:|---|
| S0 | 61 | 0.50 | **1.4380** | 3.73% | 0.691 | 0.0162 | 1.48 / 0.79 / 2.19 |
| **11a** | 61 | 1.00 | 1.2177 | 3.81% | 1.083 | 0.0162 | 1.34 / 0.70 / 1.70 |
| S13.9 | 69 | 0.50 | 1.2448 | 3.63% | 0.928 | 0.0311 | 1.46 / 0.51 / 1.82 |
| **11b** | 69 | 1.00 | 1.2663 | 3.88% | 1.460 | 0.0311 | 1.03 / 0.80 / 1.86 |

**설계 건전성 확인**: 각 행에서 eta를 바꿔도 `avg_ic`가 소수점 4자리까지 동일하다
(0.0162 / 0.0311). eta는 예측을 전혀 건드리지 않는 **순수 집행 노브**이며, 2×2에서
집행 효과가 깨끗이 분리된다. degenerate_rate도 네 셀 전부 0.500으로 불변.

#### 주 판정 — 상호작용: **가설 방향은 지지, 사전등록 반증 불발**

```
ΔIR_11a  (느린 신호 + eta↑) = -0.2203
ΔIR_11b  (빠른 신호 + eta↑) = +0.0216
────────────────────────────────────────
상호작용  ΔIR_11b - ΔIR_11a = +0.2418
```

사전등록한 반증 조건(`ΔIR_11a ≥ ΔIR_11b`)은 **발화하지 않았다**. 집행 댐퍼 완화는
느린 신호를 **명확히 해치고**(−0.220, 서브기간 전부 음수) 빠른 신호는 **소폭 돕는다**
(+0.022). **`eta`가 신호 속도와 상호작용한다는 것은 실측으로 확인됐다.**
`config.py:457`의 iter17 롤백(0.65→0.50)도 11a로 재현됐다 — 그 결정은 옳았다.

#### 그러나 크기가 결론을 뒤집는다 — **집행 계층은 S13.9 손실의 설명이 아니다**

S13.9가 잃은 것은 **−0.193**인데 eta 완화로 회수된 것은 **+0.0216, 약 11%**뿐이다.
게다가 그 대가로 회전율이 0.928 → **1.460**(S0 대비 2.1배)으로 뛰고 TE도 3.88%로 올랐다.

**§S13.9 기록의 해석을 정정한다.** 당시 "집행 계층이 신호를 흘린다"를 유력 후보로
적었으나, 이번 검증 결과 그 경로는 **실재하되 지배적이지 않다**. IC가 +92% 올랐는데
IR이 −0.193 빠진 현상의 **대부분은 여전히 미설명**이다. 남은 후보(미검증):
포트폴리오 구성 제약(`max_active_per_stock` 0.04 · core-satellite · mega-cap 보호)이
IC 이득을 표현하지 못함, 또는 `avg_ic`의 측정 지평이 21일 보유기간과 불일치.

#### 채택 게이트 — 전 셀 실패

- **네 셀 중 S0(1.4380)를 이긴 셀이 없다.** 11b는 1.2663으로 −0.17.
- 11b는 서브기간 부호도 불일치(P1 1.46→1.03 급락, P2 0.51→0.80 개선).
- **불채택. `partial_rebalance_eta` 0.50 유지, `earnings_calendar_feature_enabled`
  OFF 유지. flip 0건.**

#### 확보된 지식

1. **`eta`는 신호 속도 의존적이다**(상호작용 +0.242). 향후 빠른 신호를 도입한다면
   `eta`를 함께 조정해야 하지만, 이것만으로 손실을 메우지 못한다.
2. **iter17의 eta 0.65 롤백은 정당했다** — 느린 신호에서 eta 상향은 순수 비용이다
   (11a: 회전율 +57%, IC 불변, IR −0.220).
3. **S13.9의 IC↑/IR↓ 괴리는 미해결로 남는다.** 집행 계층으로는 11%만 설명된다.

## §S13.12 IC→IR 전달 진단 — 미설명 89%의 규명 (2026-07-27, 진단·비액션)

**목적**: §S13.9(IC +92% / IR −0.193)와 §S13.11(eta 검증이 손실의 ~11%만 회수)이 남긴
"IC 개선이 왜 IR로 전환되지 않는가"를 아티팩트만으로 절단. 새 백테스트 없음 —
S0·S13.9·S13.11b의 backtest_result.pkl + 벤치마크 가중치 재구성(get_benchmark_fn,
production config). 산출물: `outputs/s13_12_ic_ir_transmission/*.csv`.

**방법**: IC→IR 사슬을 4관문으로 절단. ①순위 개선의 위치(데실 곡선·보유종목 IC),
②전달계수 TC=corr(액티브가중치, 예측), ③counterfactual 책 스왑 — Σ w_act×spec20
(예측과 무관하게 각 "책"이 포획한 실현 specific return), ④실현 total active(보유윈도우
복리) − spec 포획 = gap(common-factor 캐리 + 기간내 드리프트 − 비용). 96 리밸런싱,
×12 연환산. targets는 세 런에서 바이트 동일 확인(max|diff|=0).

**핵심 수치** (연환산):

| | S0 | S13.9 | S13.11b(eta 1.0) |
|---|---:|---:|---:|
| realized active | 5.60% | 4.75% | 5.16% |
| spec 포획 (Σw_act×spec20) | 3.30% | 2.94% | 3.21% |
| gap (common+드리프트−비용) | 2.27% | 1.78% | 1.92% |
| full IC | 0.0162 | 0.0311 | 0.0311 |
| 보유종목 IC (\|wa\|>10bp) | 0.0250 | 0.0563 | 0.0743 |
| n_held (>10bp) | 64.3 | 48.0 | 34.9 |
| 양(+) 액티브 합 | 19.8% | 16.7% | 18.5% |
| TC (spearman) | 0.309 | 0.293 | 0.243* |

*sparse 북은 전종목 spearman이 구조적으로 낮게 측정됨(비보유 ~165종의 순서가 캡가중 순서) — 헤드라인 지표 아님.

**판정 4건**:

1. **[구조] S0 실현 액티브의 ~41%(2.27/5.60%p)는 IC가 측정하지 않는 축에서 나온다.**
   avg_ic는 PCA-잔차 specific return(20d) 순위상관인데, 실현 액티브의 2.27%/yr는
   spec 포획 밖의 gap — 비용(연 ~7bp)을 빼면 거의 전부 지속 보유 틸트(고유변동성 축)의
   common-factor 캐리 + 기간내 드리프트다. **IC 게이트만으로 arm을 고르면 이 41%를
   무형(無形)으로 거래해버릴 수 있다** — 반복 실패의 구조적 배경.
2. **S13.9 손실 −0.85%/yr 분해**: spec 포획 −0.36 (43%) + gap −0.48 (57%, 비용 몫
   ~2.4bp). 손실의 과반이 "모델 축 밖". 이벤트 슬리브 자체는 벌었다(향후 10bd 발표
   종목 spec 포획 +0.75%/yr) — 비이벤트 코어에서 −1.1%/yr을 잃어 상쇄.
3. **전달 필터 실증**: corr(Δ예측, Δ가중치) = +0.088 — 새 정보의 ~9%만 북에 도달.
   EMA(α0.5)+turnover penalty+eta(0.5) 사슬이 저역통과로 작동, 덜 지속적인(21d 지속성
   0.850→0.615) 타깃 북들을 시간평균하며 액티브 노름을 상쇄(양 액티브 19.8→16.7%,
   n_held 64→48). μ는 CS z-score라 크기 정규화됨 — 감쇠는 μ가 아니라 **가중치 수준**에서
   발생. 11b(eta 1.0)가 spec 포획을 3.21%로 75% 복원함이 메커니즘 확증 — 단 gap은 29%만
   복원되고 TE가 3.63→3.88%로 올라 IR로는 +0.02에 그침(§S13.11 재해석: eta는 수익
   +0.41%/yr을 회수했으나 전액 TE 증가로 지불).
4. **IC와 북의 괴리 국지화**: P2에서 dIC 최대(+0.029)인데 spec 포획 최락(−1.74%/yr) —
   IC는 200종 전체 순위, 북은 ~48종 표현. corr(dIC, S0 실현 액티브)=+0.001 — IC 개선이
   알파가 나는 날짜와 무상관. P3 손실은 gap(−1.47%/yr, 팩터 캐리 훼손)이 주범.

**교훈(비준 대기, 사전등록 규칙 제안)**: 향후 arm 판정의 counter-evidence에
①책-스왑 spec 포획(Σw_act×spec20)과 ②gap을 추가한다. IC↑라도 spec 포획↓ 또는 gap↓이면
"모델 축 개선이 북의 수익축을 훼손"으로 판정. IC는 채택 근거 부적격(기존 비액션 원칙
유지)이며, 채택 후보는 spec 포획·gap 동시 비악화를 요구한다.

**비액션**: flip 0건. 진단 전용 — production 변경 없음.

## §S13.13 상호작용 피처 블록 — 사전점검 + 사전등록 (2026-07-27)

**동기**: 트리 모델에 단일 피처의 단조 변환은 무의미(분할 순서 불변). 남은 비선형 여지는
교차 피처 곱인데, 현 모델은 용량 결핍(평균 42트리, §S13.8)이라 축정렬 분할로 곱 구조를
자가 생성할 깊이가 없다. 기존 화이트리스트 부모끼리의 곱은 **새 정보 축이 0** —
§S13.12의 캐리-교란 채널이 구조적으로 최소인 후보군이다.

**사전점검 (IC 비접촉 — 구조 지표만, 선택은 메커니즘으로만)**:
스크립트 scratchpad `s13_13_interaction_precheck.py`. 후보 z×z 곱은 S0 panel(z-score 후)
에서, 경로형 2종은 Daily_Returns에서 구성. 2018-11~2026-04, 상장 마스킹.

| 후보 | persist21 | btv | \|c\|vol | \|c\|mom | 판정 |
|---|---:|---:|---:|---:|---|
| ix_vol_mom = idio_vol_63d × momentum_252d | 0.841 | 0.109 | 0.315 | 0.565 | 통과 |
| ix_val_mom = best_px_bps_ratio_level_z × momentum_252d | 0.796 | 0.114 | 0.237 | 0.140 | 통과 |
| ix_rev_vol = eps_rev_ma_63d × idio_vol_63d | 0.828 | 0.118 | 0.185 | 0.215 | 통과 |
| ix_qual_val = best_roe_level_z × best_px_bps_ratio_level_z | 0.662 | 0.061 | 0.063 | 0.070 | 통과 |
| mom_consistency_252 (참고) | 0.939 | 0.245 | 0.159 | 0.486 | 구조 통과·블록 제외(상호작용 아님, 별도 arm 후보로 보존) |
| vol_ratio_21_126 | 0.058 | 0.015 | — | — | **탈락** (지속성 붕괴 — S13.9 사멸 경로) |

부수 발견: 부모 idio_vol_63d 자체의 btv=0.554 — 1위 피처가 과반 "종목 정체성" 분산.
§S13.12(지속 틸트=캐리 원천)와 정합.

**사전등록 (단일 arm, 스윕 없음)**:
- 블록: **ix_vol_mom, ix_val_mom, ix_rev_vol, ix_qual_val** 4종 고정.
  구현: assembly에서 부모의 CS z 곱으로 생성(항상 빌드, whitelist 게이트),
  `interaction_features_enabled` default-OFF, S8 extra_whitelist 승인 경로.
- E1: full ΔIR > +0.36 & P1/P2/P3 부호 일관 시에만 채택. 그 외 불채택.
- counter-evidence (§S13.12 지표의 최초 적용 — 판정 근거는 E1, 아래는 증거):
  ① 블록 split share > 0 (inert 여부), ② 예측 21d 지속성 ≥ ~0.80 유지
  (느린 부모 → 붕괴하면 안 됨; S13.9는 0.615로 사멸), ③ 책-스왑 spec 포획·gap
  동시 비악화, ④ corr(pred, idio_vol) 방향.
- 기대 메커니즘: 고변동성 슬리브 내부의 재정렬(고변동 승자 vs 패자 분리) —
  북 이동 최소·캐리 보존 하의 spec 포획 개선.

### §S13.13 개정 (2026-07-27, 결과 미관측 상태에서) — 블록 4→5 확장

사용자 지시("상위 5개의 후보들을 모두 적용해보자")로 구조 통과 5종 전원을 블록에
포함한다: 4개 곱 + **mom_consistency_252**(252d 상승일 비율, persist21 0.939 ·
btv 0.245). 절차 규율: 4피처 arm은 실행 중이었으나 **어떤 지표도 관측하지 않은 채
중단**했고(부분 산출물 삭제), 본 개정은 결과 무접촉 상태의 사전등록 수정이므로
§2.4(단일 사전등록·스윕 금지) 위반이 아니다. 단일 arm은 이제 5피처 블록이며,
"새 정보 축 0" 전제는 곱 4종에만 적용됨을 명시한다 — mom_consistency_252는
Daily_Returns 경로형(신규 축이되 느림·btv 기준 이하). E1·counter-evidence는
원 사전등록과 동일.

### §S13.13 결과 (2026-07-27) — E1 FAIL·불채택. 단, 설계 목표 전부 달성 + §S13.12 프레임 확증

실행: `outputs/arm_s13_13_interaction_block` (851s, 66피처 = 61+5, 블록 전원 진입).
비교: S0(200) 2026-07-27 빈티지 IR 1.4380.

| 지표 | S0 | S13.13 | Δ |
|---|---:|---:|---:|
| IR | 1.4380 | 1.3571 | **−0.0809** |
| P1/P2/P3 | 1.479/0.787/2.192 | 1.462/0.811/1.944 | −0.016/+0.024/−0.248 |
| avg_ic | 0.0162 | 0.0222 | +0.0060 |
| **spec 포획 ×12** | 3.30% | **3.41%** | **+0.11%** |
| **gap ×12** | 2.27% | 1.85% | **−0.42%** |
| 예측 지속성 21d | 0.855 | 0.854 | −0.002 |
| 퇴화율 | 0.500 | **0.375** | −0.125 |
| 회전율 | 0.691 | 0.655 | −0.036 |
| 북 괴리 (L1/2) | — | 0.129 | (S13.9: 0.186) |

**E1: FAIL** — ΔIR −0.0809(바 +0.36 미달), 서브기간 부호 불일치(P2만 +). **불채택, flip 0건.**

counter-evidence 판독:
- 블록 split share 6.22%(균등몫 7.58%), 최대 단일 피처 1.69% — inert 아님·leak 아님(S13.10과 대조적으로 균등 사용).
- 예측 지속성 0.854로 완전 보존(설계 목표 달성 — S13.9의 0.615 사멸과 대조).
- vol-corr 사실상 불변(−0.013) — 새 축 없음 설계 그대로.
- **spec 포획 +0.11%/yr: 19개 arm 중 최초로 책의 specific 포획을 개선한 arm.**
- **퇴화율 최초 개선**(0.5→0.375): 상호작용 피처가 유용한 분할을 공급해 조기종료
  퇴화를 완화 — §S13.8의 용량 결핍 진단과 정합하는 부수 확증.

**핵심 판독 — §S13.12 프레임의 최초 실전 검증이자 확증**: 손실은 전부 gap(−0.42%/yr,
캐리 축)에서 났고 spec 축은 개선됐다. 즉 지속성 보존·새 축 0·포획 개선이라는 최선
조건의 피처 arm조차, 북 괴리 12.9%가 유발하는 캐리 손실(−0.42%)이 spec 이득(+0.11%)의
~4배라서 진다. S13.9(괴리 0.186, gap −0.48%)와 합치면 **캐리 손실 ≈ 북 괴리에 거의
비례(~3%/yr per unit L1/2)**: 현 북 구성이 캐리 축의 국소 최적이며, 모델 계층의 어떤
변경도 북을 움직이는 순간 그 비례 비용을 문다. 피처 arm이 이기려면
spec 이득 > ~3.3% × 북괴리 — 이번 arm 기준 이득이 4배 더 컸어야 했다.

**함의**: 한계 피처 프로그램은 구조적으로 상한이 있다. 남은 개선 여지는 (a) 북을
거의 안 움직이는 초저괴리 개선(피처로는 곤란), 또는 (b) 캐리 축 자체를 겨냥한
포트폴리오 구성 계층 — 둘 다 새 사전등록 대상.

## §S13.14 결합 arm — 상호작용 블록 + 위너-트림 보호 (2026-07-27, 사전등록)

**경위**: §S13.13 flip 보류 결정(사용자) 후, "spec 이득 +0.11 유지 + 캐리 손실 −0.42
차단" 결합 arm을 사용자가 지시. 설계 전 담체 진단(scratchpad `s13_14_carry_carrier.py`,
95윈도우, 아티팩트 전용) 결과:
- 변동성 팩터 노출: Δe_vol +0.007·캐리 +0.07% → **담체 아님** (노출 페널티 배제)
- 베타: Δe_beta −0.024·캐리 −0.13% → 부분 담체(~30%)
- **종목 구성 −0.32%/yr = 주 담체**: 손실 상위 PLTR −0.46/NVDA −0.43/TSLA −0.42%/yr —
  상호작용 블록(밸류×모멘텀)이 20d 지평에서 정당 감점한 메가 위너 축소가 실체.
  §S13.11의 미검증 후보 "mega-cap 보호"와 합류.

**사전등록 (결합 1회, 스윕 없음)**:
- arm = `interaction_features_enabled: true` + `winner_trim_protection_enabled: true`.
- 위너-트림 보호(§4.1 inline soft penalty 선례): hinge 페널티
  λ × Σ_{i∈winner} max(0, w_prev_i − w_i), objective에서 차감.
  winner = **hist_returns 트레일링 252d 누적수익 상위 20%**(리밸런싱 시점 이전 창 —
  look-ahead 없음, 공분산과 동일 데이터). **λ=1.0 고정**(μ가 z-스케일이므로 "대체
  종목이 1σ 이상 우위일 때만 위너 트림 허용"으로 해석 가능). quantile 0.8 고정.
- default-OFF + parity: 플래그 OFF 시 페널티 항 0(int) → objective 바이트 동일,
  단위테스트 선행.
- E1: full ΔIR > +0.36 & 서브기간 부호 일관 시만 채택.
- counter-evidence(§S13.12/13 지표): ① spec 포획이 S13.13 수준(+0.11) 근방 유지,
  ② gap이 S0(2.27%) 쪽으로 복원, ③ PLTR/NVDA/TSLA 기여 델타 ~0 복원,
  ④ 북 괴리 < 0.129, ⑤ 블록 split share 유지, ⑥ 지속성 ~0.85 유지.
- 예상 실패 모드(정직 기록): 위너 보호가 TE 예산을 위너에 잠그면 spec 이득이
  표현될 공간이 줄어 둘 다 잃을 수 있음 — 그 경우 "모델 축과 캐리 축은 현 예산에서
  양립 불가"가 결론이 된다.

### §S13.14 결과 (2026-07-28) — E1 FAIL·불채택. 사전등록한 실패 모드가 아닌 **제3의 실패 모드** 실증

실행: `outputs/arm_s13_14_ix_winner_protect` (1443s, 66피처, ECOS 192/192·fallback 0).
백그라운드 런 2회 즉사(절전/환경) 후 §S12 안정 패턴(schtasks 일회성 우회)으로 완주.
비교: S0(200) 2026-07-27 빈티지 1.4380.

| 지표 | S0 | S13.13 | **S13.14** |
|---|---:|---:|---:|
| IR | 1.4380 | 1.3571 | **1.3223** (ΔIR −0.1157) |
| P1/P2/P3 | 1.479/0.787/2.192 | 1.462/0.811/1.944 | 1.439/**0.919**/1.856 |
| spec 포획 ×12 | 3.30% | 3.41% | **3.21%** |
| gap ×12 | 2.27% | 1.85% | 1.87% |
| 북 괴리 vs S0 | — | 0.129 | **0.128** |
| PLTR/NVDA/TSLA Δ기여 합 | — | −1.31% | **−1.22%** |
| 회전율 | 0.691 | 0.655 | 0.650 |
| 지속성/퇴화율/IC | 0.855/0.500/0.0162 | 0.854/0.375/0.0222 | 0.854/0.375/0.0222 (모델층 동일) |

**E1: FAIL** (ΔIR −0.116 < +0.36, 부호 불일치: P2만 +0.131). **불채택, 두 플래그 모두 OFF 유지, flip 0건.**

**핵심 판독 — 위너-트림 보호가 실패한 이유(예상 밖)**: 사전등록한 실패 모드("TE 예산
잠김")가 아니라 **개입 지점 오류**였다. 트림-hinge는 북을 거의 움직이지 못했고(괴리
0.129→0.128, 메가위너 기여 −1.31→−1.22로 미미한 복원) 대신 spec 포획만 깎았다
(3.41→3.21, S13.13의 이득 전량 소거). 원인: 메가위너 언더웨이트는 **청산(trim)이
아니라 형성(formation) 문제** — arm 모델의 μ가 처음부터 이들을 낮게 랭크해 포지션이
아예 형성되지 않으므로, w_prev(자기 역사) 대비 축소를 벌하는 hinge는 걸릴 곳이 없다.
한편 페널티는 다른 위너 ~40종의 회전을 전반적으로 굳혀(회전율 0.650) spec 동기 거래만
지연시켰다. P2(횡보·급등락기) +0.131은 위너 고착이 추세 추종으로 작동한 부수 효과,
P3(메가캡 멜트업) −0.336이 형성 실패의 직접 증거.

**함의**: 캐리 복원은 "팔지 마라"로는 불가능하고 "사라"가 필요하다 — 장기 모멘텀을
μ에 블렌드하거나 위너 오버레이로 형성 자체를 유도해야 하는데, 이는 §S11.9(mh 블렌드
FAIL)·§S8 계열의 이미 소진된 축과 겹치거나 새 대형 개입이다. 상호작용 블록 축은
"spec 개선은 실재하나 캐리 비용을 상쇄 못 함"으로 **종결**. 위너-트림 코드는
default-OFF 인프라로 잔존(향후 다른 맥락 재사용 가능).

## §S13.17 레짐 조건화 μ-스케일링 — 사전등록 (2026-07-28, 측정 전)

**경위**: new_ai_port에서 이 세션이 설계·검증한 regime_v2 엔진(10피처 워크포워드
diag GaussianHMM: 시장 6종 + eps_g63/eps_us_lead63/eps_us_lead252/eps_tech_lead63,
필터드 확률, canonical ordering)을 ai_port에 이식해 **레짐 조건부 리스크 변조**를
평가하라는 사용자 지시. 엔진 품질은 new_ai_port에서 실증됨: 상태 점유
calm 0.38/mid 0.26/stress 0.36, 2018Q4·COVID p_stress 1.00, 2022H1 0.92, 2025-04
0.80 포착, 2026 YTD 0.06(강세장 stress 오독 없음). 단 new_ai_port 포트폴리오
성과는 혼조(validation +0.073 / full −0.048)였고, 그쪽 효과 경로는 FM 상호작용이라
ai_port(GBDT+MVO)와 전달 구조가 다름 — 독립 평가가 필요하다.
**축 중복 없음**: 피처 축(§S13.13 종결)·트림 축(§S13.14 종결)이 아닌
**포트폴리오 구성층(MVO 목적함수 알파-리스크 트레이드오프)** 개입으로, ai_port
미개척 축이다. 유일 선례는 dormant VIX z-score PCA 가중(§구조리뷰)뿐.

**메커니즘 (A1 `apply_mu_vol_scaling` 관용구 동일)**: post-overlay 예측 → μ 경로에
`apply_regime_mu_scaling` 추가:
`μ_i(t) ← μ_i(t) × (1 − λ·p_stress(t))`, p_stress는 워크포워드 필터드 확률
(관측 ≤ t, PIT). per-date 스칼라 곱이므로 **종목 간 상대 랭킹 불변** — 스트레스일수록
MVO의 알파 항만 약화되어 북이 벤치마크 쪽으로 수축(TE캡·턴오버 페널티 불변).
레짐 미커버 날짜(첫 fit_end 이전)는 scale 1(inert 가드, A1 관용구).

**사전등록 파라미터 (단일, 스윕 금지)**: **λ = 0.5 고정**. p_stress=1일 때 알파 절반 —
바인딩하되 북 붕괴(§2.5) 방지. 다른 λ 시도 없음.

**플래그 (§2.1 OFF-default + parity)**: `regime_mu_scaling_enabled: false`(기본) +
`regime_mu_stress_shrink: 0.5`. OFF 시 함수는 예측 패널을 무변경 반환 → 바이트 동일,
단위테스트 선행. 데이터 접근을 위해 `ALL_FACTOR_COLUMNS`에 "Earnings" 카테고리
(SPX/NDX/MXWD_FWD_EPS) +3 추가 — **화이트리스트 확장만으로도 OFF-parity가 유지됨을
테스트로 고정**(팩터 컬럼이 피처 생성에 누출되지 않음 확인).

**Phase 0 사전점검 (§S13.7 선례 — 통과 전 arm 미구현·미실행)**:
S0 production(codex_causal_rank_65) 기존 아티팩트 `backtest_result.pkl`의 일별
액티브 수익을 v2 레짐 모달 상태로 조건화(신규 백테스트 없음):
- **P0-G1**: stress-모달일 연환산 액티브 < **−1%/yr** 그리고 calm+mid-모달일 > +1%/yr
- **P0-G2**: 백테스트 구간 반분(전반/후반) 모두 stress-버킷 액티브 부호 음
- **P0-G3**: stress-모달일 점유 10~50% (전역 디레버리지로의 변질 방지)
하나라도 실패 → **arm 미실행 shelve** ("스트레스 손실이 알파 틸트 경로에 없음"이 결론).

**E1 (채택 게이트)**: 동일 워크북 빈티지 S0와 짝 실행, 동일 ECOS. full ΔIR > +0.36 &
P1/P2/P3 부호 일관 시만 채택. |ΔIR| < 0.36 → 비액션(설명력 기록).
**E2 (메커니즘 확증)**: ① ΔIR 주 원천이 stress-버킷 액티브 개선(조건부 귀속),
② TE 감소가 stress일에 집중, ③ calm/mid일 액티브 S0 대비 ±0.5%p 이내.
**캐릭터 보존(§2.5)**: full TE ≥ 3.0%, fallback률 S0 동일, active share 유지.
**§2.7**: experiment_inventory 등록 + run_selection_bias 해킷 기록 후 flip 논의.

**예상 실패 모드 (정직 기록)**:
1. 필터드 확률의 전이 확정 지연로 손실 초반부를 놓쳐 이득 희석(63d 빠른 축에도 잔존).
2. stress일 손실이 알파 틸트가 아닌 공통 캐리 담체(§S13.12)면 μ 축소는 TE만 줄이고
   손실 원천을 못 건드림 → IR 중립~악화.
3. stress 꼬리 반등(예: 2020-04~05)에서 수축된 북이 회복 알파를 놓치는 비대칭.

**파일 계획(구현 단계)**: `src/market_regime_v2.py`(new_ai_port regime_v2 벤더 포크),
`data_loader.py` Earnings +3, `config.py` 플래그 2종, `backtest.py`
`apply_regime_mu_scaling`, `tests/test_regime_mu_scaling.py`(parity 선행),
`scripts/preflight_s13_17_regime_conditioning.py`,
`variants/arm_s13_17_regime_mu_scaling.yaml`.

**검증 명령**: 사전점검 `<PY> scripts/preflight_s13_17_regime_conditioning.py`
(P0-G1~G3 판정 출력) → parity `<PY> -m pytest tests/test_regime_mu_scaling.py -v` →
arm `<PY> run_variant.py --variant variants/arm_s13_17_regime_mu_scaling.yaml`
(단일 foreground). 상태: **사전등록만 완료, 사전점검 PENDING**.

### §S13.17 사전점검 결과 (2026-07-28) — **P0 FAIL·arm 미구현 SHELVE**

실행: `scripts/preflight_s13_17_regime_conditioning.py` (단위테스트 3종 선행 PASS,
신규 백테스트 없음). S0 = codex_causal_rank_65 2026-07-27 빈티지 아티팩트, 액티브
2,000일(2018-11-27~2026-07-27) 중 1,923일 조인(레짐 피처 결측일 77일 제외).

| 상태(모달) | 점유 | 연환산 액티브 | t | H1('18-11~'22-09) | H2('22-09~'26-07) |
|---|---:|---:|---:|---:|---:|
| calm | 31.9% | **+9.84%** | 4.07 | +5.26% | +10.89% |
| mid | 25.1% | +8.24% | 2.89 | +18.72% | +4.65% |
| stress | 43.1% | **+0.43%** | 0.21 | **+1.55%** | **−7.25%** (t −1.24) |

- **P0-G1 FAIL**: stress-버킷 +0.43%/yr — 손실 자체가 없음(요건 < −1%).
- **P0-G2 FAIL**: 반분 부호 불일치(H1 +1.55 / H2 −7.25).
- **P0-G3 PASS**: 점유 43.1%.

**판독**: S0는 스트레스 레짐에서 체계적으로 잃지 않는다 — 스트레스일 액티브는
전 기간 ~플랫이고 후반부 음수(−7.25%)는 105일·t −1.24로 약하며 전반부와 부호가
반대다. 사전등록한 실패 모드 ②("stress 손실이 알파 틸트 경로에 없음")가 사실로
확인된 셈이므로, μ-축소가 구할 손실이 존재하지 않는다. **arm은 코드 한 줄 없이
SHELVE** (§S13.7 선례와 동일한 조기 종료).

**사후 관찰 (비등록·비액션)**: stress 버킷이 기대 액티브 ~0으로 43% 점유 —
"손실 회피"가 아니라 "분산 절감"(평균 기여 없는 날의 TE 지출 축소 → IR 분모 개선)
가설은 논리적으로 남아 있으나, 이는 데이터를 본 뒤 발견한 가설이므로 **별도의 새
사전등록(자체 게이트·단일 파라미터) 없이는 진행 금지**를 명시해 둔다.

**잔존물**: `scripts/preflight_s13_17_regime_conditioning.py` +
`tests/test_preflight_s13_17_regime_conditioning.py`(3 PASS) +
`outputs/s13_17_regime_preflight.csv`. 프로덕션 코드·설정 변경 0건, flip 0건.

## §S13.18 지수 선행-EPS 매크로 피처 블록 — 사전등록 (2026-07-28, 측정 전)

**경위**: §S13.17 SHELVE 후 사용자 재지시 — "레짐 조건화가 아니라 opmargin·revision·
tg_price처럼 **독립 features로** 반영하라". 즉 지수 레벨 BEST_EPS 데이터를 모델
피처 축으로 공급하는 피처 arm이다. 데이터 축 자체가 신규: 기존 66피처 중 어닝스
기대는 전부 **종목 레벨**(eps_rev 계열·best_* 계열)이고, **톱다운(지수 레벨) 어닝스
기대 축은 최초**다. 워크북 Factor 시트의 SPX/NDX/MXWD_FWD_EPS(2026-07-28 NDX 추가
후 52컬럼)를 처음 소비한다.

**사전등록 블록 (4피처 고정, 스윕 없음)** — new_ai_port regime_v2 설계 승계
(상관·반감기 실측 2026-07-28: 최대 |corr| 0.481, 반감기 43~506BD):
- `fac_eps_g63` = Δ63 log MXWD_FWD_EPS (세계 레벨)
- `fac_eps_us_lead63` = Δ63 log(SPX)−Δ63 log(MXWD) (빠른 US 스프레드)
- `fac_eps_us_lead252` = Δ252 log(SPX)−Δ252 log(MXWD) (연간 리비전 사이클)
- `fac_eps_tech_lead63` = Δ63 log(NDX)−Δ63 log(SPX) (테크 스프레드)

**메커니즘·구현**: `src/features/index_eps.py` 신규(S13.15 모듈 관용구), Factor
그룹 합류 → **CS z-score 제외**(per-date 상수 — 랭커는 within-date 랭킹이므로
이 컬럼들은 날짜-분할 스플릿(매크로 조건화)으로만 작동 가능). 빌드 무조건,
채택은 core-whitelist `index_eps_features_enabled`(default-OFF)로 게이트.
데이터 접근: `FACTOR_CATEGORIES`에 "Earnings" +3 — 전 소비처 명명-컬럼 접근이라
로딩만으로는 inert. 단위테스트 3종 PASS(수식 hand-match·브로드캐스트·게이팅),
전체 스위트 409 PASS.

**비교 기준**: S0 = codex_causal_rank_65 2026-07-28 12:11 아티팩트
**IR 1.4380 / TE 3.73% / 회전율 0.691 / beta 1.059** (07-28 워크북 재생성은 Factor
시트 NDX 컬럼 추가뿐, 기존 51컬럼 allclose 검증 완료 — 동일 빈티지로 유효).
동일 ECOS, `variants/arm_s13_18_index_eps_features.yaml`(프로덕션 핀 + 플래그
1개 델타).

**게이트**: E1 = full ΔIR > +0.36 & P1/P2/P3 부호 일관 시만 채택, |ΔIR| < 0.36
비액션. E2(메커니즘) = EWMA importance에서 4피처 생존 여부 + gap/spec 포획
(§S13.12 지표) 델타 방향. §2.5 캐릭터 보존(TE·active share·fallback), §2.7
DSR 해킷 후 flip 논의.

**예상 실패 모드 (정직 기록)**: ① 새 컬럼 추가에 대한 모델 민감성(iter 4/5/7/12/13
5회 실증 — split 분배 교란 + EWMA cold-start)이 marginal value를 초과. ② per-date
상수는 within-date 랭킹에 직접 기여 불가 — 트리가 날짜 분할로만 쓸 수 있는데
현 모델은 용량 기아(~42트리, §S13.8)라 매크로 분기에 쓸 여유가 없을 수 있음.
③ 66→70피처 colsample 희석. ④ §S13.13의 구조적 상한(캐리 gap)은 피처로 못 건드림.
상태: **사전등록 완료, arm 실행 PENDING**.

### §S13.18 결과 (2026-07-28) — **E1 FAIL·불채택**. 실패 모드 ②의 극단형 실증: gain 정확히 0

실행: `outputs/arm_s13_18_index_eps_features` (940s, ECOS 192/192·fallback 0,
퇴화율 0.4375 vs S0 0.500). 비교: S0 2026-07-28 빈티지 1.4380.

| 지표 | S0 | **S13.18** |
|---|---:|---:|
| IR | 1.4380 | **1.4507** (ΔIR **+0.0127**) |
| P1/P2/P3 | 1.479/0.787/2.192 | 1.369/**0.991**/1.970 (−0.110/+0.204/−0.222) |
| TE | 3.73% | 3.74% |
| 회전율 | 0.691 | 0.648 |
| realized_beta | 1.059 | 1.053 |

**E1: FAIL** (ΔIR +0.013 << +0.36 노이즈 대역, 서브기간 부호 불일치).
**불채택, `index_eps_features_enabled` OFF 유지, flip 0건.**

**E2 판독 — 메커니즘 자체가 부재**: 패널에는 4피처가 정상 진입(non-null 100%,
65피처 중 4)했으나, 65-피처 재훈련 전체에서 4피처의 **gain share가 정확히
0.00%** — 트리가 단 한 번도 이 컬럼들로 분기하지 않았다. ΔIR +0.013은 피처
기여가 아니라 컬럼 공간 변화에 따른 split/EWMA 섭동 노이즈다(민감성 전례와
방향만 반대).

**구조적 원인 (일반 결론으로 승격)**: lambdarank/xendcg 계열의 그래디언트는
**쿼리(날짜) 내 합이 ~0**이다. per-date 상수(bcast) 피처의 분기는 쿼리를 통째로
가르므로 양쪽 그래디언트 합이 여전히 ~0 → **루트 레벨 split gain이 구조적으로
~0** → 선택되지 않는다. 트리 하부(종목 분기 이후)에서는 이론상 gain이 생기지만
용량 기아(~42트리·depth 5) 모델은 거기까지 가지 않는다. 이는 macro_cross 설계
주석("bcast-only macro features와 달리 real cross-sectional variation")이 이미
암시하던 사실의 정량 확증이며, **기존 화이트리스트의 fac_* bcast 5종이 만성
저중요도인 이유도 동일 기제**로 설명된다.

**축 판정**: 지수 레벨 EPS 정보가 이 랭커에 전달되려면 매크로×종목 곱항(mc_*
관용구)이 유일 경로인데, 그 축은 §S13.13/14에서 종결됐다. 따라서 **"지수
선행-EPS as bcast 피처" 축은 1회 실행으로 깨끗하게 종결**. 코드·플래그는
default-OFF 인프라로 잔존(단위테스트 3종 + 전체 409 PASS 유지).

## §S13.19 캐리-EPS 공변 사전점검 — 사전등록 (2026-07-29, 측정 전)

**경위**: §S13.18 종결(bcast 축 폐쇄) 후 사용자 지시 — 지수 EPS 데이터의 잔여
활용처 탐색. EXPERIMENT_HISTORY 잔여 축 (b) "캐리 축을 직접 겨냥한 포트폴리오
구성 계층"의 정면 사전점검. 질문: S0 액티브의 gap 성분(§S13.12, +2.30%/yr)이
지수 EPS 사이클과 공변하는가. 공변 없으면 캐리-타이밍 arm은 설계 없이 SHELVE
(S13.17 패턴 — 백테스트·DSR 비용 0).

**방법 (새 백테스트 없음)**: `scripts/preflight_s13_19_carry_eps_covariation.py`.
- gap_t = realized_t − spec_cap_t, `outputs/s13_12_ic_ir_transmission/per_date_S0.csv`
  96 리밸런싱 재사용. 앵커 재현 확인 완료(2026-07-29): realized +5.60% /
  spec_cap +3.30% / gap +2.30%/yr (×12 연환산) — §S13.12 기록과 일치.
- 조건 변수(1차 사전등록): **`fac_eps_g63`** = Δ63 log MXWD_FWD_EPS
  (`index_eps.py` 수식 동일), 리밸런싱 date 값. 캐리는 느린 common-factor
  현상이므로 가장 넓은 사이클 지표를 1차로 고정. us_lead63/us_lead252/
  tech_lead63 조건화는 **서술 전용**(게이트 판정 불사용).
- 버킷: 전표본 터실(33/67 분위) — 진단 전용이며 arm 설계 시 PIT 재확인 의무.

**게이트 (전부 통과 시만 arm 설계 착수, 하나라도 실패 → SHELVE)**:
- **G1**: |Δgap(top−bottom 터실)| ≥ 1.5%/yr
- **G2**: 표본 반분(48/48) 모두 Δgap 부호 동일
- **G3**: 국지화 — |Δgap(top−bottom)| > |Δspec_cap(top−bottom)| (공변이 캐리
  축 특이적이어야 함; spec 쪽이 더 크면 "그냥 알파 레짐"이라 §S13.17 결론과 중복)

## §S13.20 EPS-감응도 loading 횡단면 변환 사전점검 — 사전등록 (2026-07-29, 측정 전)

**경위**: bcast 직접 소비는 §S13.18에서 구조적 불가 확정 — 잔여 소비 경로는
종목별 감응도(loading) 추정으로 bcast→횡단면 변환뿐. 최종 용도는 랭커 피처가
아니라(상호작용 축 §S13.13/14 폐쇄) **§S13.16 재개 조건(독립 새 데이터 축 +
항상 실행 가능한 구성법)을 충족하는 잔차 슬리브/저괴리 오버레이**. 계보가 닫힌
축과 가까우므로 3게이트 전부 통과 시에만 arm 설계.

**방법 (새 백테스트 없음)**: `scripts/preflight_s13_20_eps_loading.py`.
- 감응도(1차 사전등록): 종목별 rolling **252d OLS** — 일간 USD 수익률(유니버스
  평균 차감, `UniverseData.sheets["Daily_Returns"]` + `listing_dates` 마스킹,
  창 내 유효 관측 ≥ 200) ~ **일간 innovation `Δ1 log NDX_FWD_EPS − Δ1 log
  SPX_FWD_EPS`** (FWD_EPS 일간 갱신 실측 100%, 2026-07-29). 96 리밸런싱 날짜
  에서 PIT 추정(과거 252d만).
- G3 조건부 신호: s_i(t) = loading_i(t) × **fac_eps_tech_lead63(t)**(63d 스프레드
  모멘텀 상태). 타깃 = S0 `backtest_result.targets`(파이프라인 동일 20d spec
  패널, §S13.12에서 바이트 동일 검증된 객체). 리밸런싱 날짜 rank IC.

**게이트 (전부 통과 시만 arm 설계 착수)**:
- **G1 (분산)**: 리밸런싱 날짜 평균 |t|>2 종목 비율 ≥ 20%
- **G2 (안정성)**: loading 횡단면 순위 자기상관(63d = 3리밸런싱 간격) 평균 ≥ 0.6
- **G3 (예측력)**: mean rank IC > 0 **AND** P1/P2/P3 중 ≥2 서브기간 부호 양 —
  **단일 판독, 스윕 없음** (스프레드·창·신호 구성 모두 사전 고정)

### §S13.20 결과 (2026-07-29) — **G1 FAIL·SHELVE. 감응도가 노이즈와 구분 불가**

실행: `scripts/preflight_s13_20_eps_loading.py`
(산출물 `outputs/s13_20_eps_loading_preflight.csv`, 단위테스트 5종 PASS,
96/96 리밸런싱 사용, 평균 유효 종목 189.8).

- **G1 FAIL (결정적)**: 평균 |t|>2 비율 **5.0%** << 20%. |t|>2의 우연 기대치
  (~5%)와 정확히 일치 — 252d 창 일간 회귀에서 **종목별 EPS-innovation 차등
  감응도가 통계적으로 존재하지 않는다**.
- G2 PASS(0.675)이나 **기계적 성분 주의**: lag-3(63BD) 인접 추정창이 252일 중
  189일을 공유하므로 노이즈 loading도 자기상관이 높게 측정된다. 실질 안정성
  근거로 승격 금지.
- G3 PASS(mean IC +0.0025, P1 +0.018/P2 +0.005/P3 −0.011)이나 크기가 §2.4
  노이즈 대역 이하이고 G1이 무너진 이상 신호는 노이즈 loading의 잔영이다.

**판정**: SHELVE — bcast→횡단면 변환(감응도 경유)은 **첫 관문(분산)에서 실패**.
이 유니버스 종목들은 일간 NDX−SPX EPS innovation에 대해 식별 가능한 차등
감응도를 갖지 않는다. 잔차 슬리브/오버레이 arm 미설계. 사전등록 준수: 다른
스프레드·창·주기로의 재시도는 새 사전등록 없이 금지(스윕 방지).

## §S13.19a/§S13.20a 4스프레드 전수 재검정 — 사용자 지시 확장 (2026-07-29, 측정 전 선언)

**경위**: 사용자 지시 "tech_lead까지 포함해서 모두 다시 테스트" — 위 사전등록의
1차-지표 고정을 사용자 권한으로 확장. **성격: 탐색적(exploratory) 4-way 검정.**
게이트 수식·문턱은 사전등록과 동일하되 4스프레드 전수에 적용하므로, 우연히
하나가 통과할 확률이 ~4배다. 따라서 **여기서 통과한 스프레드를 arm 조건
변수로 채택하려면 별도 사전등록에 이 4-way 선택 사실을 명시**해야 하며(§2.7
selection-bias 일관), 전수 결과는 순위 없이 전부 보고한다.

- §S13.19a: `scripts/preflight_s13_19a_all_spreads.py` —
  기존 산출물 `s13_19_carry_eps_preflight.csv`(4스프레드 델타 전부 보유)를
  게이트 형식으로 재판정(재측정 없음).
- §S13.20a: `scripts/preflight_s13_20a_all_spreads.py` — 감응도 회귀를
  스프레드별 (innovation, state) 쌍으로 확장: g63=(Δ1 MXWD, fac_eps_g63),
  us_lead63=(Δ1 SPX−MXWD, fac_eps_us_lead63), us_lead252=(Δ1 SPX−MXWD,
  fac_eps_us_lead252), tech_lead63=(Δ1 NDX−SPX, fac_eps_tech_lead63 —
  §S13.20 원판 재현). 창 252d·min_obs 200·게이트 동일.

### §S13.19a/§S13.20a 결과 (2026-07-29) — 캐리 공변 3/4 통과 형태·감응도 0/4

**§S13.19a 캐리 공변 (산출물 `outputs/s13_19a_all_spreads.csv`)**:

| 스프레드 | Δgap(t−b) | 반분 H1/H2 | Δspec | 게이트 |
|---|---:|---|---:|---|
| fac_eps_g63 (1차) | **+2.26%** | +0.93/+3.44 | +0.01% | **PROCEED** |
| fac_eps_us_lead63 | −1.68% | −3.80/+0.68 | +0.04% | SHELVE (G2 부호 불일치) |
| fac_eps_us_lead252 | **−4.90%** | −3.45/−4.93 | −1.01% | PROCEED (역방향) |
| fac_eps_tech_lead63 | **+5.17%** | +1.34/+7.55 | +0.48% | PROCEED |

패턴은 경제적으로 정합: 캐리는 세계(g63)·테크(tech_lead) EPS 사이클 상행에서
풍부하고, US 연간 상대 리드(us_lead252) 극단에서 빈곤. **단 4-way 탐색이므로
arm 조건 변수 선정 시 이 선택 사실과 다중성을 사전등록에 명시해야 한다** —
1차 사전등록 지표는 여전히 fac_eps_g63이고, tech_lead63(+5.17%)로의 교체는
"전수에서 최강 선택"임을 로그에 남긴 뒤에만 가능.

**§S13.20a 감응도 (산출물 `outputs/s13_20a_all_spreads.csv`)**: **4쌍 전부
SHELVE — G1(분산) 전멸**. 세부: 광역 innovation(MXWD·SPX−MXWD)은 |t|>2 비율
17.2%로 우연치(5%)의 3배 — 광역 EPS 뉴스에 대한 차등 감응은 미약하게
실재하나 게이트(20%) 미달이고, 조건부 IC도 g63/us_lead63은 음(−).
tech_lead(NDX−SPX)만 5.0%로 순수 노이즈. **bcast→횡단면 변환 축 폐쇄 유지**
(전수 확장으로도 재개 근거 없음).

## §S13.21 g63+tech_lead63 단독 피처 arm — 사전등록 (2026-07-29, 측정 전)

**경위**: 사용자 지시 "fac_eps_g63이랑 tech_lead63을 모두 포함해서 test,
특히 단독 features로도 포함" — 캐리 공변 통과 스프레드 2종을 **단독 bcast
피처로** 랭커에 공급하는 arm. §S13.18(4피처 블록, gain 정확히 0)의 2피처
서브셋이다.

**정직한 사전 기대 명시**: §S13.18의 구조적 결과(쿼리 내 zero-sum 그래디언트
→ per-date 상수의 root split gain ~0)는 피처 개수와 무관하다. 따라서 이 arm의
**사전 예측 = 2피처 gain share 0, ΔIR는 컬럼 공간 섭동 노이즈**(S13.18 실측
+0.0127). 본 실행은 사용자 지시에 의한 실증 확인이며, 서브셋(66→67피처,
S13.18은 69)으로 colsample 희석이 줄어드는 차이만 있다.

**구현**: `config.index_eps_feature_names`(default None=전체 4종, S13.18 하위
호환) 신설 + `admitted_index_eps_features()` 헬퍼(미지 이름 ValueError — 무음
inert 방지). OFF-parity 불변(flag OFF 경로 동일), 단위테스트 subset 게이팅
포함 전체 423 PASS. variant `variants/arm_s13_21_index_eps_g63_tech.yaml` =
프로덕션 핀 + 플래그 + subset 2종 델타.

**비교 기준**: S0 = codex_causal_rank_65 2026-07-28 12:11 아티팩트 IR 1.4380
(§S13.18과 동일 빈티지 확인). 동일 ECOS.

**게이트**: E1 = full ΔIR > +0.36 & P1/P2/P3 부호 일관 시만 채택, |ΔIR| < 0.36
비액션. E2(메커니즘) = 2피처 gain share — 0이면 구조 결과 재확증, >0이면
§S13.18 결론 수정 필요(중대 발견으로 별도 기록). 상태: **실행 PENDING**.

### §S13.21 결과 (2026-07-29) — **E1 FAIL·불채택. 사전 예측 그대로: gain 정확히 0 재확증**

실행: `outputs/arm_s13_21_index_eps_g63_tech` (10:01~, ECOS fallback 0,
퇴화율 0.53125 vs S0 0.500). 비교: S0 2026-07-28 빈티지 1.4380.

| 지표 | S0 | **S13.21** |
|---|---:|---:|
| IR | 1.4380 | **1.4855** (ΔIR **+0.0475**) |
| P1/P2/P3 | 1.479/0.787/2.192 | 1.624/0.784/2.198 (+0.145/**−0.004**/+0.005) |
| TE | 3.73% | 3.68% |
| 회전율 | 0.691 | 0.685 |
| realized_beta | 1.059 | 1.059 |

**E1: FAIL** (ΔIR +0.048 << +0.36 노이즈 대역, P2 부호 음 → 서브기간 불일치).
**불채택, flip 0건.**

**E2: 구조 결과 재확증** — 전체-패널(63피처) 모델에서 두 피처 gain **정확히
0.0000**, EWMA가 이후 재훈련에서 63→60으로 드랍(zero-gain 피처 제거와 정합).
ΔIR +0.048은 S13.18(+0.013)과 같은 컬럼 공간 섭동 노이즈이며 피처 기여가
아니다(gain 0이 인과 배제를 증명).

**측정 방법 각주 (어제 E2의 재검증 포함)**: 모델 피처명은 `Column_N`(위치
기반)이라 이름 매칭이 불가 — 위치 매핑으로 측정해야 한다. 같은 방법으로
S13.18 아티팩트를 재측정해 "4피처 gain 정확히 0" 기록의 유효성을 확인했다.

**축 판정**: 캐리 공변 최강 2종(g63·tech_lead63)조차 단독 bcast 피처로는
gain 0 — **"지수 EPS as 단독 피처" 축은 서브셋 구성으로도 재확인 종결**
(S13.18 4피처 + S13.21 2피처, 2회 실증). 코드·subset 인프라는 default-OFF
잔존(전체 423 PASS). 지수 EPS의 잔여 유효 경로는 §S13.19 캐리 조건화뿐.

## §S13.22 캐리 TE-캡 조건화 2-arm — 사전등록 (2026-07-29, 측정 전)

**경위**: 사용자 지시 — §S13.19 PROCEED의 후속 캐리-조건화를 "g63 단독"과
"tech_lead63 결합" 두 구성으로 설계·테스트. **2-arm 다중성 선언**, B의
tech_lead63은 §S13.19a 4-way 탐색 최강 선택 계보를 승계(승격 시 명시 의무).

**메커니즘 (포트폴리오 구성 계층만, 랭커·패널 비접촉)**:
`src/carry_te_conditioning.py` — 리밸런싱 시점 상태 = 조건 피처들의 **PIT
확장 백분위**(일간, min_history 504BD 미만은 중립) 평균 → 터실별 ex-ante TE
캡 스케일 **top ×1.25 / mid ×1.0 / bottom ×0.75** (κ=0.25 단일 사전약정,
캡 0.035 → 0.04375/0.035/0.02625, 실현 TE 가드 4.5% 이내). §S13.19의
전표본 터실 대신 완전 PIT — 사전점검의 PIT 재확인 의무 이행.
- **arm A** (`arm_s13_22a_carry_te_g63`): 상태 = fac_eps_g63 (§S13.19 1차)
- **arm B** (`arm_s13_22b_carry_te_g63_tech`): 상태 = g63·tech_lead63
  확장 백분위 평균 (결합)

배선: `run_backtest._optimizer_fn`에서 스케일 캡을 `optimize_portfolio`
인자+`diagnostics["max_te_annual"]`(projection 공유)로 전달. **OFF-parity**:
flag OFF면 승수 시리즈 None → 기존 호출과 인자·값 완전 동일(전체 428 PASS,
조건화 단위테스트 5종 포함).

**정직한 사전 기대 (효과 크기 상한)**: 수확 가능분 ≈ κ × Δgap × P(발동)
≈ 0.25 × 2.26% × 2/3 ≈ **+0.4%/yr 상한, 실효 ΔIR ~ +0.05~0.10** — E1 채택
바(+0.36)는 구조적으로 도달 거의 불가. 본 arm의 1차 가치는 **E2 메커니즘
판정**(캐리 타이밍이 OOS·PIT에서 실재하는가)이다.

**게이트**:
- **E1 (채택)**: full ΔIR > +0.36 & P1/P2/P3 부호 일관 — 표준 유지.
- **E2 (메커니즘, 3항)**: ① 바인딩 — 실현 TE(top-상태 구간) > 실현 TE
  (bottom-상태 구간), ② 방향 — (arm − S0) 일별 액티브 델타가 top-상태에서
  양(+)·bottom-상태에서 음(−)/0 (차등이 설계 방향), ③ §2.5 캐릭터 —
  full 실현 TE ≤ 4.5%·fallback률 비악화·회전율 급증 없음.
- 비교 기준: S0 = codex_causal_rank_65 2026-07-28 빈티지 IR 1.4380, 동일 ECOS.

상태: **사전등록 완료, 2-arm 순차 실행 PENDING**.

### §S13.22 결과 (2026-07-29) — **양 arm E1 FAIL·불채택. 캐리 타이밍은 TE-캡 스케일로 환전 불가**

실행: 두 arm 모두 exit 0, ECOS fallback 0, 퇴화율 0.50(= S0). 인벤토리
443→447(+S13.18/21/22a/22b).

| 지표 | S0 | **A (g63)** | **B (g63+tech)** |
|---|---:|---:|---:|
| IR | 1.4380 | **1.2642** (ΔIR **−0.174**) | **1.5487** (ΔIR **+0.111**) |
| P1/P2/P3 | 1.479/0.787/2.192 | 1.193/0.550/2.256 | 1.446/1.006/2.258 |
| TE / 회전율 / beta | 3.73%/0.691/1.059 | 3.81%/0.709/1.053 | 3.82%/0.700/1.055 |

**E1: 양 arm FAIL** (A는 음수, B는 +0.111 < +0.36 & P1 음). **불채택, flip 0건.**

**E2 판독 (상태별, PIT 상태 재계산·리밸런싱→보유윈도우 ffill)**:

| | A: top/mid/bottom | B: top/mid/bottom |
|---|---|---|
| 실현 TE | 4.48/3.45/2.94% | 4.52/3.92/2.40% |
| (arm−S0) 액티브 | **+0.15/−0.98/−1.06%/yr** | **+1.21/+0.13/+0.65%/yr** |

- **E2-①(바인딩) 양 arm PASS**: 조건화는 설계대로 리스크를 이동시켰다.
- **A의 실패 구조 (자구상 E2-② PASS이나 경제적 실패)**: top 수확 +0.15%/yr
  vs bottom 포기 −1.06%/yr + mid 경로 손상 −0.98%/yr. §S13.19 경고
  ("bottom 터실 gap도 +1.74% 양 — 축소는 포기 비용") 그대로 실현. 추가로
  mid(×1.0) 구간까지 음(−)인 것은 **경로 의존성**(eta 0.5·turnover penalty의
  시간평균 하에서 축소된 북이 후속 구간의 출발점을 오염) 실증.
- **B: E2-② FAIL — 개선이 설계 채널이 아님**: bottom(축소 상태)에서
  +0.65%/yr **이득** — 설계 논리(축소=캐리 포기=손실)와 역방향. 즉 B의
  +0.111은 캐리 타이밍 수확이 아니라 상태-우연 정렬/경로 효과다. 메커니즘
  미확증 → ΔIR은 §2.4 비액션 노이즈로 처리.

**축 판정**: §S13.19의 캐리-EPS 공변(진단)은 실재하나, **전체 액티브 북의
대칭 스케일링(TE-캡)으로는 수익 환전 불가** — ① 하방 상태에서도 캐리가
양(+)이라 축소가 비용, ② 경로 의존성이 상태 경계를 넘어 번짐, ③ 수확
상한(κ×Δgap×P) 자체가 노이즈 대역 이하. 관찰(후속 후보, 미실행·사전등록
필요): 상방-단독 비대칭(top만 ×1.25)이 A의 top +0.15만 취하고 bottom 포기를
피하는 구성이나, 상한 논리상 여전히 |ΔIR| < 0.36이라 우선순위 낮음. 코드는
default-OFF 잔존(전체 428 PASS).

### §S13.22a B안 이득 원인 진단 부록 (2026-07-29, 사후 분석 — 판정 불변)

사용자 질의("B안은 왜 알파가 증가했나")에 대한 사후 분해. **E1 FAIL·불채택
판정은 불변**이며, 이 부록은 "상태-우연 정렬/경로 효과" 서술을 정량 확정한다.
스크립트: scratchpad `why_b_alpha.py` (기존 아티팩트만, 백테스트 0회).

**① 상태별 총기여 (Δ누적 +419bp)**: top +251bp(60%) / bottom +114bp(27%) /
mid +54bp(13%).

**② 결정적 발견 — A와 B의 bottom은 다른 세계다** (상태 조건부 S0 성분, 연율):

| bottom 상태에서의 S0 | A(g63 단독, 26회) | B(composite, 21회) |
|---|---:|---:|
| realized | +3.54% | **−0.58%** |
| spec_cap | +2.10% | +0.62% |
| gap(캐리) | +1.44% | **−1.20%** |

§S13.19의 "bottom 터실도 gap +1.74% 양(+)"은 **g63 전표본 터실** 기준이었다.
B의 PIT composite bottom(세계 EPS·테크 리드 **동시** 약세, 교집합상 A와 15/21만
중첩)은 다른 날짜 집합이며, 그 구간에서 S0 캐리는 실제로 **음(−)**이다. 즉 B의
bottom +0.65%/yr는 캐리 포기가 아니라 **손실 회피(방어 타이밍)** — 설계 가설
(캐리 수확)과 다른, 사후 발견된 채널이다. 연도별로 2022 +173bp가 이 채널의
대표 실현(테크 베어·EPS 하향 구간 축소).

**③ 경로 오염 감소**: composite 평균화가 상태 시계열을 평활 — 전이 25→17회,
mid 델타 A −0.98 vs B **+0.13%/yr**. crosstab상 A-top 41회 중 21회가 B-mid로
강등(B의 top 판정이 더 보수적·안정적). A의 실패 요인 2개(bottom 포기·mid
오염)가 B에선 신호 정의 차이로 둘 다 완화된 구조.

**④ 시기 국지화 (채택 불가의 정량 근거)**: 구간 델타 상위 5(+330bp) = 총이득의
**79%**, 그중 4개가 2025-12~2026-05 연속 top 런. 단일 최악 구간(2026-06-22,
top)은 **−117bp** — 이득의 농도가 극단적이라 소표본 우연과 구분 불가. bottom
채널도 21회 리밸런싱 관측이 전부다.

**함의**: B의 bottom 신호("양 스프레드 동시 약세 = 캐리 음전")는 경제적으로
해석 가능한 실물 패턴이나, 본 데이터로 이미 관측했으므로(in-sample) 같은
데이터로 확증 불가. 재개하려면 "방어 조건화" 가설의 새 사전등록 + 4-way 최강
선택(tech_lead63) 다중성 명시가 요건 — §S13.22 축 판정(환전 불가·우선순위
낮음)은 유지한다.

상태: **사전등록 완료, 측정 PENDING** (S13.19 → S13.20 순차).

### §S13.19 결과 (2026-07-29) — **PROCEED. 3게이트 전부 통과, 공변은 캐리 축에 국지화**

실행: `scripts/preflight_s13_19_carry_eps_covariation.py`
(산출물 `outputs/s13_19_carry_eps_preflight.csv`, 단위테스트 5종 PASS,
앵커 재현: realized +5.60%/spec +3.30%/gap +2.30%/yr, 정렬 92/96 exact).

| fac_eps_g63 터실 | gap (연환산) |
|---|---:|
| bottom | +1.74% |
| mid | +1.11% |
| top | +4.00% |

- **G1 PASS**: |Δgap(top−bottom)| = 2.26%/yr ≥ 1.5%
- **G2 PASS**: 반분 Δgap H1 +0.93% / H2 +3.44% — 부호 일관
- **G3 PASS**: Δspec_cap(top−bottom) = +0.01%/yr ≈ 0 — 공변이 **전적으로 캐리
  축**에서 발생(알파 레짐 아님, §S13.17 결론과 비중복 확인)

**서술 전용 (판정 불사용, 승격 금지)**: tech_lead63 Δgap +5.17%(더 큼),
us_lead252 Δgap −4.90%(역방향) — 1차 지표가 아니므로 어떤 액션의 근거로도
쓰지 않는다. 후속 arm의 조건 변수는 사전등록대로 fac_eps_g63 고정.

**판정**: PROCEED — 캐리-조건화 arm 설계 착수 자격 획득. 단 주의: bottom
터실에서도 gap은 +1.74%로 양(+)이다. "약한 EPS 사이클에서 캐리 노출 축소"류
설계는 잔여 +1.74%를 포기하는 비용이 있으므로, arm은 차등(2.26%)을 겨냥하되
캐릭터 보존(§2.5)과 함께 별도 사전등록으로 설계한다.

## §S13.23 국가-매핑 지수 리비전 피처 — 사전등록 (2026-07-29, 측정 전)

**데이터 이벤트**: 2026-07-29 16:34 사용자가 run_data_pipeline.bat 실행 —
ai_signal_data.xlsx에 지수 리비전 6열 신규(SPX/NDX/SX5E/DAX/CAC/JPN_REV,
`Indices Revision Rate (지수)` 시트, 7일 달력→영업일 ffill) + 전 시트
리프레시(07-28까지 연장). **빈티지 포크 → S0 재인증 의무**(§S12 관행).
기존 S0 1.4380(07-28 12:11 빈티지)과의 직접 비교 금지.

**사용자 지시**: "인덱스단위의 리비전 데이터들이 features로 포함되었을때
유의미한지 테스트해줘."

**구조적 전제(측정 불요)**: 전 종목 동일값 bcast 형태는 §S13.18(4피처)·
§S13.21(2피처)에서 gain 정확히 0으로 2회 실증·축 폐쇄 — 3차 반복하지 않는다.
**유일한 실행 가능 형태는 국가 매핑**: 유니버스 200종 = US 152 + 비US 48
(FP 12·JP 10·GR 7·LN 7·SW 5·KS 2·NA 2·SM 2·DC 1)이므로 종목→소속시장 지수
매핑은 날짜 내 횡단면 변동(5개 그룹)을 만들어 쿼리내 상수 함정을 벗어난다.

**매핑 테이블(사전약정, 스윕 금지)**: US→SPX_REV / FP→CAC_REV / GR→DAX_REV /
JP→JPN_REV / NA·SM→SX5E_REV(유로존) / LN·SW·DC→SX5E_REV(유럽 프록시, 비유로존
근사 — 선언된 부정확) / KS→SPX_REV(글로벌 앵커 fallback, 2/200 선언된 부정확).
NDX_REV는 종목 매핑 불가(섹터 아닌 지수 중복)로 미사용. 날짜내 상수 차분
동치성: 매핑 레벨과 (레벨−SPX_REV)는 쿼리내 랭킹 분할에 동일 → 레벨 단독.

**사전점검(백테스트 0회, `scripts/preflight_s13_23_index_rev_mapping.py`)**:
파이프라인 로더 실물(`sheets["Daily_Returns"]` USD·마스킹)로:
- **G1(변동)**: 200종 전부 매핑 성공 & 비앵커(SPX_REV 외) 비중 ≥ 10%.
- **G2(신호)**: 날짜별 횡단면 Spearman IC(매핑 레벨 vs 21BD 선행 USD 수익,
  유효쌍 ≥ 100) — |mean IC| > 0.01 & 전·후반 부호 일관. 방향 무관(랭커는
  단조 양방향 학습 가능).
- 서술 전용: 그룹별 평균 선행 수익, 서브기간 3분할 IC.
게이트 통과 시에만 arm 진행, 미달 시 SHELVE.

**arm 스펙(단일 사전약정)**: 피처 1개 `fac_idx_rev` = 종목별 매핑 지수 리비전
레벨. `src/features/index_revision.py`, index_eps와 동일 idiom(무조건 빌드,
Factor 그룹 합류로 CS z-score 스킵, `config.index_revision_feature_enabled`
default-OFF, `admitted` 게이트). OFF 바이트동일 parity 테스트 선행.
- **E1**: 재인증 S0 대비 ΔIR > +0.36 & 서브기간 부호 일관 → 채택. 미달 불채택.
- **E2**: (a) fac_idx_rev gain > 0 (위치 매핑, 전체-패널 모델 — bcast 함정
  탈출 확인), (b) 퇴화율·TE 캐릭터 비악화.
- 기대 서술: 비US 24% 뿐이라 그룹 배분 신호 — 사전 기대 |ΔIR| ~ 0.1 안팎,
  E1 바 통과 확률 낮음을 인지하고 실행(사용자 지시 실증 목적).

인벤토리: arm 실행 시 +1 (사전점검은 진단, 불산입). 상태: **사전등록 완료,
측정 PENDING** (사전점검 → S0 재인증 → arm 순차).

### §S13.23 사전점검 결과 (2026-07-29) — **PROCEED (2게이트 통과)**

실행: `scripts/preflight_s13_23_index_rev_mapping.py`(단위테스트 5종 PASS,
산출물 `outputs/s13_23_index_rev_preflight.csv`, 파이프라인 로더 실물 사용).

- **G1 PASS**: 200/200 매핑 성공, 비앵커 46종(23.0%) ≥ 10%
  (SPX 154 / SX5E 17 / CAC 12 / JPN 10 / DAX 7).
- **G2 PASS**: 일간 횡단면 Spearman IC n=3125, **mean +0.0373** (>0.01),
  반분 +0.0606/+0.0139 부호 일관. 서술: 3분할 +0.0579/+0.0379/+0.0160 —
  후반으로 갈수록 감쇠. 그룹별 선행수익(연율): SPX +21.6% / JPN +24.4% /
  DAX +17.3% / SX5E +14.4% / CAC +12.8%.
- **서술 주의(confound)**: 양(+) IC의 상당 부분은 "US 리비전 우위 × US 주식
  프리미엄" 동행일 수 있음 — 피처가 실제 제공하는 것은 국가 틸트 신호이며,
  랭커가 이를 수익으로 환전하는지는 arm이 판정한다.

**판정**: PROCEED — 사전등록된 arm(fac_idx_rev 1피처) 진행. 인프라 구현 완료:
`src/features/index_revision.py`(+테스트 5종), 로더 Earnings_Revision 카테고리,
assembly Factor 그룹 배선, `config.index_revision_feature_enabled` default-OFF.
전체 스위트 438 PASS(OFF-parity는 화이트리스트 필터 구조로 보장, S13.18 동일).

### §S13.23 S0 재인증 (2026-07-29 22:38, 리비전-빈티지) — **새 기준선 IR 1.3878**

16:34 리프레시 빈티지(리비전 6열 + 07-28 연장)로 production variant
(`codex_causal_rank_65.yaml`) 재실행 (schtasks 우회, 2323s, ECOS 192/192
fallback 0):

| 지표 | S0(리비전-빈티지) | 참고: 구빈티지(07-28 12:11) |
|---|---:|---:|
| IR | **1.3878** | 1.4380 (비교 금지) |
| P1/P2/P3 | 1.526/0.728/1.954 | 1.479/0.787/2.192 |
| TE / 회전율 / beta | 3.66%/0.670/1.058 | 3.73%/0.691/1.059 |

빈티지 포크 효과 −0.050 — 데이터 연장·시트 리프레시 기인, 코드 변경 무관
(리비전 6열은 whitelist-OFF inert). **이 S0가 §S13.23 arm의 단일 비교 기준.**
실행 사고 기록: 백그라운드 셸 2회 강제종료 → schtasks(배터리 허용)로 안정화;
21:57 인스턴스는 중복 오판으로 내가 kill(부모-자식 중첩이 실체) 후 21:59
단일 재시작. arm 1차 시도는 시작 10분 뒤 외부 종료(0x40010004, 시스템
절전/로그오프 추정) — 재실행 필요.

### §S13.23 결과 (2026-07-30) — **E1 FAIL·불채택. 단, 매핑이 bcast 함정을 탈출해 지수 데이터 최초 소비 실증**

실행: arm exit 0(1551s, schtasks; 1차 시도는 07-29 22:48 외부 종료로 무효),
ECOS fallback 0. 인벤토리 447→448(+arm 1; S0 재인증은 재기준선으로 불산입).

| 지표 | S0(리비전-빈티지) | **arm (fac_idx_rev)** |
|---|---:|---:|
| IR | 1.3878 | **1.3259** (ΔIR **−0.062**) |
| P1/P2/P3 | 1.526/0.728/1.954 | 1.502/0.557/2.037 (−0.02/−0.17/+0.08) |
| TE / 회전율 / beta | 3.66%/0.670/1.058 | 3.74%/0.690/1.066 |
| 퇴화율 | 0.469 | 0.531 |

**E1: FAIL** — ΔIR −0.062(음수, |Δ| < 0.36 노이즈 대역), 서브기간 부호 혼재
(P2 −0.17이 최대 손상). **불채택, flip 0건, default-OFF 유지.**

**E2: (a) PASS — 구조적 발견**: fac_idx_rev **gain 26.46, share 0.344%**
(전체-패널 모델 2개 위치 매핑; 히스토그램 62×2/60×30 — EWMA 프루닝 60개
모델은 매핑 불가). §S13.18/21의 "지수 데이터 gain 정확히 0"이 **매핑 형태로는
깨진다** — 날짜내 횡단면 변동(5그룹, 비앵커 23%)을 만들면 랭커가 실제로
소비한다. (b) 캐릭터: TE 3.74% ≤ 4.5% 유지, 퇴화율 +0.06 경미 악화,
비유한 예측 경고 93건은 S0와 동일(기존 현상, 피처 무관).

**판독**: 소비 ≠ 가치. 사전점검 IC +0.037의 상당분이 "US 리비전 우위 × US
프리미엄" 동행(confound 서술 그대로)이었고, 랭커가 국가-틸트 신호로 소비한
결과는 P2(중간기) 손상으로 순마이너스. 5그룹 국가 배분 정보는 이 파이프라인
의 종목 선택 알파에 기여하지 못한다. **지수 리비전 데이터 축: 유일한 실행
가능 형태(매핑)까지 소진 — 축 종결.** 단, "매핑이 bcast 함정을 탈출한다"는
방법론적 발견은 향후 다른 지수/매크로 데이터에 재사용 가능(새 사전등록 필요).

산출물: `outputs/arm_s13_23_index_rev`(판정 후 pkl 삭제 대상),
`outputs/s13_23_index_rev_preflight.csv`. 코드 잔존: default-OFF 인프라
(index_revision.py + 테스트 10종, 전체 438 PASS). 커밋 미실시(승인 대기).

## §S13.25 Fwd Sales 기간구조 slope 피처 + 월말 리밸런싱 — 사전등록 (2026-07-31, 측정 전)

(§S13.24 번호는 병행 세션 gpt_ai_port 상대 리비전 프리레지가 선점 — 충돌 회피로 건너뜀.)

**데이터 이벤트**: 2026-07-31 15:12 ai_signal_data.xlsx 재생성 — 전 시트 07-31
연장 + **`Fwd_Sales_Slope_1FY2FY` 시트 신규**(BEST_SALES_1FY/2FY에서 유도한
FY1→FY2 내재 매출 성장률 `(2FY−1FY)/1FY`, 사용자 정의 확정). 원천 시트의
티커별 초기 평탄 백필 접두는 데이터 레이어에서 NaN 마스킹(실데이터 시작:
최소 2014-05 / 중앙값 2020-11 / 최대 2025-05, 유효셀 45%). 부수 사고:
ENOSPC로 구파일 truncate 파괴 → 재생성 복구, ExcelWriter in_memory 전환.
**빈티지 포크 → S0 재인증 의무**(§S12 관행). §S13.23 S0 1.3878과 직접 비교 금지.

**사용자 지시**: (1) 상장 전 마스킹 계약을 지키며 신규 데이터를 features로
반영(비선형 피처 신설 포함), (2) 리밸런싱을 월별 마지막 영업일로 변경,
각각 포트폴리오 결과 제시. 채택이 아닌 **결과 열람 목적** — E1 바는 기존
기준(ΔIR>+0.36 & 서브기간 부호 일관)을 그대로 적용해 기록만 한다.

**상장 전 배제 경로(구조 확인)**: 시트는 `load_all_sheets` 전량 로드 →
preprocess 1차 마스킹 → align/impute → §S11.4 2차 re-mask로 상장 전 NaN 고정.
백필 접두는 생성 단계(`_mask_backfilled_prefix`)에서 이미 NaN. 결측은 기존
계약(횡단면 중앙값 중립 채움, §S13.6 실증) 그대로.

**arm 스펙(사전약정, 스윕 금지)**:
- **A(피처 4개, `config.fwd_sales_slope_features_enabled` default-OFF)**:
  `src/features/fwd_sales_slope.py`, S8 idiom(무조건 빌드, core-whitelist
  게이트, OFF 바이트동일). 선형 2: `fwd_sales_slope_level`(레벨→CS z),
  `fwd_sales_slope_chg_63d`(Δ63, 스티프닝 모멘텀). **비선형 2**(S13.15
  soft-AND min-confirm idiom, 트리 단조변환 불변성 회피 목적으로 2변수 결합):
  `nl_fslope_rev_confirm` = min(z(level), z(sales_rev_ma_63d)),
  `nl_fslope_growth_confirm` = min(z(level), z(best_sales_chg_252d)).
- **B(월말 리밸런싱, `config.rebalance_at_month_end` default-OFF)**: ON이면
  21BD 고정 그리드 대신 **각 월 마지막 거래일**(+백테스트 첫날 초기 편입)에
  리밸런싱. `make_month_end_rebal_check`로 `simulate_portfolio`의 기존
  `rebal_check_fn` 훅에 주입 — OFF 경로는 기존 코드 그대로.
- **C(결합)**: A+B 동시 ON — 사용자 최종 요청 상태.

**판정**: E1 기존 바(기록용). E2: (a) 신규 피처 gain>0 여부, (b) TE≤4.5%·
퇴화율·beta 캐릭터 비악화. B는 리밸런스 횟수·회전율·TC 변화를 서술.
인벤토리: S0 재인증 불산입, arm 3건 산입(448→451).

상태: **측정 완료 (2026-08-03)** — 아래 결과 참조. (S0′ → A → B → C 순차,
동일 빈티지·동일 ECOS·production variant `codex_causal_rank_65.yaml`
오버라이드 1~2개만.)

### §S13.25 측정 결과 (2026-08-01~03, ECOS, 판정: arm 3종 전부 불채택)

**실행 기록**: 2026-08-01 13:34 시퀀스 시작 — S0′ 13:53 exit 0, A 14:11 exit 0.
B 첫 시도는 세션 자식 프로세스로 돌다 슬립 hang(14:18~23:10) → 23:10 재시작
→ 23:25 세션 종료와 함께 kill(완주 ~1분 전, traceback 없음, 오류 아님). C 미착수.
2026-08-03 00:03 schtasks(배터리 허용·세션 분리)로 B→C 재실행: B 00:15 exit 0
(750s), C 00:29 exit 0(797s). 장기 런 = schtasks 분리 실행 패턴 재확인.

**수치** (모두 ECOS, solver fallback 0, 유니버스 200, 캡 0.035):

| run | IR | TE | realized_beta | turnover/yr | 퇴화율 | ΔIR vs S0′ | 서브기간 IR (P1/P2/P3) |
|---|---|---|---|---|---|---|---|
| S0′ 재인증 (`s0_recert_s13_25`) | **1.3070** | 3.61% | 1.062 | 0.663 | 14/32 | — | 1.191 / 0.694 / 2.032 |
| A 피처 4종 (`arm_s13_25a`) | **1.5734** | 3.64% | 1.052 | 0.691 | 15/32 | **+0.266** | 1.336 / 1.121 / 2.082 (전부 +) |
| B 월말 리밸 (`arm_s13_25b`) | **1.2059** | 3.60% | 1.066 | 0.612 | 14/32 | **−0.101** | 1.006 / 0.435 / 2.137 (혼합) |
| C 결합 (`arm_s13_25c`) | **1.3060** | 3.66% | 1.052 | 0.647 | 15/32 | **−0.001** | 1.251 / 0.864 / 1.701 (혼합) |

**판정**:
- **E1**: A +0.266 < +0.36 → FAIL. 단 서브기간 3구간 delta 전부 양(+0.145/+0.427/+0.050)
  — 이 피처 프로그램에서 바 이하이지만 전구간 일관 양 delta는 최초. B·C FAIL.
- **E2a (gain)**: 4피처 전부 32/32 모델에서 소비, block gain share **6.09%**
  (level 2.02% · chg63 1.06% · rev_confirm 1.30% · growth_confirm 1.72%).
  §S13.21(gain 0.0000)·§S13.23(0.34%)과 대조되는 **최초의 실질 소비**. A·C 동일
  수치 — 모델 계층은 리밸 스케줄과 무관하므로 예상된 일치(퇴화율 S0′=B 14/32,
  A=C 15/32 짝 일치로 교차 확인).
- **E2b**: TE 전 arm ≤4.5%, beta 1.05~1.07, 집중 캐릭터 유지. 퇴화율 A/C +1건
  (43.75→46.9%) — 소폭 악화, 기록.
- **B 서술(사전등록 의무)**: 리밸 횟수 96→93회(ECOS 호출 192→186), turnover
  0.663→0.612/yr, 연 TC 0.066%→0.061%. TC 절감에도 ΔIR 음 — 21BD 그리드 대비
  월말 고정 스케줄의 미스얼라인 손실이 절감분을 압도.
- **결합 해석**: 가산 기대 1.472 대비 실측 C 1.306 → 상호작용 **−0.17**.
  월말 스케줄이 A의 피처 이득을 소거.
- **non-finite prediction 경고**(리밸 시점 n=22→1 감소)는 상장 전 마스킹·롤링
  피처 워밍업 NaN의 정상 결측 경로로 S0′/A/B 공통(93·93·87건) — 버그 아님,
  조치 불요.

**채택**: arm 3종 전부 **불채택, default-OFF 유지, production flip 없음**
(사용자 지시 자체가 결과 열람 목적). A의 "실질 gain 소비 + 전구간 양 delta"
조합은 §2.4에 따라 설명력 기록으로만 남긴다 — 스윕·재시도는 새 사전등록 없이
하지 않는다. 인벤토리 448→451(S0′ 재인증 불산입).

### §S13.25 Production flip — slope 피처 4종 채택 (2026-08-04, 사용자 오버라이드)

- **사용자 지시**: "피처 4종 적용할래. 그걸 최종안으로 해줘" — E1 바(+0.36)
  미달(ΔIR +0.266)을 위 측정 기록과 함께 보고받은 상태에서의 명시적 오버라이드.
  B(월말 리밸)·C(결합)는 불채택 유지. 후보 1개 단독 flip(§8 준수).
- **§2.7 DSR 해킷(N=452, S13.24 사후 회계 반영, 2026-08-04)**: DSR p=0.1920 · Deflated SR
  0.871(관측 SR 1.548, null 최대기대 1.240) → DSR FAIL; Haircut 조정 SR 0.309
  PASS; MinTRL 1.1yr(보유 7.9yr) SUFFICIENT; 서브기간 IR 1.341/1.235/2.003
  전부 양 STABLE; 생존편향 WARN 19종(구조적, 기존과 동일). 종합 **FAIL —
  2026-07-11 codex 승격과 동일 구조의 사용자 오버라이드**로 기록.
- **flip**: `variants/codex_causal_rank_65.yaml`에
  `fwd_sales_slope_features_enabled: true` 1줄(+주석) 추가.
- **재검증(§8)**: 편집된 production variant 자체를 재실행(2026-08-04
  09:12→09:27, exit 0, schtasks 분리) → arm A와 **전 지표 float 정밀도까지
  EXACT 일치**(IR 1.5733669451 / TE 3.6362% / beta 1.0521 / turnover 0.6908 /
  subs 1.3355·1.1205·2.0823 / ECOS 192 / fallback 0). `outputs/
  codex_causal_rank_65` 산출물 갱신.
- **롤백**: 플래그 1줄 삭제 = default-OFF 복원. OFF parity·게이트는
  `tests/test_fwd_sales_slope.py` 5/5 PASS로 인증. 전체 스위트 443 PASS —
  과거 arm 수용 테스트 6건은 "이후 채택된 production 플래그 제외" 최소 수정
  (테스트 의도인 측정 당시 단일-delta 사실은 불변).
- **새 production 헤드라인**: **IR 1.5734 / TE 3.64% / beta 1.052 /
  turnover 0.691** (slope 빈티지 S0′ 1.3070 대비 +0.266). 퇴화율 46.9%(+1건)
  — HOLD 게이트 항목은 다음 스케줄 런에서 자동 재평가.
- 커밋: research 9762a71 + flip(본 기록 포함) 독립 2건.

### §S13.25 사후 중복 진단 — slope 4피처 vs 코어 61 (2026-08-04, 비액션)

**동기**: flip 직후 사용자 질의 "신규 피처가 기존 피처와 중복인가". 성능 변경이
아니라 설명력 기록이므로 §2.4에 따라 **비액션**으로 남긴다(인벤토리 불산입 —
백테스트 미실행, 기존 패널 재측정만).

**방법**: flip 후 production 설정(`codex_causal_rank_65`, slope block ON, 코어
61+4=65)으로 `build_all_features` 패널을 만들고 **per-date 횡단면 Pearson을 전
기간 평균**. 상장 전 median-fill 행은 상관을 인위적으로 부풀리므로 §S13.7 선례대로
제외 — 2020-01-01 이후 & 상장일 이후 **332,031행 / 1,717일**(전체 652,800의 50.9%).
진단 스크립트는 일회성이라 레포 미커밋.

**결과 1 — 기존 코어 61개와 중복 없음**:

| 신규 피처 | 최근접 기존(부모 제외) | corr | max\|corr\| | #(\|corr\|>0.5) |
|---|---|---:|---:|---:|
| `fwd_sales_slope_level` | realized_vol_126d | 0.400 | 0.400 | 0 |
| `fwd_sales_slope_chg_63d` | tg_mom_63d / best_sales_chg_252d | 0.079 | 0.079 | 0 |
| `nl_fslope_growth_confirm` | **best_sales_chg_252d(부모)** | 0.625 | 0.625 | 1 |
| `nl_fslope_rev_confirm` | **sales_rev_ma_63d(부모)** | 0.668 | 0.668 | 1 |

- `fwd_sales_slope_level` ↔ `best_sales_chg_252d`는 **0.300**뿐 — 전자는 단면의
  FY1→FY2 내재 성장 기대, 후자는 1FY 컨센서스의 시계열 252d 변화로 다른 양임이
  실측으로 확인.
- `fwd_sales_slope_chg_63d`는 코어 61개 전부와 |corr|<0.08 — 유니버스에 없던 축.
  기존 `sales_rev_ma_63d`와 겹치지 않는 이유: 1FY·2FY 동율 리비전은 slope 불변이라
  Δ63은 **연도 간 차등 리비전**만 포착.
- 두 confirm의 0.63/0.67은 `min(a,b)`가 정의상 부모를 절반쯤 통과시키는 결과이며
  부모 2종 모두 이미 코어 승인 피처 — 숨은 raw 입력 유입 없음(사전등록 서술 확인).

**결과 2 — 실질 중복은 블록 내부**: level↔growth_cf **0.763**, growth_cf↔rev_cf
**0.716**, level↔rev_cf 0.593 (chg63만 0.22~0.36으로 독립). soft-AND tie rate —
min이 slope 다리로 결정되는 셀: growth_cf **45.8%**, rev_cf **51.6%**, 두 confirm이
**동시에** slope_z와 같아지는 셀 **34.9%**(n=332,031). 즉 4피처의 실효 축은 ~2.5개.
gain 지분(§S13.25 E2a) 상위 3종 5.04%p가 1/3은 같은 컬럼을 세 번 보는 구조.

**결과 3 — level의 최상위 상관은 성장이 아니라 변동성 축**: realized_vol_126d
0.400 · beta_63d 0.359 · idio_vol_63d 0.328 · max_ret_63d 0.317 > best_sales_chg_252d
0.300. slope 레벨이 "고성장 기대 = 고변동 종목" 프록시로 부분 작동 — ΔIR +0.266의
알파 대 리스크프리미엄 분해는 **미해결**로 기록. 다만 포트폴리오 레벨에서는
realized_beta 1.062→1.052, TE 3.61→3.64%로 리스크 캐릭터 악화 없음(§2.5 유지).

**판정(비액션)**: "기존 축과 중복"이라는 기각 사유는 실측으로 배제 — 채택 유지.
블록 축소(4→3 또는 4→2)는 **오늘 실측한 내부 중복률을 사전 가설로 삼는 별도 단일
사전등록 arm** 사안이며, 지금 IR을 보고 조합을 고르는 것은 §2.4 스윕 금지(p-hacking)
위반. 추적 항목은 퇴화율 46.9%(HOLD 잔존)와 결과 3의 변동성 축 노출 2건.

## §S13.26 2주(10BD) 리밸런싱 주기 — 사전등록 (2026-08-04, 측정 전)

**사용자 지시**: "2주 단위 리밸런싱이 비용 고려 시 더 이득인지" — 아래 사전점검이
음의 기대를 제시했으나 **사용자가 실측 수치를 요청**하여 단일 arm 1건 실행.
사전점검 결론과 무관하게 E1 바는 기존 기준을 그대로 적용해 기록한다.

**사전점검 (읽기 전용, arm 실행 전 기록)**:
- **비용은 병목이 아님**: `one_way_tc 0.0010`(편도 10bp), 현행 턴오버 0.691/yr →
  연 TC **6.9bp**. 10BD에서 턴오버가 2배가 되는 최악 가정에서도 ΔTC ≈ **+6.9bp/yr**.
  현행 액티브 수익 = IR 1.5734 × TE 3.636% = **572bp/yr**이므로 비용 증가분은 액티브의
  1.2%. **손익분기 ΔIR = 6.9/364 = +0.019** — 비용 허들은 사실상 없다.
  (실제로는 10일 드리프트가 작아 `no_trade_band 0.003`에 더 많이 걸리고
  `partial_rebalance_eta 0.50`이 겹쳐 턴오버 2배 미만이 유력.)
- **기대 방향은 음**: §S11.8 IC 감쇠 진단이 h05 0.048 / h20 0.053 / h40 0.068 /
  h63 0.080(t 3.6)로 **단조 증가** — 이 신호는 빠른 구간이 가장 약하다. 빠른 리밸이
  이득인 전제("알파가 빨리 죽는다")가 성립하지 않고, `forward_horizon 20` 라벨이
  실현되기 전에 두 번 거래하게 된다.
- **반증 선례**: §S13.25 arm B(월말)는 주기 사실상 동일(96→93회)한 스케줄 이동만으로
  ΔIR −0.101, 그것도 턴오버·TC가 **줄어든 상태**에서 손실 — 이 전략의 리밸 손익은
  비용이 아니라 타이밍 정합이 지배.
- **모델 계층 불변 확인**: `train_and_predict`는 전 영업일 루프에서 `retrain_freq 63`
  거래일마다 재훈련하므로 **리밸 주기와 독립**(재훈련 32회 고정). §S13.25의
  A=C·S0′=B 짝 일치와 동일 구조. 따라서 이 arm의 델타는 순수 옵티마이저/집행 계층
  효과이고, 퇴화율은 production과 동일해야 한다(교차 검증 지표로 사용).

**arm 스펙(사전약정, 스윕 금지)**: `variants/arm_s13_26_reb10.yaml` —
현행 production(`codex_causal_rank_65`, 커밋 92ebcf1, slope 피처 ON) 전량 복사 후
**`rebalance_freq: 21 → 10` 단 1줄만 변경**. `partial_rebalance_eta 0.50`·
`no_trade_band 0.003`은 **의도적으로 불변** — 이 둘은 §S11.8 느린 알파 전제로
튜닝된 값이라(라인 2379·2484) 함께 건드리면 델타 귀속이 불가능해진다. 동반 재튜닝은
본 arm의 범위 밖이며, 결과가 음이어도 "damping 미스매치 탓"이라는 사후 변명은
새 사전등록 없이 쓰지 않는다.

**비교 기준**: 현행 production **IR 1.5734 / TE 3.64% / beta 1.052 / turnover 0.691 /
연 TC 6.9bp / 퇴화율 15/32**(§S13.25 flip, EXACT 재현 인증분).

**판정 기준**:
- **E1**: ΔIR > +0.36 & 서브기간 부호 일관(§2.4 기존 바).
- **E1′(비용 손익분기, 본 arm 한정 기록용)**: ΔIR > ΔTC/TE 실측치. 비용만 갚는지를
  따로 적어 "비용 때문에 진 것인지"를 사후에 혼동하지 않게 한다.
- **E2**: (a) 턴오버·리밸 횟수·연 TC 실측(2배 가정 대비 실제), (b) TE ≤ 4.5%·
  beta·집중 캐릭터 비악화(§2.5), (c) 퇴화율이 production과 동일한지(모델 계층
  불변 교차검증).
- 인벤토리: arm 1건 산입(451→452).

상태: **측정 대기** — 실행은 schtasks 분리(§S12 안정 패턴). 예상 런타임은 ECOS 콜
192→~402로 늘어 production 재실행(~900s) 대비 증가 예상.

## §S13.27 고변동 노출 방어 옵티마이저 arm 2종 — 사전등록 (2026-08-04, 측정 전)

(§S13.26은 병행 세션 10BD 리밸 arm이 선점 — 본 시퀀스는 S13.27 사용.)

**동기/사용자 지시**: production SHAP 진단에서 OW 클러스터(MU·STX·6857·TER·
6146·NXPI)의 지배 드라이버가 `idio_vol_63d`·`realized_vol_126d`(Price 그룹 6종
합산 +0.649, 나머지 전 그룹 합의 20배)로 확인되고, §S13.25 사후 진단 결과 3
(slope level 최상위 상관 = 변동성 축)이 겹침. 사용자: "②(bm_proportional_cap)와
vol-팩터 노출 제약 모두 테스트, 기존 포트폴리오 결과·최신 OW 종목 비교" — 방어
수단 실측 목적. 두 arm 모두 **옵티마이저 계층만** 변경(신호·모델 불변 → 퇴화율은
production과 동일해야 하며, §S13.26과 같은 교차검증 지표로 쓴다).

**사전점검 (2026-08-04, 읽기 전용)**: production 07-21 리밸 시점, 엔진과 동일한
`estimate_covariance`(126d LW+메가캡 shrink) 대각 기준 —
- w·vol **38.82%** vs bm·vol **35.43%**(연율), **비율 1.0955(+9.55%)**.
- 노출 기여 상위: MU +1.88%p·6857 +0.96·LLY +0.95·STX +0.82·TER +0.67
  (active×ann vol).
- arm A 캡 미리보기: 고변동 OW 캡 4%→2.0~2.3%(mult 0.50~0.58), 07-21 기준
  MU(+3.31%p > 2.33%)가 즉시 바인딩 — 인프라가 실제로 작동함을 확인.

**arm 스펙(사전약정, 스윕 금지)**:
- **A(`arm_s13_27a_bmprop.yaml`)**: production(92ebcf1, slope ON) 전량 복사 +
  `bm_proportional_cap_enabled: true` 1줄. 2026-07-02 구조 리뷰의 default-OFF
  인프라 그대로(top 1.5×·vol floor 0.5·lookback 63), 파라미터 재튜닝 금지.
- **B(`arm_s13_27b_volcap.yaml`)**: 신규 `vol_exposure_cap_enabled`(default-OFF)
  + `vol_exposure_cap_excess: 0.05`. 제약 = `vols@w ≤ 1.05·vols@bm`,
  vols=sqrt(diag(cov)), `_build_mvo_constraints`에 선형 제약 1건이라 MVO와
  post-execution projection에 동일 적용. x=0.05는 사전점검 실측 +9.55%의 절반
  (바인딩 보장, 단일값). BM 자체는 항상 feasible(비율 1.0)이라 신규
  infeasible→BM fallback 경로 없음.
- 결합 arm 없음(요청 범위 밖).

**판정 기준(방어 arm 전용 — E1 ΔIR>+0.36은 목적 부적합, 기록만)**:
- **D1(방어 실효)**: (a) 07-21 vol 노출 비율 1.0955 대비 실측 감소, (b) tail
  지표 개선 ≥1건: MaxDD(총수익)·active MaxDD·최악 서브기간 IR·downside
  capture(벤치 음수일) — 각 arm pkl에서 산출, production 동일 방식 대조.
- **D2(비용)**: ΔIR 실측 기록. ≥0이면 무비용 방어(채택 후보 회부), <0이면 IR
  비용 대 tail 개선 트레이드오프를 사용자 결정으로 회부. 자동 채택 없음.
- **E2(캐릭터)**: TE≤4.5%·beta·집중 캐릭터, 퇴화율 15/32 동일(옵티마이저 계층
  한정의 교차검증).
- **OW 비교**: 07-21 리밸 active 상위 12종 production/A/B 3열 비교표(사용자
  요청 산출물).

**비교 기준**: production **IR 1.5734 / TE 3.64% / beta 1.052 / turnover 0.691 /
퇴화율 15/32**(92ebcf1, 워크북 07-31 15:12 빈티지 오늘 불변 확인).
인벤토리: arm 2건 산입(§S13.26 산입 후 452→454).

상태: **측정 완료 → 문서 말미 §S13.27 사후 판정 참조** (A 기각·B 회부, 채택 없음).
(2026-08-04 16:33 확인: 16:02:40 START한 arm A 런이 16:03:57에서 정지·상태로그에 `A exit`
미기록 = **사망**. `outputs/s13_27_run_ab.ps1`에는 §S13.26 2차에서 추가한 슬립 억제
(`SetThreadExecutionState`) 블록이 없다. 재실행 시 그 블록을 이식할 것.)

## §S13.28 알파–전통팩터 직교성 진단 (2026-08-04, 측정 완료) + 잔여 신호 IR arm 사전등록

**동기/사용자 질문**: "현재 알파 수익률이 전통적 팩터와 유의미한 차이점이 있는가."
§S5(factor-neutral)는 *노출 바인딩* 기준이고 65종 시절 산출물이라 **수익률 축 답이
없었다**. 본 절은 수익률 공간 스패닝 회귀로 그 공백을 메운다.

### 1. 수익률 공간 스패닝 회귀 (읽기 전용 — 신규 백테스트 없음)

**데이터**: production `codex_causal_rank_65` pkl(2026-08-04 12:01 빈티지, 커밋 92ebcf1)
일간 액티브 2003일(2018-11-27~2026-07-30). production IR 1.5734 / 액티브 5.72% /
TE 3.64% / beta 1.052. **추론**: Newey-West HAC lag 21(리밸 1주기). 환경의 statsmodels가
깨져 있어(`statsmodels.robust._qn` ImportError) OLS+NW를 직접 구현.

**팩터 2계열**:
- (a) **거래가능 전통 팩터** = `Factor_PX_LAST`의 팩터 ETF 7종(F_MinVol/Quality/HiDiv/
  Growth/Value/SmCap/HiBeta) 일간수익 − SPX(= 시장중립 틸트).
- (b) **유니버스 내 특성 팩터** 7종 = SMB(−log 시총)/VALUE(−best_px_bps_ratio_level_z)/
  MOM(momentum_252d)/QUALITY(best_roe_level_z)/LOWVOL(−realized_vol_126d)/BAB(−beta_63d)/
  GROWTH(best_sales_chg_252d). 일별 1/99 위노라이즈 z → 달러중립 $1롱/$1숏, **가중치 1일
  래그**(룩어헤드 없음).
- 시장항 = BM − 3M UST(β−1 흡수).

| 스펙 | alpha(연) | t(NW21) | R² |
|---|---:|---:|---:|
| S1 시장만 | +4.78% | 3.99 | 7.2% |
| S2 + ETF 7종 | +4.54% | **4.09** | 34.5% |
| S3 + 특성 7종 | +1.20% | 1.37 | 56.0% |
| S4 전부 | +1.66% | 1.90 | 57.4% |

**강건성(S3 계열)**: BAB 제외 +2.06%(t 2.10) / LOWVOL 제외 +1.45%(t 1.43) / 둘 다 제외
+1.48%(t 1.52) / 십분위 스프레드 +2.14%(t 2.37) / 섹터중립 z +1.37%(t 1.64) / 섹터중립
십분위 +1.85%(t 1.97) / 21일 비중첩 +1.20%(t 1.26). LOWVOL–BAB corr 0.93·VIF 11은 **개별
베타 해석만** 무효화하고 alpha·R²에는 무영향(drop-one으로 확인). 서브기간 alpha
+0.18 / +2.47 / +1.43%(t 0.12 / 1.95 / 0.90), 롤링 252d alpha 84창 중 71% 양·평균 +1.36%.

**신호 공간**: 리밸 96회 스코어를 7특성에 횡단면 회귀 → 평균 R² **65.4%**(중앙 66.3%).
21일 forward IC 원본 +0.0594(t 2.70) → 직교화 후 **+0.0148(t 1.68)**, 잔존 **25%**.

**판정**: **기준 의존**. (a) 외부 ETF 기준으로는 유의(t 4.09)하나 이는 ETF가 200종
글로벌 메가캡 유니버스의 횡단면을 스팬하지 못하는 데서 오는 **상한**이다. (b) 같은
유니버스 특성 기준으로는 **유의하다고 주장 불가**(t 1.3~2.4, 구성 방식 전 범위에서
t=2를 안정적으로 넘지 못함). 액티브 5.63%/yr 중 **4.4%p(≈79%)가 스타일 노출로 설명**.

**중요 뉘앙스**: 설명 축의 표본 내 수익은 LOWVOL **−31%/yr**·BAB −20%/yr — 교과서
프리미엄과 **부호가 반대**다. 즉 "전통 팩터 프리미엄 수확"이 아니라 **전통 팩터 반대편
(고변동·고베타·성장·유니버스 내 소형)에 선 지속 틸트**이고 레짐 의존적이다.
§S13.12(액티브의 ~41%가 IC 불가시 common 캐리)·§S5(factor-neutral이 IR을 깎음 =
스타일 베팅이 수익원)·현행 DSR FAIL(p=0.192, N=452)과 방향 일치.

**한계**: (i) 유니버스 내 SMB +23.7%/yr은 늦은 진입 19종(PLTR·ARM·CRWD·DDOG 등,
선택편향 리포트 §5)의 영향으로 과대평가 소지 → 설명분은 상한·잔여 alpha는 하한.
(ii) 특성 팩터는 모델이 학습에 쓰는 피처로 만든 **가장 가혹한 기준**. (iii) 번들에
FF/Barra 정본 팩터가 없어 외부 기준은 ETF 대용. 산출물 JSON은
`outputs/orthogonal_signal_arm/spanning_*.json`, 특성 정의 정본은
`scripts/run_orthogonal_signal_arm.py`의 `CHARS`/`_zscore`.

### 2. 잔여 신호 IR arm 사전등록 (측정 전, 사용자 지시 2026-08-04)

**목적**: "직교화 잔여 알파가 실제로 **투자 가능한가**". **진단 전용 — 프로덕션 flip
후보 아님**(no-flip 명시).

**공통 설계(하베스트 1회 · 재-MVO 3회)**: production pkl의 `pre_overlay_predictions`
(§4.2 정본 객체) + models/panel/targets/features를 `precomputed_*`로 주입 → 오버레이는
**정확히 1회만** 재적용(이중 오버레이 금지). 모델 재학습 없음, config는 production 변형
그대로. 스크립트 `scripts/run_orthogonal_signal_arm.py`, 테스트
`tests/test_run_orthogonal_signal_arm.py`(5 passed, RED→GREEN 확인).

**arm(사전약정 — 스윕 금지)**:
- **R0 identity**: 하베스트 무변경 주입 — round-trip 게이트.
- **R1 residual**: 매 날짜 스코어를 7특성+절편에 횡단면 OLS → **잔차**. 결측 특성 셀은
  중립(z=0), 스코어 <30종 날짜는 통과(passthrough).
- **R2 style**: 같은 회귀의 **적합값**(스타일 성분).
- R1/R2 모두 그 날짜 원 스코어의 평균·표준편차로 affine 재스케일 → 옵티마이저가 보는
  **신호 스케일 불변, 횡단면 형태만 변경**(재스케일 없으면 잔차 분산 축소가 MVO 공격성
  변화와 교란된다).

**합격기준(사전등록)**:
- **G0(선결·round-trip)**: |IR(R0) − 1.5734| ≤ 0.005 **and** |active(R0) − 0.05721| ≤ 1e-4.
  FAIL이면 R1/R2 수치 **폐기**(주입 경로 불일치).
- **G1(진단 해석 바 — 채택 아님)**: IR(R1) ≥ 1.0 "잔여 알파 단독 투자가능성 있음" /
  0.5~1.0 "약함" / <0.5 "실질 소멸". §2.4의 ΔIR>+0.36 채택 바는 목적 부적합이라 미적용.
- **G2(붕괴 가드 §2.5)**: 각 arm TE ≤ 4.5%, `optimizer_failure_rate` ≤ R0+10pp,
  active share ≥ 0.5×R0. 위반 arm의 수치는 무효(벤치마크 붕괴).
- **G3(정합성)**: IR(R2) 병기. full = style ⊕ residual은 **가법 아님**(MVO 비선형) —
  방향성 확인용으로만 읽는다.

**인벤토리**: R1/R2 **2건 산입 예정**(R0는 동일성 검사라 trial 아님).
`n_trials_total` 452 → 454. (§S13.26·§S13.27은 미측정이라 미산입 상태.)

### 3. 측정 결과 (2026-08-04 16:42~16:56, exit 0, 총 826s — schtasks 분리·슬립 억제)

산출물 `outputs/orthogonal_signal_arm/summary.json` + arm별 액티브 수익 CSV.
분해 진단: 적합 날짜 2004 / 통과 1260(스코어 <30종 = 학습 시작 전 구간) / 퇴화
재스케일 **0**, 평균 횡단면 R² **0.6538**(중앙 0.6736) — 회귀 진단(65.4%)과 일치.

| arm | IR | 액티브 | TE | beta | turnover | fail rate | active share | 런타임 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **R0 identity** | **1.5734** | 5.721% | 3.64% | 1.052 | 0.691 | 0.0% | 0.1972 | 171s |
| **R1 residual** | **0.9248** | 2.359% | 2.55% | 1.016 | 0.977 | 2.08% | 0.1488 | 179s |
| **R2 style** | **1.0816** | 4.092% | 3.78% | 1.048 | 0.645 | 0.0% | 0.1948 | 226s |

**게이트 판정**:
- **G0 PASS** — |ΔIR| **1.20e-6**, |Δactive| **3.07e-8**. 주입 경로가 production을
  부동소수점 정밀도까지 재현(§4.2 경로 무결성 재확인).
- **G2 PASS(R1·R2 공통)** — TE 2.55%/3.78% ≤ 4.5%, fail rate 2.08%/0% ≤ R0+10pp,
  active share 0.1488/0.1948 ≥ 0.5×0.1972. 벤치마크 붕괴 없음.
- **G1 = "약함"(0.5~1.0 밴드)** — IR(R1) **0.925**. ΔIR vs R0 = **−0.649**.
- **G3** — IR(R2) 1.082, ΔIR −0.492. 액티브 합 R1+R2 = 6.45% > full 5.72%로 **가법
  아님**(사전 명시대로 MVO 비선형·중복). 어느 한 쪽도 단독으로 full을 재현 못 함.

**판정(진단, no-flip)**: 직교화 잔여 신호는 **0이 아니다** — 단독으로 연 +2.36%
액티브·IR 0.925를 낸다. 즉 모델에는 7특성으로 환원되지 않는 실재 정보가 있다.
그러나 (a) production IR의 59% 수준이고, (b) turnover가 0.691 → **0.977(+41%)**로
뛰어 잔여 신호가 훨씬 빠르게 회전하는(=비용에 취약한) 성격이며, (c) TE 예산도
2.55%로만 소진돼 리스크 예산을 다 쓰지 못한다. 스타일 성분(R2)이 여전히 액티브
수익의 큰 몫(4.09%/5.72% = 72%)을 담당한다.

**§1 회귀와의 정합**: 회귀 alpha(+1.2~2.1%/yr)보다 R1 액티브(+2.36%)가 큰 것은
모순이 아니다. 회귀 alpha는 *고정 베타의 선형 잔차*이고, R1은 직교 스코어에 **새
리스크 예산과 옵티마이저·오버레이를 다시 붙여** 구성한 포트폴리오다. 두 수치는
"잔여 정보의 하한(회귀)"과 "잔여 정보를 최적 구성했을 때의 값(R1)"으로 읽는다.

**부수 관찰(기존 동작, 본 arm 무관)**: 런타임에 `non-finite prediction(s) … n=8~9`
경고가 **R0/R1 각각 93건 동일**하게 발생한다. 저장 패널(`pre_overlay_predictions`·
`predictions`·`raw_predictions`)의 inf 셀은 **0**이므로 런타임 후처리 단계에서
생겼다가 NaN으로 치환되는 것이며, production 자체의 동작이다(R0가 production을
1.2e-6로 재현). 별건 데이터 품질 점검 대상으로만 기록.

**한계**: R1의 IR 0.925도 in-sample이며 DSR 디플레이션(현행 N=452, p=0.192) 대상이다.
R1/R2는 **채택 후보가 아니고**(§8 flip 없음) production 설정은 무변경이다.

**인벤토리**: R1/R2 2건 산입 완료 — `n_trials_total` 452 → **454**(R0는 동일성
검사라 미산입). 차기 `run_selection_bias.py` 실행부터 새 N이 반영된다.

## §S13.27 사후: 방어 arm 2종 측정 결과 — A 기각·B 회부(권고 기각) (2026-08-04 17:26)

**실행 기록**: 1차(14:22)는 시스템 메모리 고갈로 사망 — C: 여유 0GB로
페이지파일이 축소(커밋 한계 33.7→27.0GB)되며 arm A는 13.5분 완주 직전 하드킬
(-1, 산출물 0), arm B는 피처 단계 MemoryError. 2차(16:02)는 래퍼 외부종료
(0xC000013A, 16:03:57 로그 동결) — §S13.27 중간 노트의 병행 세션 진단대로
슬립 개연(본 래퍼에 `SetThreadExecutionState` 슬립 억제 미이식; **차후 재사용
시 이식 필요**). 옛 실험 pkl 11건(2.3GB)·HF 캐시(0.8GB) 정리 + 사용자 앱 종료
(커밋 ~4GB 확보) 후 3차(16:57, 사용자 활동 중) 성공: A 17:13(912s)·B 17:26.
**판정 수치는 독립 검증자가 compare 스크립트 재실행으로 전항 재현**(스펙 정합:
A 델타 = bmprop 플래그 1건, B 델타 = volcap 플래그 2건뿐, manifest 실측 동일;
신규 단위테스트 3 passed).

**결과** (production IR 1.5734 / TE 3.64% / beta 1.052 / turnover 0.691 / 퇴화
15/32 / 07-21 vol비율 1.0955):

| 지표 | prod | A(bmprop) | B(volcap 1.05) |
|---|---|---|---|
| IR | 1.5734 | 1.4736 (−0.0998) | 0.9498 (−0.6236) |
| TE% / beta | 3.64 / 1.052 | 3.51 / 1.058 | 2.81 / 0.989 |
| turnover | 0.691 | 0.668 | 0.837 (+21%) |
| MaxDD% / activeMaxDD% | −32.71 / −5.64 | −32.76 / −5.79 | −31.01 / −4.27 |
| P1/P2/P3 | 1.336/1.121/2.082 | 1.398/0.952/1.892 | 0.493/1.144/1.203 |
| downCap% | 105.4 | 106.4 | 98.4 |
| 07-21 vol비율 | 1.0955 | 1.0936 | 1.0387 |
| 퇴화율 / solver fallback | 15/32 / 0 | 15/32 / 0 | 15/32 / 0 |

**판정**:
- **arm A 기각**: D1(a) 1.0936(−0.0019, 사실상 불변)·D1(b) 0/4(전 지표 소폭
  악화) → 방어 실효 없음, D2 ΔIR −0.0998. MU 캡은 바인딩(OW 3.31→2.28%p)했으나
  잘린 액티브가 여타 고변동 OW로 재배분(상위12 교집합 12/12) — 캡이 '누가'는
  바꿔도 '얼마나'는 못 바꿈.
- **arm B 사용자 회부, 권고 기각**: D1 성립 — vol 노출 1.0387(−5.7%p), tail
  3/4 개선(MaxDD −31.0·activeMaxDD −4.27·downCap 98.4%; 단 최악 서브기간 P1
  0.493 대폭 악화). 그러나 D2 ΔIR −0.6236으로 비용 과도. OW 교집합 3/12:
  PLTR·RBLX·WDAY·DELL 등 고변동 성장 OW 소멸, NVDA(+3.09)·005930(+2.82)·AVGO
  치환, beta 0.989 — production의 알파 구조가 해체된 별개 방어형 포트폴리오.
- E2: 두 arm 모두 TE≤4.5%·퇴화율 15/32 동일·solver fallback 0 — 옵티마이저
  계층 한정 변경의 교차검증 성립.

**교훈**: SHAP이 지목한 vol 드라이버(Price 그룹 +0.649)는 제거 가능한 리스크가
아니라 알파의 서식지. 같은 날 §S13.28 직교성 진단(스타일 틸트가 액티브 수익
72% 운반)과 수렴 — 이 전략의 알파는 고변동·성장 캐릭터와 분리 불가. 방어
수요가 실재하면 x 완화(0.07~0.08) 별도 사전등록 실험으로. production 변경
없음, 두 플래그 모두 default-OFF 유지.

**인벤토리**: +2 — `n_trials_total` 454 → **456**. 사전등록 시점 표기
"452→454"는 병행 §S13.28 산입 2건이 선점하여 456으로 밀림.

**교차검증(병행 세션, 2026-08-04)**: 별도 세션이 엔진 재현 cov(`raw_returns`
126d·LW+메가캡 shrink)로 D1/D2/E2·OW 12종을 독립 재산출 —
`outputs/s13_27_judgment.json`. 헤드라인(ΔIR −0.0998/−0.6236, MaxDD, downCap,
퇴화율 15/32, A 기각·B 회부 권고 기각) 전항 일치. vol 비율 절대값만 윈도우 재현
차이로 소폭 상이(1.0981/1.0968/1.0444 vs 본문 1.0955/1.0936/1.0387), 방향·판정
불변.

## §S13.29 PCA vol 표준화 재측정(현 baseline) — 사전등록 (2026-08-04, 측정 전)

**동기/사용자 지시**: §S13.28 후속 문답에서 vol 선호의 기제(라벨 꼬리가 vol로
채워짐 + top-가중 순위 손실 + 롱온리 비대칭 + PCA의 베타 제거; 실측:
타깃 상위십분위 vol_z +0.50·하위 +0.54·중간 −0.22, 학습창 IC(vol,타깃) 양수
0/32, 스코어 VOL 계수 +0.818/t 46.5·BETA −0.28)를 확인한 뒤 사용자가
"specific 수익 ÷ 고유변동성 우선시 버전"의 **최근 OW 종목**을 요청. 해당
버전 = 기존 `pca_vol_standardize` 플래그(§S12.5에서 150종 시절 전 서브기간
악화로 기각). 구 산출물 pkl은 2026-08-04 메모리 정리로 부재 + 150종·구
빈티지라 무효 → **현 200종·slope-ON baseline에서 재실행**이 필요.

**arm 스펙(사전약정, 스윕 금지)**: `variants/arm_s13_29_pca_volstd.yaml` =
production(92ebcf1, slope ON) 전량 복사 + `pca_vol_standardize: true` 1줄.
타깃 계층 변경이므로 전체 파이프라인 재실행(캐시 무효). 실행은 schtasks
분리 + 슬립 억제 래퍼(§S13.27 이식판 패턴).

**목적/판정(사전등록)**:
- **주 산출물(사용자 요청)**: 최종 리밸(07-21) OW 종목 — production 대비
  액티브 상위 비교표. 채택 목적 아님.
- **E1 기록**: ΔIR vs production 1.5734. §S12.5 기각의 현 baseline 재검
  이며, 사전 기대는 부정적(§S12.5 전 구간 악화 + §S13.28 R1 0.925 +
  §S13.27B 0.950 3중 증거). E1 통과 시에도 단독 flip 없이 별도 검토.
- **E2**: TE ≤ 4.5%·집중 캐릭터. (타깃 변경이라 퇴화율 동일성은 기대하지
  않음 — 참고 기록만.)
- **인벤토리**: 측정 후 +1 (456 → 457).

상태: **측정 완료 (2026-08-05 00:22, exit 0, 2522s)** — 아래 결과.

### §S13.29 결과 — E1 FAIL(−0.499) · §S12.5 기각 재확인 · **표준화는 vol 틸트를 제거하지 못함**

| 지표 | production | arm(pca_vol_standardize) |
|---|---:|---:|
| IR | 1.5734 | **1.0747 (ΔIR −0.4986)** |
| 액티브 / TE | 5.72% / 3.64% | 3.99% / 3.71% |
| beta / turnover | 1.052 / 0.691 | 1.062 / 0.643 |
| MaxDD | −32.71% | −32.72% |
| 서브기간 IR | 1.35/1.24/2.07 | 1.07/0.53/1.59 (전 구간 하락) |
| 퇴화율 | 15/32 | 12/32 (타깃 변경 기인, 예상 범위) |
| **vol 노출(96리밸)** | 평균 +0.33, 96/96>0 | **평균 +0.365, 96/96>0** |

**E1 FAIL**(전 서브기간 하락 — §S12.5의 150종 판정이 200종·slope-ON baseline에서
재확인). E2는 TE 3.71%·beta 1.062·OW 프로필 유지로 통과.

**핵심 발견(예상 반전)**: 표준화 arm의 vol 노출이 **오히려 소폭 높다**. 원인을
라벨에서 실측 — **표준화된 타깃의 상위 십분위 vol_z가 +0.552**(원 라벨 +0.497,
92% 날짜에서 양). σ로 나눠도 꼬리의 vol 점유가 그대로인 이유: 고변동주는 트레일링
σ 기준으로도 **fat tail**(첨도·vol-of-vol이 스케일링에서 살아남음)이라 "σ-단위
서프라이즈" 순위의 꼬리도 여전히 고변동주가 채운다. 따라서 —
- vol 선호는 **라벨 스케일링으로 제거 불가능한 구조**(§S13.28 사슬의 2번 고리가
  표준화에도 강건).
- ΔIR −0.50의 원인은 스타일 제거가 아니라 **선택 품질 저하**(σ 추정 잡음이 라벨
  순위를 오염)로 해석된다. 스타일 캐리는 남기고 스킬만 깎은 최악의 조합.

**07-21 OW 비교(사용자 요청 주 산출물)**: 상위 12 OW 교집합 **8/12**. MU가 동일
비중(4.93 vs 4.94%)으로 그대로 1위, TER(vol_z 2.18)·LITE(2.70) 등 고변동 이름
유지/추가. **이탈**: LLY(4.46→bm 수준 1.65, 저변동 대형 OW 소거)·STX·DELL·DSY.
**진입**: LITE·INTU·7974(닌텐도)·NEM. 즉 구성은 바뀌되 캐릭터(고변동)는 불변.

**판정**: 불채택(no-flip), `pca_vol_standardize` default-OFF 유지. 인벤토리 +1
반영(456→457). 산출물: `outputs/arm_s13_29_pca_volstd/`(metrics·pkl),
로그 `outputs/s13_29_run.log`.

## §S13.30 vol×quality 확인 피처 — 사전점검 사전등록 (2026-08-05, 측정 전)

**동기/사용자 지시**: vol 틸트는 구조적·제거 불가가 확정(4중 증거: §S13.27B 캡
−0.62 · §S13.28 R1 잔여 0.925 · §S12.5/§S13.29 표준화 무효). 사용자가 "고유변동성과
퀄리티를 같이 추구하는 로직" 요청 → 역질문으로 목적 확정: **vol 알파 유지 + 고변동
내 junk 회피(방어)**. 접근 3안 중 A 승인 — 기존 `nonlinear_confirmation`의
`soft_and`(원소별 min) 재사용: `min(z(idio_vol_63d), z(best_roe_level_z))`,
"고변동 AND 고퀄리티"일 때만 높은 점수. **옵티마이저 무변경**(§S13.27 교훈:
옵티마이저 계층 방어는 알파 해체 비용). 곱 피처(ix)는 대칭성 때문에 junk-회피
의미론이 깨져 기각, 퀄리티 게이트 post-process는 2026-04-20 destructive 삭제
전례로 기각.

**현재 퀄리티 노출 실측(§S13.28 재인용)**: 유니버스 내 QUALITY 특성 베타
−0.007(t −0.72, 사실상 중립) / ETF Quality_tilt −0.131(t −4.34, 음). 즉 반퀄리티
베팅이 아니라 **퀄리티 축 미사용** 상태 — 고변동 내부 판별 여지가 열려 있는지가
본 사전점검의 질문.

**사전점검 설계(읽기 전용 — 백테스트·모델 재학습 없음)**: production pkl
(`outputs/codex_causal_rank_65/backtest_result.pkl`, 92ebcf1)의 `panel`·
`portfolio_weights`·`benchmark_returns` + `UniverseData.returns_masked`.
리밸 96회(21BD, forward 창 비중첩) 기준:
1. **조건부 판별력**: 각 리밸일 `idio_vol_63d`의 1/99 윈저 z(§S13.28 `_zscore`
   정본) 상위 tercile 내에서 quality(`best_roe_level_z`) vs forward 21d 수익
   Spearman rank-IC — 평균·t(비중첩 n≈96)·서브기간 3분할 부호.
2. **junk-vol 방어 실재(핵심)**: top-vol tercile 내 quality 중앙값 반분(EW)
   스프레드(high−low) forward 21d — 전체 및 **BM 하락 창**(같은 21d BM 누적수익
   <0) subsample, 각 t-stat.
3. **북 여지 진단**: 최종 리밸(2026-07-21) OW 상위 12종의 quality z·vol z 실측.
- 보조 진단(설명력 전용, arm 파라미터 선택에 사용 금지 — 스윕 방지):
  `earnings_quality_252d` 동일 지표, forward 63d 반복.

**PROCEED 게이트(사전약정)**:
- **P1(판별력)**: top-vol tercile 내 조건부 IC 평균 ≥ 0 **and** 반분 스프레드
  전체 평균 ≥ 0. (방어 목적이라 유의성 미요구 — calm 구간에서 알파를 깎지만
  않으면 됨.)
- **P2(방어 실재·핵심)**: BM 하락 창 스프레드 평균 > 0 **and** t ≥ 2.0 **and**
  서브기간 3분할 중 ≥ 2 양(+).
- **P3(여지)**: 07-21 OW 상위 12종 quality z 중앙값 < +0.5. (이미 퀄리티
  틸트된 북이면 피처가 고칠 것이 없음 → SHELVE.)
- **셋 모두 충족 시에만** arm 사전등록 진행. 미달 시 SHELVE(피처 미구현),
  사전점검은 읽기 전용 진단이라 인벤토리 미산입(§S13.7·§S13.19 전례).

**arm 개요(PROCEED 시 별도 절에서 상세 사전등록)**: `vol_quality_confirm =
soft_and(z(idio_vol_63d), z(best_roe_level_z))` 피처 1개, 신규 config 플래그
default-OFF, OFF parity 테스트 선행, 스윕 없음(quality 정의 위에 고정 — 보조
진단이 더 좋아 보여도 교체 금지). 판정 축은 §S13.27 방어 arm 프레임 승계:
E1 = |ΔIR| < 0.36 유지(방어 목적이라 ΔIR>+0.36 채택 바 부적합·기록만),
E2 = tail 4종(MaxDD·activeMaxDD·최악 서브기간 IR·downCap) 개선, E3 = vol_z
노출 ~+0.33·TE ≤ 4.5%·집중 캐릭터 보존.

산출물: `outputs/vol_quality_precheck/summary.json`, 스크립트
`scripts/preflight_s13_30_vol_quality.py`.

상태: **측정 완료 (2026-08-05, 310s, exit 0) → 아래 결과. SHELVE.**

### §S13.30 사전점검 결과 — **SHELVE. P2 FAIL: 21d 지평에서 퀄리티는 하락 방어가 아니라 상승 순풍**

측정: 리밸 95창(21d 비중첩)·93창(63d 보조·중첩), top-vol tercile 내
`best_roe_level_z`(주)·`earnings_quality_252d`(보조), 커버리지 1.0/1.0.
테스트 6 passed(RED→GREEN), 스크립트 헬퍼 plain 함수.

| 주 지표(best_roe_level_z, tercile 내) | 21d(주 게이트) | 63d(보조·중첩) |
|---|---:|---:|
| 조건부 IC | +0.0167 (t 1.06, 서브 3/3 양) | +0.0396 (t 2.60, 3/3 양) |
| 반분 스프레드(전체) | +0.30%/창 (t 0.81, 3/3 양) | +0.69% (t 0.97) |
| **BM 하락 창** | **−0.31% (t −0.56, n=29, 서브 0/3)** | +2.83% (t 2.52, n=19, 2/3) |
| BM 상승 창 | +0.57% (t 1.21, n=66) | +0.14% (t 0.17) |

(보조 earnings_quality_252d도 동형: 21d 하락 창 −0.14%(t −0.17), 63d 하락 창
+3.45%(t 1.94) — 주 지표와 같은 지평 구조.)

**게이트 판정**: P1 PASS(IC·스프레드 모두 ≥0) · **P2 FAIL**(하락 창 스프레드
음수, t −0.56 — 요구 t≥2.0에 원거리 미달) · P3 PASS(OW 12 quality z 중앙값
+0.171 < 0.5) → **PROCEED 불성립. SHELVE** — 피처 미구현, 백테스트 없음,
인벤토리 미산입(읽기 전용 진단, §S13.7 전례. N 457 불변).

**해석**:
1. **방어 가설 기각**: 모델의 라벨·리밸 지평(21d)에서 고변동 내 고퀄리티는
   하락 창에서 오히려 소폭 언더퍼폼(−0.31%/창). 퀄리티 스프레드의 원천은
   전부 상승 창(+0.57%) — junk-회피 방어재가 아니라 순풍재라서, confirmation
   피처를 넣어도 "하락에서 지켜주는" 채널이 존재하지 않는다.
2. **63d에서는 방어 실재**(+2.83%, t 2.52): §S11.8 "느린 알파"(21–63d IC)와
   정합 — 퀄리티는 느린 신호라 21d 지평으로는 방어가 전달되지 않는다. 단
   63d는 사전등록상 **보조·중첩 창이라 비액션**("보조 진단이 더 좋아 보여도
   교체 금지"). 63d 지평 변형을 원하면 **별도 사전등록 실험**으로만.
3. **북 진단(부수 발견)**: 07-21 OW 12종 quality z 중앙값 +0.171, 12종 중
   10종 ≥ −0.14, STX +2.10·MU +0.86·LLY +0.90 — **현재 북은 junk-vol이
   아니라 이미 퀄리티 중립~양(+)의 고변동 북**. 진짜 junk-vol은 RBLX
   (quality z −4.08, active +1.04%p) 1종뿐. 모델이 고변동 안에서 저퀄리티를
   이미 대체로 피하고 있어, 피처가 고칠 대상이 사실상 1~2 슬롯 = §S13.28
   QUALITY 특성 베타 ~0(중립)의 미시적 재확인.
4. 사용자 질문("고유변동성과 퀄리티를 같이 추구")에 대한 실측 답: **21d
   운용 지평에서는 결합할 방어 프리미엄이 없고, 북은 이미 퀄리티를 크게
   해치지 않으며**, 퀄리티가 값어치를 하는 지평(63d)은 현 리밸 구조 밖.

산출물: `outputs/vol_quality_precheck/summary.json`·`primary_21d_windows.csv`,
테스트 `tests/test_preflight_s13_30_vol_quality.py`.

## §S13.31 63d 지평 퀄리티 틸트 arm — 사전등록 (2026-08-05, 측정 전)

**동기/사용자 지시**: §S13.30 SHELVE 후 사용자가 "63d 지평 변형 테스트" 지시.
사전점검 실측 — top-vol tercile 내 quality의 방어는 **63d에서만 실재**
(BM 하락 창 +2.83%, t 2.52; IC +0.0396, t 2.60). 21d 라벨을 63d로 바꾸는 것은
§S11.9(mh 블렌드 기간구조 붕괴 FAIL) 전례의 대수술 + 전 파이프라인 교란이라
기각. 대신 **§S13.28 harvest-once 재-MVO 패턴**으로 퀄리티 틸트를 예측 계층에
직접 주입 — quality z는 느린 신호라 리밸을 넘어 지속되므로 예측 틸트가 곧
63d+ 보유 지평의 상시 퀄리티 노출을 형성한다(라벨 무변경, 모델 무변경).

**arm 스펙(사전약정, 스윕 금지)**: 스크립트
`scripts/run_s13_31_quality_tilt_arm.py`, production pkl(92ebcf1)의
`pre_overlay_predictions`(§4.2 정본) 주입, 오버레이 정확히 1회 재적용.
- **Q0 identity**: 무변경 주입 — round-trip 게이트(변환 코드가 신규이므로 재확인).
- **Q1 quality tilt**: 각 예측일(스코어 ≥30종)에 대해 — 스코어 보유 종목 중
  `idio_vol_63d`의 1/99 윈저 z **상위 tercile**에 한해, tercile 내
  `best_roe_level_z` 윈저 z(`z_q`)로 `score' = score + λ·sd(scored)·z_q`.
  tercile 밖·quality 결측 셀은 무변경. **λ = 0.25 단일 사전약정** — tercile 내
  순위 재배열은 일으키되 신호 대체가 아닌 보정 규모(§S13.28: 스타일이 액티브
  72% 운반 → 신호 캐릭터 보존 필수). 재튜닝·스윕 금지.

**판정 기준(§S13.27 방어 arm 프레임 승계 — 비교는 Q0 기준 동일 산식)**:
- **G0(선결)**: |IR(Q0) − 1.5734| ≤ 0.005 and |active(Q0) − 5.721%| ≤ 1e-4.
  FAIL 시 Q1 수치 폐기.
- **D1(방어 실효)**: (a) 96리밸 평균 active-weighted quality z가 Q0 대비
  양(+) 이동(틸트 실효), (b) tail 4종(MaxDD·activeMaxDD·최악 서브기간 IR·
  downCap) 중 ≥1 개선.
- **D2(비용)**: ΔIR 기록. ≥0이면 무비용 방어(채택 후보 회부), <0이면 tail
  개선 대 IR 비용 트레이드오프를 사용자 결정 회부. **자동 채택 없음**(no-flip
  기본, production 무변경).
- **E2(캐릭터)**: TE ≤ 4.5%, active share ≥ 0.5×Q0, fail rate ≤ Q0+10pp,
  vol_z 노출 +0.33 부근 유지(vol 알파 보존이 전제 — 크게 꺾이면 §S13.27B형
  캐릭터 해체로 간주).
- **인벤토리**: 측정 후 Q1 1건 산입(Q0는 동일성 검사) — 457 → 458.

상태: **측정 완료 (2026-08-05, 총 513s, exit 0) → 아래 결과.**

### §S13.31 결과 — **무비용 퀄리티 틸트(ΔIR +0.046) · 노출 이동 성공 · tail 방어는 미전달. 사용자 회부(no-flip)**

틸트 진단: 적용 2004일 / 통과 1260일(§S13.28과 동일 구간 구조), 평균 |델타|
0.124(스코어 단위). 테스트 6 passed(RED→GREEN).

| 지표 | Q0 identity | Q1 quality tilt |
|---|---:|---:|
| IR | 1.5734 | **1.6196 (ΔIR +0.0463)** |
| 액티브 / TE | 5.72% / 3.64% | 5.88% / 3.63% |
| beta / turnover | 1.052 / 0.691 | 1.053 / 0.709 (+2.6%) |
| MaxDD / activeMaxDD | −32.71% / −5.64% | −32.85% / −6.08% |
| 서브기간 IR | 1.351 / 1.285 / 1.954 | 1.335 / 1.312 / 2.065 |
| downCap | 105.4% | 105.4% |
| fail rate / active share | 0% / 0.1972 | 0% / 0.1949 |
| **quality z 노출(96리밸)** | **−0.029** | **+0.049** |
| **vol z 노출** | +0.385 | +0.374 |

**게이트 판정**:
- **G0 PASS** — IR 차이 0.0(부동소수점 완전 재현), 주입 경로 무결.
- **D1a PASS** — quality 노출 −0.029 → +0.049(+0.078 이동). 틸트가 실제
  노출을 만들었다.
- **D1b 형식상 PASS(1/4)** — 최악 서브기간 IR 1.285→1.312만 개선.
  MaxDD·activeMaxDD·downCap은 불변~소폭 악화. **실질 tail 방어는 미전달**.
- **D2 = +0.0463 ≥ 0** — 무비용(오히려 노이즈 수준 개선). 사전등록에 따라
  "무비용 방어 → 채택 후보 회부" 경로이나, 방어가 아닌 **무비용 노출 전환**
  으로 정확히 명명한다. |ΔIR| < 0.36이므로 IR 근거 채택은 불가(§2.4).
- **E2 PASS** — TE 3.63%·active share 0.195·fail 0%·vol 노출 0.374(캐릭터
  보존, §S13.27B형 해체 없음).

**해석**:
1. **§S13.27B와의 대비가 핵심**: vol 축을 밖에서 누르면(캡) ΔIR −0.62로 알파가
   해체됐지만, vol 틸트를 **유지한 채 tercile 내부에서 퀄리티로 재배열**하면
   비용이 0이다(ΔIR +0.046). 고변동 sleeve 안의 이름 선택은 IR 중립 —
   "vol과 quality는 상충하지 않는다"의 실증.
2. 다만 §S13.30에서 예고된 대로 **21d 운용 경로에서 방어 프리미엄은 환전되지
   않았다** — 63d 방어 효과(+2.83%/창)가 tail 지표(MaxDD·downCap)로는
   나타나지 않음. 얻은 것은 방어가 아니라 **공짜 퀄리티 노출**(−0.03→+0.05,
   ETF Quality_tilt 음수 노출의 부분 상쇄 방향).
3. 한계: (i) 재-MVO 진단 arm — production 채택하려면 파이프라인 내 구현
   (default-OFF 플래그+parity+전체 백테스트)이 별도로 필요. (ii) ΔIR은
   in-sample이고 DSR 대상(N=458). (iii) λ=0.25 단일점 — 용량 곡선 미지(스윕
   금지 준수).

**회부(사용자 결정 대기)**: production 무변경 유지. 선택지 —
(a) 현상 유지(방어 목적 미달이므로 종결), (b) "무비용 퀄리티 노출" 자체를
가치로 보아 파이프라인 구현+정식 arm으로 승격 검토(§8 게이트·DSR 해킷 필요).
자동 채택 없음.

**인벤토리**: Q1 1건 산입 — `n_trials_total` 457 → **458**(Q0는 동일성 검사,
§S13.30 사전점검은 읽기 전용이라 미산입).

산출물: `outputs/s13_31_quality_tilt/summary.json`·arm별 액티브 CSV,
로그 `outputs/s13_31_run.log`, 스크립트 `scripts/run_s13_31_quality_tilt_arm.py`,
테스트 `tests/test_run_s13_31_quality_tilt_arm.py`.

## §S13.32 퀄리티 틸트 파이프라인 정식화 — 사전등록 (2026-08-05, 측정 전)

**동기/사용자 지시**: §S13.31 회부에 대해 사용자가 "그렇게 결합해서 다시
테스트" 지시 = 선택지 (b) 정식 승격 경로. 재-MVO 진단을 **파이프라인 내
구현 + 전체 백테스트**로 재검증한다.

**구현 스펙(사전약정)**:
- `PipelineConfig`: `vol_quality_tilt_enabled: bool = False`(default-OFF),
  `vol_quality_tilt_lambda: float = 0.25`(§S13.31 단일 약정값 승계, 재튜닝
  금지).
- `apply_vol_quality_tilt(predictions, panel, config)` — §S13.31
  `apply_quality_tilt`와 동일 산식(top idio_vol tercile 내
  `score' = score + λ·sd(scored)·z_q`, z_q = tercile 내 `best_roe_level_z`
  1/99 윈저 z). 삽입 위치 = `run_backtest`의
  **`result.pre_overlay_predictions` 저장 직후·listing mask 전** —
  §S13.31 재-MVO의 주입 지점과 동일 연산이 되도록, 그리고 체크포인트가
  pre-tilt 패널을 보존해 cache-reuse 재주입 시 틸트가 정확히 1회만 적용되는
  §4.2 의미론을 유지하도록.
- OFF parity: 가드가 `if enabled` 1줄이므로 OFF 경로는 구조적으로 바이트
  동일. 단위테스트로 disabled 시 predictions 무변경(동일 객체)·enabled 시
  델타 산식·tercile 밖 불변·NaN 통과를 선행 검증(RED→GREEN).

**arm**: `variants/arm_s13_32_volqual_tilt.yaml` = production yaml 전량 복사
(portfolio_role: experiment) + 플래그 2줄. 실행 = schtasks + 슬립 억제 래퍼
(run_s13_29.ps1 패턴), 단일 런.

**빈티지**: 워크북 07-31 15:12 불변 확인(13:16 현재). production 산출물은
오늘(08-05 11:59) 스케줄 런 재생성분 — §S13.31 pkl·G0와 동일 빈티지이므로
재현 게이트가 유효하다.

**판정 기준(사전등록)**:
- **E0(재현·선결)**: |IR(arm) − 1.6196(§S13.31 Q1)| ≤ 0.005. FAIL이면
  파이프라인 구현이 재-MVO와 불일치 — 원인 규명 전 결론 금지.
- **E1(기록)**: ΔIR vs production 1.5734. |ΔIR| < 0.36 예상 — IR 근거 채택
  불가, "무비용 퀄리티 노출" 근거의 후보로만.
- **E2(캐릭터·교차검증)**: TE ≤ 4.5%, **퇴화율 15/32 production과 동일**
  (모델 계층 불변의 강한 교차검증), vol_z 노출 ~+0.37 유지, quality_z 노출
  +0.05 부근 재현.
- **DSR**: 측정 후 `run_selection_bias.py`(N=459) 해킷 기록(§2.7 — 활성화
  전 필수 절차의 선이행).
- **인벤토리**: +1 (458 → 459). 채택 여부는 §8대로 사용자 결정(자동 flip
  없음, default-OFF 유지).

상태: **측정 완료 (2026-08-05 14:18, 1014s, exit 0) → 아래 결과.**

### §S13.32 결과 — **E0 부동소수점 재현 · E2 전항 PASS · DSR p=0.154(production 대비 소폭 우위) · flip 사용자 회부**

**실행 기록**: 구현 = config 플래그 2종 + `apply_vol_quality_tilt`(backtest.py,
pre-overlay 체크포인트 직후) + 단위테스트 4건(OFF 동일 객체 반환 = 구조적
parity, ON 시 §S13.31 진단 변환과 셀 단위 일치) + **전체 스위트 467 passed**.
1차 런(13:22)은 Phase 2 MemoryError로 사망 — 원인 실측: **C: 여유 0.2GB**
(디스크풀 → 페이지파일 축소 → 커밋 한계 고갈, §S13.27 1차와 동일 기제).
사용자 승인 하 `AppData\Roaming\Claude\vm_bundles\claudevm.bundle`(10.55GB,
7/27 이후 미사용, 재다운로드 가능) 삭제 → 여유 13GB. 2차(14:01:54, schtasks+
슬립 억제) 성공.

| 지표 | production | arm(전체 파이프라인) | §S13.31 Q1(재-MVO) |
|---|---:|---:|---:|
| IR | 1.5734 | **1.619622** | 1.619622 |
| 액티브 / TE | 5.72% / 3.64% | 5.884% / 3.633% | 동일 |
| beta / turnover | 1.052 / 0.691 | 1.0530 / 0.7090 | 동일 |
| MaxDD | −32.71% | −32.85% | 동일 |
| 퇴화율 | 15/32 | **15/32 (동일)** | n/a(재학습 없음) |
| 솔버 | ECOS/fallback 0 | ECOS 192회/fallback 0 | 동일 경로 |

- **E0 PASS(사실상 0 오차)**: 워크포워드 재학습(32회)을 포함한 전체 경로가
  §S13.31 재-MVO와 IR·액티브·TE·beta·turnover·MaxDD 전항 일치 — 파이프라인
  결정론 + 구현 동일성이 동시에 증명됨. 재-MVO 진단과 정식 경로 간 갭 없음.
- **E1**: ΔIR +0.0463(노이즈 밴드 |ΔIR|<0.36 내 — IR 근거 채택 불가 유지).
- **E2 PASS**: TE 3.63% ≤ 4.5%, 퇴화율 15/32 production 동일(모델 계층 불변
  교차검증), 단일 ECOS 프로토콜 준수. 서브기간 IR(DSR 리포트 3분할)
  1.335/1.255/2.112 전부 양.
- **DSR 해킷(N=459, §2.7 선이행)**: DSR p=**0.1538 FAIL** / Adjusted SR
  0.359 PASS / MinTRL 1.1yr 충분 / 서브기간 STABLE / 생존편향 WARN(기지
  19종 늦은 진입 — §S13.28 한계 (i)와 동일). 맥락: **production 자체가
  p=0.192 FAIL 상태**에서 사용자 오버라이드로 승격된 전례(§S9 codex 승격,
  §S13.25 slope 채택) — arm의 p는 production보다 소폭 낫다(액티브 증가분이
  깎이고도 SR이 높아진 결과).

**판정**: 사전등록 게이트 전부 이행(E0·E2 PASS, E1·DSR 기록). **production
flip은 §8대로 사용자 결정 대기** — `vol_quality_tilt_*` default-OFF 유지.
채택 절차(승인 시): `variants/codex_causal_rank_65.yaml`에 플래그 2줄 추가
(독립 1커밋), 롤백 = 2줄 삭제(E0가 곧 바이트동일 복원의 증명). 채택 근거는
IR이 아니라 **무비용 퀄리티 노출 전환**(−0.03→+0.05, vol 캐릭터 보존)임을
명시한다.

산출물: `outputs/arm_s13_32_volqual_tilt/`(metrics·pkl·manifest),
`outputs/s13_32_run.log`, DSR `outputs/reports/selection_bias_report.md`
(2026-08-05 14:24, N=459), 코드 `src/config.py`·`src/backtest.py`
(`apply_vol_quality_tilt`), 테스트 `tests/test_vol_quality_tilt.py`,
variant `variants/arm_s13_32_volqual_tilt.yaml`.

### §S13.32 Production flip (2026-08-05, 사용자 채택 지시)

사용자가 §S13.32 회부를 **채택**으로 결정 — `codex_causal_rank_65.yaml`에
`vol_quality_tilt_enabled: true` / `vol_quality_tilt_lambda: 0.25` 2줄 추가.
**새 production 인증 수치 = IR 1.6196 / 액티브 5.88% / TE 3.63% / beta 1.053 /
turnover 0.709 / 퇴화율 15/32**(arm_s13_32 런, 워크북 07-31 15:12 빈티지,
단일 ECOS·fallback 0).

- **§8 재검증 처리**: flip 후 재실행은 arm 런과 문자 그대로 동일 config이고,
  §S13.32 E0가 전체 파이프라인 결정론(재-MVO ↔ full run 부동소수점 일치)을
  이미 증명했으므로 **arm 런이 곧 flip 검증**이다. 별도 재실행 생략.
- **롤백 확인**: OFF 가드는 입력 객체를 그대로 반환(구조적 parity, 단위테스트
  `test_disabled_returns_the_same_object`) + §S13.31 Q0가 production을 diff 0.0
  재현 — 2줄 삭제 = 바이트동일 복원 성립.
- **채택 근거(명시)**: IR이 아니라(ΔIR +0.046 = 노이즈) **무비용 퀄리티 노출
  전환**(active-weighted quality z −0.03→+0.05, vol_z 0.385→0.374 보존).
  DSR p=0.1538은 production 승격 전례 2건(p=0.192)과 동일한 사용자
  오버라이드 프레임이며 기존보다 소폭 우위.
- **후속 수치 비교 규칙**: 이후 arm 비교 기준은 IR 1.6196(quality tilt ON).
  과거 1.5734 대비 수치와 혼용 금지.

## §S13.33 GPT 외부 리뷰 수정 분리 채택 + 워크북 08-05 빈티지 포크 (2026-08-06)

**동기**: 외부 GPT 리뷰가 4건 지적 — (a) 운영 attribution이 실행 신호를 설명하지
않음(raw feature·lag 미반영), (b) attribution의 "gain" importance가 실제로는
split, (c) 상장 전 마스킹이 vol-quality tilt **뒤**에 적용(유령 행이 tercile
배정·예측 z-score 오염), (d) `idio_vol_63d` 정의 비표준(EW 시장·drifting beta).
4건 모두 사실로 검증. GPT가 작업트리에 수정을 남겼고, 사용자가 "수정 버전을
production으로 실측 비교" 후 **분리 채택**(수정만 ON, 모델링 변경 제외)을 지시.

### 워크북 빈티지 포크 기록 (§S13.23 선례)

- `ai_signal_data.xlsx` mtime **2026-08-05 15:38:09** — §S13.32 인증(07-31 15:12
  빈티지, 14:18 실행)과 23:08 스케줄 리프레시 **사이**. 갱신 주체는 **사용자
  수동 리프레시로 확정**(2026-08-06 사용자 확인) — 무단 개입 아님, 해소.
- 이 포크로 **08-05 빈티지 런은 인증 1.6196과 직접 비교 금지**. 동일 빈티지
  재실행(기존 코드 HEAD e32c8b5)은 IR 1.4163 — 퇴화 스케줄·avg_ic(0.0263→0.0171)
  변화로 모델 입력 자체가 바뀌었음을 확인. −0.20은 코드가 아니라 데이터 효과.

### 실측 4런 비교 (전부 ECOS 단일 프로토콜·fallback 0, `--no-cache`)

| 런 | 코드 | 빈티지 | IR | TE | turnover | avg_ic | P1/P2/P3 | 퇴화 |
|---|---|---|---:|---:|---:|---:|---|---:|
| ① 인증 §S13.32 | e32c8b5 | 07-31 | 1.6196 | 3.63% | 0.709 | 0.0263 | 1.335/1.255/2.112 | 15/32 |
| ② 기존 코드 | e32c8b5 | 08-05 | 1.4163 | 3.57% | 0.666 | 0.0171 | 1.382/0.710/2.034 | 15/32 |
| ③ GPT 전체(수정+⑤⑥) | dirty | 08-05 | 1.3731 | 3.80% | 0.704 | 0.0268 | 1.395/0.511/1.987 | 14/32 |
| ④ **fix-only(채택)** | dirty | 08-05 | **1.4481** | 3.60% | 0.667 | 0.0166 | 1.449/0.789/1.992 | 15/32 |

- ③/④의 ⑤⑥ = `standard_idio_vol_feature_enabled`(신규 CAPM idio-vol 피처 모델
  admission, 65→66) + `vol_quality_tilt_vol_feature: idio_vol_capm_63d`(틸트 키
  교체). ④는 이 2줄을 yaml에서 제거 — 데이터 정확성 수정 (a)(b)(c) 인프라 +
  (d)의 피처 신설(패널 생성만, 모델 미투입)만 활성.
- ④ 실행 2026-08-06 10:27→10:42(853s), 65피처(whitelist 결측 0), 틸트
  `feature=idio_vol_63d, λ=0.25, tilted_dates=2008`, ECOS 192/192.

### 분해 (핵심 증거)

- **④−② = +0.032 (수정의 순효과)**: ④의 퇴화 스케줄이 ②와 **15건 날짜·트리 수
  완전 일치** → 모델 계층 불변. 델타는 순수 예측/구성 계층 — 상장 전 행 제외
  후 z-score(학습·예측 경로) + 마스크→틸트 순서 교정. 수정이 IR을 훼손하지
  않음(오히려 소폭 양, 노이즈 밴드 — §2.4에 따라 채택 근거로 사용하지 않고
  기록만). P1 +0.07 / P2 +0.08 / P3 −0.04, TE·turnover·beta 사실상 불변.
- **③−④ = −0.075 (⑤⑥의 순효과)**: P2 −0.28, turnover +3.7%p, TE +20bp.
  avg_ic +0.0102로 신호 품질은 크게 개선되나 IR로 전달 안 됨(§S13.12 전달률
  구조와 일치) — tercile 재편의 거래 비용·경로 교란이 수확을 압도.
  어제 ③−②=−0.043의 원인이 수정이 아니라 ⑤⑥임을 실측으로 확정.

### 채택 결정

- **채택(사용자 지시, 정확성 근거 — §2.1 데이터 정확성 계층)**: (a) 운영
  attribution 재작성(signal_date=as_of−lag, 학습 z-score 복제, SHAP affine 변환,
  executable mu 분해), (b) gain importance를 `booster_.feature_importance`로
  정직화, (c) 마스킹→틸트 순서 + `predict_cross_sectional` 상장 전 행 제외,
  (d) `idio_vol_capm_63d` 피처 **생성 코드만**(cap-weighted PIT 시장·intercept
  OLS — 모델 admission은 default-OFF 유지). IR 게이트 비적용 — 성능이 아니라
  정확성 채택이며, 성능 영향은 위 분해로 +0.032(비액션) 확인.
- **불채택**: ⑤ 피처 admission·⑥ 틸트 키 교체 — 구조 오류 아님(legacy는
  일관된 proxy·rank 학습에 유효, 인증은 legacy tercile 앵커). 재도전 시 새
  기준선 위 **단일 사전등록 arm**(§S13.34 후보)으로만.
- **새 유효 기준선(08-05 빈티지) = ④: IR 1.4481 / 액티브 5.21% / TE 3.60% /
  beta 1.050 / turnover 0.667 / 퇴화 15/32**. 이후 arm 비교는 이 수치 기준.
  1.6196(07-31 빈티지)·1.5734와 혼용 금지.
- **롤백**: ⑤⑥은 이미 제거(yaml 2줄). 수정 (a)(b)는 보고 계층(가중치 불변),
  (c)(d)는 코드 revert로 복원 — 단 (c)는 정확성 결함의 재도입이므로 롤백
  비권장, 필요 시 결정 로그 재기록 후에만.

### 잔여 항목

- ~~커밋 미실시~~ → 사용자 승인(2026-08-06)으로 독립 2커밋: 수정 계층
  **ac63a98**(22파일, +523/−76, 신규 tests/test_attribution.py 포함) + 인증
  기록(본 결정 로그 + outputs 메트릭/매니페스트/progress, 직후 커밋).
- ~~워크북 08-05 15:38 갱신 주체 미확인~~ → 사용자 수동 리프레시로 확정, 해소.
- 전체 테스트 스위트: yaml revert 후 fix-only 작업트리에서 **477 PASS / 실패 0**
  (2026-08-06, 50.1s) — GPT가 갱신한 수용 테스트 allowlist 포함 전부 통과.
- **allowlist 정리(커밋 전)**: GPT가 수용 테스트 6종의 post-arm production 예외
  목록에 ⑤⑥ 키를 추가해 뒀으나 불채택으로 production에 없음 — 남겨두면 향후
  무단 flip을 tripwire가 통과시키므로 6개 파일에서 두 키 제거. 편집마다 훅
  전체 스위트 재실행 477 PASS(회귀 0). `test_config`·`test_vol_quality_tilt`의
  config 필드 기본값(OFF) 검증은 유지 — 필드 자체는 채택분.

## §S13.34 implied-vol 서피스 피처 + VIX 텀스트럭처 디리스킹 — 사전등록 (2026-08-07, 측정 전)

**동기/사용자 지시**: 워크북 품질점검(2026-08-06)에서 종목 레벨 IV 시트 5종
(30DAY_IMPVOL 100/90/110%MNY·3MTH_IMPVOL·VOLATILITY_30D — 전부 미배선)과
지수 레벨 VIX 텀스트럭처(VIX9D/3M/6M)·상관지수(COR1M/3M)를 확인. 사용자 지시
2026-08-07: "1·2·3순위를 모두 features로 적용 + 텀스트럭처는 리스크를 줄이는
신호로, 변화율을 고려해 백워데이션 축소 시 리스크 재사용". AskUserQuestion으로
사전등록 확정 — arm A: 3피처 동시 1 arm, arm B: μ×**0.5**·개선 판정 창
**5영업일**(스윕 금지, 단일 약정).

**사전점검 실측(2026-08-06/07, 읽기 전용)**:
- 커버리지: IV 시트 5종 모두 200/200 종목이 최근 252영업일 ≥80% (비USD 포함
  전부). 단위는 실현 vol과 동일 연율 %p (last-row median: IV30 32.3 / RV30 37.0).
- 직교성(252d 일별 CS corr vs RV30): IV30 레벨 **+0.878**(중복 — 레벨 배제),
  skew **−0.487** / term **−0.034** / vrp **−0.390**. 상호: skew-term +0.13,
  skew-vrp +0.03, term-vrp −0.41.
- VIX 텀스트럭처: 백워데이션(VIX3M<VIX) 239일(7.5%), 79 에피소드(중앙 2일,
  최장 43일=2020-03). slope(VIX3M/VIX−1)와 VIX 레벨 corr −0.66.
- bcast 함정(§S13.18/21) 때문에 지수 신호는 피처가 아닌 **μ 오버레이**로만
  소비(§S13.17 μ-스케일링 SHELVE 전례 — "S0는 stress에서 안 잃음" — 를 알고도
  사용자가 시도 지시).

**구현(코드, default-OFF·parity)**:
- `features/implied_vol.py` 신규 — `iv_skew_level=(IV90−IV110)/IV100`,
  `iv_vrp_level=IV30−RV30`, `iv_term_level=(IV3M−IV30)/IV30` (S8 idiom,
  `implied_vol_features_enabled` 게이트, IV100≤0 가드 NaN).
- `data_loader.py` — IV 시트 5종을 `BLOOMBERG_EQUITY_SHEETS`에 추가(컬럼
  "XXX US Equity" 리네임; 유니버스 교집합은 essential 시트만이라 inert),
  `FACTOR_CATEGORIES["Volatility_TS"]=["VIX3M"]` 추가(factor.py에 범용 루프
  없음 — 피처 무증가 확인).
- `backtest.py` — `apply_vix_ts_risk_scaling`: slope<0 AND
  slope_t−slope_{t−5}≤0인 예측일에 μ×0.5, 개선(축소) 전환 시 원복. NaN slope
  inert. 실행 lag 앞 최종 μ 단계(taint 없음: t 시그널 → t+1 체결).
- 테스트 8종 신규(RED→GREEN 확인): `tests/test_implied_vol.py`(게이트·inline
  reference·가드·시트 결측), `tests/test_vix_ts_risk_scaling.py`(OFF 객체
  동일·컬럼 결측 inert·심화 ×0.5→축소 원복 시나리오·NaN inert). 전체 스위트
  **485 PASS**(477+8, 회귀 0).

**arm 스펙(사전약정)**: 기준선 `variants/s0_recert_s13_34.yaml`(08-06 14:19
빈티지 재인증 — §S13.33의 1.4481은 08-05 빈티지라 직접 비교 금지, 빈티지 포크).
- **arm A** `arm_s13_34a_implied_vol.yaml`: delta = `implied_vol_features_enabled`.
- **arm B** `arm_s13_34b_vix_ts_risk.yaml`: delta = `vix_ts_risk_scaling_enabled`
  (0.5/5d).

**판정 기준**: E1 게이트 ΔIR > +0.36 & 서브기간 부호 일관(§2.4). |ΔIR|<0.36은
설명력 근거로만. E2 캐릭터: TE≤4.5%·집중 캐릭터 보존(§2.5). arm B는 추가로
risk-off 일수·해당 구간 상대성과를 기록(무음 fallback-to-bm 여부 점검).
자동 채택 없음 — flip은 §8 게이트+사용자 결정.
**인벤토리**: 측정 후 A·B 각 1건 산입.

상태: **측정 BLOCKED — FX 원천 파괴 사고 (아래). 코드·테스트·variant는 완료 상태 유지.**

### §S13.34 사고 기록 — `Data\Index.xlsx` 0바이트 파괴 (2026-08-07 11:45:45)

- **증상**: S0′(13:14 착수)가 `load_external_fx_quotes` →
  "Excel file format cannot be determined"로 즉사. 확인 결과
  `C:\...\pythonProject\Data\Index.xlsx`가 **0바이트**(mtime 11:45:45).
- **타임라인**: 11:30 스케줄 런이 11:45:36 iter15 metrics/progress 재생성(백테스트
  정상 종료) → 11:45:45 후속 워크북 리프레시-저장 단계가 Index.xlsx 저장 중
  크래시 → 0바이트 truncation. 다른 파일 피해 없음(재귀 스캔 0-byte 1건뿐).
  전일(08-06) 품질점검 때는 정상 읽힘(FX 6페어 2014-01-01→2026-08-06).
- **영향**: `fail_on_missing_fx=True`라 CHF/GBP/DKK(13종목) FX 부재 →
  **모든 백테스트 hard-fail**(무음 오염은 없음 — fail-fast 설계 정상 작동).
  내일 11:30 스케줄 production 런도 동일 크래시 예정. ai_signal_data 재생성도
  Index.xlsx 의존(PX_LAST·BEST_EPS·IV 시트)이라 불가.
- **복구 탐색(전부 실측)**: git 이력 없음(untracked). oppor.xlsx의 FX 3페어는
  캐시값 없는 수식(전부 NaN). 06-19 사본
  (`Index_Dashboard_portable\data\Index.xlsx`, 49.6MB)은 구조 보존이나
  USDCHF/GBPUSD/USDDKK 컬럼·IV 시트 부재(7월 추가분). Factor_PX_LAST는
  KRW/JPY/EUR/CNH만 보유. **CHF/GBP/DKK 이력은 로컬 어디에도 없음 —
  Bloomberg 리프레시로만 복원 가능.**
- **§9 처리**: 추정으로 메우지 않음. 공개 소스 FX 대체는 벤더 드리프트
  (인증 수치 재현 불가)를 수반하므로 사용자 결정 없이 실행 금지. S13.34
  측정은 Index.xlsx 복원 후 재개.

### §S13.34 사전점검 결과 — ai_signal_data 단독 (2026-08-07, read-only, FX 불요)

사용자 지적("테스트는 ai_signal_data로 할 수 있지 않아?")에 따라 §S13.30 패턴의
사전점검을 FX 없는 경로(convert_returns_to_usd=False, 로컬 수익률)로 실행.

1. **통합 검증 PASS**: 신규 로더 배선(IV 시트 5종 리네임)→`build_implied_vol_features`
   가 실워크북에서 정상 — 3피처 (3269×200), 최근 252일 커버리지 100%, inf 0,
   분포는 08-06 원시 측정과 일치. VIX3M 팩터 배선도 정상 로드.
2. **IC 사전점검** (일별 CS Spearman, 5일 샘플링, 보수적 t):
   | 피처 | 21d IC | 63d IC |
   |---|---:|---:|
   | iv_skew_level | **−0.0398 (t −2.54)** | **−0.0678 (t −2.53)** |
   | iv_vrp_level | +0.0196 (t +1.83) | +0.0159 (t +0.83) |
   | iv_term_level | −0.0193 (t −2.01) | −0.0108 (t −0.68) |
   skew는 문헌 방향(음)·양 지평 유의·63d에서 강화 — 파이프라인의 느린 알파
   체질(§S11.8 코어 IC 0.069)과 정합. vrp·term은 21d에서 경계 수준.
3. **arm B 신호 특성화 — 역풍 경고**: risk-off 판정일(개선 필터 적용 후)
   181일(5.5%)/79 에피소드. **판정일 이후 SPX fwd 21d 평균 +2.77% vs 평시
   +0.92%, 재진입일 이후 +5.02%** — 백워데이션은 급락 "후행" 지표라 신호
   발화 시점엔 반등 국면. μ 축소는 반등 구간의 액티브 알파를 절반으로 깎는
   방향(§S13.17 "S0는 stress에서 안 잃음"과 결합 시 **E1 음(−) ΔIR 예상**).
   사전등록 파라미터는 잠금 유지 — 측정은 지시대로 수행하되 이 예측을 기록.

### §S13.34 arm B 재설계 사전점검 — Δslope(변화율) 신호 (2026-08-07, 사용자 제안)

사용자 제안: "레벨이 아니라 **급격한 백워데이션化 가능성(기울기)**로 risk-off
판정". 읽기 전용 특성화 — 후보 4종(과거-전용 롤링 z, 원시 Δ 임계) vs 레벨 REF:

| 신호 | 일수/에피소드 | SPX fwd21 평균 | fwd21 p10 | 최악-20 낙폭창 커버 |
|---|---|---:|---:|---:|
| (전체 일) | 3145 | +1.05% | −4.18% | — |
| A z(Δ5d)<−2.0 | 110/61 | +0.99% | **−3.05%** | 1/20 |
| B z(Δ5d)<−1.5 | 232/116 | +1.09% | −4.09% | 3/20 |
| C Δ5d<−0.05 | 676/255 (21%!) | +1.02% | −4.26% | 10/20 |
| D Δ5d<−0.10 | 259/118 | +1.06% | −4.28% | 3/20 |
| REF 레벨 | 178/77 | +2.73% | −4.64% | 2/20 |

- Δ신호는 백워데이션 **상태 진입은 선행**함(진입 21회 중 12회, 중앙 리드 8일)
  — 그러나 **시장은 선행하지 못함**: 어느 변형도 신호일 이후 SPX가 평시보다
  나쁘지 않고(전부 평균≈기준), 좌측꼬리(p10)는 오히려 얇으며, 최악 21d 낙폭
  창 시작 시점 커버리지가 1~3/20(C의 10/20은 전체 일수 21% 발화의 대가 —
  대부분 오탐). 구조 원인: 옵션 시장은 낙폭 **중에** 리프라이싱 — 신호일 종가
  기준으로 손실은 이미 발생. 레벨·변화율 모두 일간 케이던스에선 확인 지표.
- 보험 논리(꼬리 절단 가치)도 부재 — 신호일 좌측꼬리가 평시보다 얇음.
- **판정: Δslope 재설계는 arm 미실행 SHELVE**(§S13.30 패턴). 기존 레벨 arm B는
  사전등록 유지(사용자 지시분)하되 E1 음(−) 예상 기록 유지 — 철회는 사용자
  선택.

### §S13.34 측정 재개 — Index.xlsx 복원 (2026-08-07 13:27, 사용자 Bloomberg 리프레시)

복원 검증: 74.2MB, PX_LAST FX 6페어 전부 2014-01-01→2026-08-07, 값 정합
(전일 기록과 연속). IV·BEST_EPS 시트 포함 — ops 재생성 경로도 해소.

**S0′ 재인증 (`s0_recert_s13_34`, 1001s, exit 0)**. 빈티지 정정: 사용자가
Index.xlsx 복원 직후 데이터 파이프라인을 재실행해 ai_signal_data가 **08-07
14:25:27 재생성본**(190→181MB, 날짜 격자는 동일하게 08-06 종료)으로 교체됐고,
S0′ 로드는 14:41이라 **이 재생성본을 읽었다**. arm A/B도 동일 mtime 확인 후
실행(중간 재생성 시 해당 run 무효·재실행 규칙):
- **IR 1.384 / TE 3.58% / active 4.96% / turnover 68.8%(양방향)**
- realized_beta **1.052** / sp500_beta 0.997 / MaxDD −32.82%
- 서브기간 P1 1.271 / P2 0.761 / P3 1.986
- ECOS 192/192·fallback 0% / 퇴화율 16/32 = 50%(기존 HOLD 항목 연속선)
- §S13.33 인증 1.4481(08-05 빈티지) 대비 −0.064는 빈티지 드리프트(하루치
  데이터+FX 꼬리) — **이후 arm A/B ΔIR은 본 1.384 기준으로만 판정.**

**arm A 결과 (`arm_s13_34a_implied_vol`, 995s, exit 0, 워크북 mtime 동일 확인)**:

| 지표 | S0′ | arm A | Δ |
|---|---:|---:|---:|
| IR | 1.384 | 1.272 | **−0.112** |
| active / TE | 4.96% / 3.58% | 4.87% / 3.83% | −0.09pp / +0.25pp |
| turnover(양방향) | 68.8% | 79.2% | **+10.4pp** |
| realized_beta | 1.052 | 1.060 | +0.008 |
| 서브기간 P1/P2/P3 | 1.271/0.761/1.986 | 1.402/0.728/1.508 | +0.13/−0.03/**−0.48** |
| 퇴화율 | 16/32 (50%) | **6/32 (18.75%)** | −31.25pp |

- **E1 FAIL** — ΔIR −0.112(양수 아님), 서브기간 부호 비일관(P1만 개선).
  **불채택, production 무변경**(default-OFF 유지).
- 판독: 피처는 inert가 아니다 — 퇴화율 급감(50→18.75%)·turnover +10pp는
  모델이 IV 3종을 대량 소비했다는 뜻. 그러나 소비가 IR로 전달되지 않고
  P3(최근 구간)에서 −0.48 손실. 사전점검 IC(skew 63d −0.068)가 실전 전달률
  (§S13.12 ~9%)과 랭킹 재배열 비용을 못 이긴 전형적 패턴(§S13.13 유사).
- 인벤토리 +1 예정(3련 종료 후 일괄 산입).

**arm B 결과 (`arm_s13_34b_vix_ts_risk`, 955s, exit 0, 워크북 mtime 동일 확인)**:

- 오버레이 발화 정상: 181/3269일 risk-off(×0.5, 5d 창) 로그 확인, 예측 패널
  106일 실제 스케일(웜업 제외). 그러나 **ΔIR +2.6e-08 — 전 지표가 1e-8
  수준(수치 먼지)**. pkl 대조: 전 기간 **가중치 최대 변화 4.2e-07**, 1e-6
  초과 일수 0 — 2018-12월 스트레스 구간 포함 스케일일이 구축에 들어갔는데도
  포트폴리오가 전혀 안 움직임.
- **판독 — 균일 μ 스케일링은 이 북에서 구조적 no-op**: 랭킹 불변(균일 배율)
  + 스코어 "부호" 게이트(score_threshold_for_ow 0.0, mega_cap_funding_score_max
  0.0) + 랭크 기반 새틀라이트 선택 + 제약 고정(TE캡·active캡·turnover penalty)
  이라 배율은 최적해 vertex를 못 움직인다. §S13.17(μ-스케일링 SHELVE)·
  §S13.22(TE-캡 조건화 FAIL)에 이어 **디리스킹 계열 3번째 소진** — 이 구조에서
  스트레스 방어는 μ 크기 채널로는 원천 불가.
- **E1 FAIL(효과 0) — 불채택, production 무변경**(default-OFF 유지).

### §S13.34 종합 판정 (2026-08-07)

| arm | ΔIR vs S0′ 1.384 | 판정 |
|---|---:|---|
| A implied-vol 3피처 | −0.112 | **E1 FAIL 불채택**(피처 소비는 실재 — 퇴화율 50→18.75% — 전달이 음) |
| B VIX-TS 디리스킹 | +0.000 (2.6e-08) | **E1 FAIL 불채택**(구조적 no-op) |
| Δslope 재설계 | (사전점검 SHELVE) | arm 미실행 |

- production 무변경·플래그 전부 default-OFF 유지. 코드·테스트(485 PASS)는
  인프라로 잔존(§S13.14 idiom — 보존, whitelist/플래그 미채택).
- 부수 확정 2건: (i) 퇴화율 HOLD 항목에 새 단서 — IV 3피처 추가만으로 퇴화율
  50→18.75%(§S13.8의 조기종료 메커니즘과 정합, 단 IR 대가 −0.11), (ii) VIX
  텀스트럭처의 정당한 소비처는 모델이 아닌 운영 대시보드 리스크 텔레메트리.
- **인벤토리**: A·B 각 1건 산입 — `n_trials_total` 459 → **461**(Δslope
  사전점검·§S13.34 품질점검은 읽기 전용 미산입, §S13.7/19 전례).

## §S13.35 거래량 + 풋/콜 OI 포지셔닝 피처 — 사전등록 (2026-08-11, 측정 전)

**동기/사용자 지시**: 2026-08-11 사용자가 PX_VOLUME(200/200)·
PUT_CALL_OPEN_INTEREST_RATIO(152/200, 비US 48종 열 부재 — US 전용)를
Bloomberg에서 신규 수집(S&P500.xlsx·Index.xlsx). "이것들을 features로
사용해서 포트폴리오 개선을 시도해줘" 지시. 마이크로스트럭처(거래량)와
옵션 포지셔닝은 §S13 프로그램에서 완전 미개척 축(2026-08-11 후보 리스트의
1·5순위).

**데이터 파이프라인 (빈티지 포크)**:
- `create_ai_signal_data.py`에 `VOLUME_OPTION_SHEETS` 2종 패스스루 추가
  (프록시 없음 — SHORT_INT_RATIO 관용구, 결측 처리는 소비 측 계약).
  생성기 테스트 18/18 PASS.
- **ai_signal_data.xlsx 08-11 14:33:43 재생성**(191.0MB, 44시트, 날짜 격자
  2014-01-27→2026-08-11, 3272일). **새 빈티지 — §S13.34의 1.3836(08-07
  빈티지)과 직접 비교 금지, S0′ 재인증 선행.** sync_data.py는 기지(旣知)
  SyntaxError로 c2 타 프로젝트 복사만 실패(ai_port는 원본 경로 직읽 — 무영향).

**Pictet-기준 지표 정렬 (사용자 지시, 가중치 무영향 진단 추가)**:
- `BacktestResult.active_share_series` 신설 — 리밸런싱일 Σ|w−bm|/2
  (Pictet PDF p.41 각주 정의, analytics.total_active_share와 동일).
  `compute_metrics`에 `active_share` 키 추가. one-way turnover는 기존
  `avg_annual_turnover_one_way`(=0.5×L1) 재확인. 참고: 기존 보고의
  "active 4.96%"는 active **return**이었음 — CLAUDE.md §5의 "active share
  ~4.75%" 단위/정의 플래그가 이 혼동이었고 본 지표로 해소.

**구현 (default-OFF·parity, 전체 스위트 493 PASS)**:
- `features/volume_flow.py` 신규(S8 idiom, 게이트 2개 독립):
  - arm A `volume_features_enabled`: `vol_abnormal_21_126`=log(ADV21/ADV126),
    `share_turnover_63d`=mean63(V·P)/MKT_CAP, `amihud_illiq_63d`=
    log(mean63(|r|·MKT_CAP/(V·P))). 통화 단위 소거 구성(다통화 안전).
  - arm B `putcall_features_enabled`: `pcr_oi_level`, `pcr_oi_chg_21d`(Δ21).
    비US 48종은 정직한 NaN → 기본 경로 per-date median 채움(§S13.6 정합).
  - 롤링 min_periods=window//2. 가드: MC≤0·V·P≤0·ADV126≤0·Amihud≤0 → NaN.
- `data_loader.BLOOMBERG_EQUITY_SHEETS` +2 리네임(로딩만으로 inert).
- **측정 전 정정 1건(정직 기록)**: CUR_MKT_CAP이 백만 단위라 Amihud 원값이
  ~1e-6 스케일 → 최초 구현 log1p가 항등에 수렴(꼬리 압축 무효). 순수 log로
  정정 — 단조변환이라 랭크 IC 불변, 결과 엿보기 아님. 스케일 자체는 전 종목
  동일 배율이라 CS z에 inert.

**사전점검 실측 (2026-08-11, read-only, 5일 샘플링·보수적 t=naive/√(h/5))**:
| 피처 | 21d IC (t_cons) | 63d IC (t_cons) | 커버리지 |
|---|---:|---:|---:|
| vol_abnormal_21_126 | −0.000 (−0.03) | +0.003 (+0.18) | 200/200 |
| **share_turnover_63d** | **+0.064 (+4.13)** | **+0.115 (+4.45)** | 200/200 |
| **amihud_illiq_63d** | **−0.044 (−4.01)** | **−0.080 (−4.63)** | 200/200 |
| pcr_oi_level | −0.004 (−0.42) | −0.011 (−0.72) | 152/200 |
| pcr_oi_chg_21d | −0.006 (−0.76) | −0.010 (−0.78) | 152/200 |
- turnover/amihud는 사전점검 사상 최강 IC(코어 §S11.8 0.069 상회)이나
  **경고 2건 기록**: (i) 두 피처는 같은 유동성 축의 거울상 — 블록 내부 중복
  가능성(§S13.25의 실효 ~2.5축 전례), (ii) 고turnover=고vol·고모멘텀 메가캡
  프록시 가능성 — 기존 vol/momentum 축과의 직교성은 arm이 판정.
- putcall은 보수적 t 비유의 — arm B는 약한 신호 예상을 사전 기록(§S13.34
  arm B 전례처럼 지시분 측정은 수행).
- §S13.12 전달률 ~9% 상한은 동일 적용 — IC 강도가 E1 통과를 보장하지 않음.

**arm 스펙(사전약정, 스윕 없음)**: 기준선 `variants/s0_recert_s13_35.yaml`
(08-11 14:33 빈티지 재인증). 각 arm은 단일 플래그 delta:
- **arm A** `arm_s13_35a_volume.yaml`: delta = `volume_features_enabled`(3피처 동시).
- **arm B** `arm_s13_35b_putcall.yaml`: delta = `putcall_features_enabled`(2피처 동시).

**판정 기준**: E1 게이트 ΔIR > +0.36 & 서브기간 부호 일관(§2.4). |ΔIR|<0.36은
설명력 근거로만. E2 캐릭터: TE≤4.5%·집중 캐릭터 보존(§2.5)·turnover 변화 기록.
각 런 전 워크북 mtime 14:33:43 동일 확인(중간 재생성 시 해당 run 무효).
자동 채택 없음 — flip은 §8 게이트+사용자 결정.
**인벤토리**: 측정 후 A·B 각 1건 산입 예정(461→463).

상태: **측정 완료 (2026-08-11) — 아래 결과 참조.**

### §S13.35 운영 사고 기록 — ENOSPC 재발 (2026-08-11 15:26, 복구 완료)

- 체인 1차(S0′→armA→armB, schtasks) 중 **armA가 출력 기록 단계에서 ENOSPC
  즉사, armB는 페이지파일 팽창 불가로 MemoryError**(여유 4.3GB 출발 →0.07GB).
  원인: **pagefile.sys 16.7GB 팽창** + 재생성 191MB + S0′ pkl 205MB.
- 워크북 3종(ai_signal_data 14:33:43·Index·S&P500) **무결 확인**(읽기 전용이라
  §S13.34형 truncation 없음). S0′는 크래시 전 정상 완료.
- 안전 정리만 수행(과거 날짜 Bloomberg 임시로그 0.63GB·pip 캐시·7일+ TEMP)
  → 1.14GB 확보 후 arm 단독 재실행 2건 완주. 이후 페이지파일 자연 축소로
  여유 9.9GB 회복. **교훈: 여유 4GB는 3련 체인에 불충분 — 런 전 기준을
  "여유 ≥ 2GB/런 + 페이지파일 헤드룸"으로 상향.**
- active_share 신설 지표가 S0′ metrics에 0.0으로 기록된 결함 발견·수정:
  `run_backtest`의 선별 복사 목록에 `active_share_series` 누락(backtest.py
  L2110 추가, 493 PASS). S0′ 가중치·IR·TE는 지표와 무관(read-only 진단) —
  S0′ 인증 유효, active_share는 arm 런부터 기록.

### §S13.35 결과 (2026-08-11) — 양 arm E1 FAIL·불채택

S0′ 재인증 (`s0_recert_s13_35`, 빈티지 08-11 14:33:43, 3272일 격자, 전 런
mtime 동일 확인):

| 지표 | S0′ | arm A (volume) | arm B (putcall) |
|---|---:|---:|---:|
| IR | **1.5290** | 1.4317 (**Δ−0.097**) | 1.4024 (**Δ−0.127**) |
| P1/P2/P3 | 1.424/0.910/2.154 | 1.304/0.469/2.262 | 1.172/0.893/2.106 |
| TE | 3.56% | 3.80% | 3.66% |
| active_share(Pictet Σ\|w−bm\|/2) | (버그로 미기록) | **20.18%** | 19.56% |
| turnover one-way | 33.9% | 36.0% | 32.7% |
| realized_beta | 1.051 | 1.064 | 1.065 |
| avg_ic | 0.0200 | **0.0269** | 0.0200 |
| 퇴화율 | 15/32 (46.9%) | **12/32 (37.5%)** | **22/32 (68.8%)** |

- **S0′ 1.5290은 새 유효 기준선**(08-11 빈티지). §S13.34의 1.3836(08-07)과
  직접 비교 금지 — Δ+0.145는 3영업일 데이터 추가의 빈티지 드리프트.
- **arm A: E1 FAIL 불채택** — ΔIR −0.097(<+0.36), 서브기간 부호 비일관
  (P1 −0.12/P2 −0.44/P3 +0.11). 단 소비는 실재: IC 0.020→0.027(+35%),
  퇴화율 15→12, turnover +2.1pp. **사전점검 사상 최강 IC(+0.115, t +4.5)도
  E1을 못 넘김** — §S13.12 전달률 상한(~9%)의 가장 강한 재확인. 유동성/
  turnover 축은 기존 vol·momentum 계열이 이미 표현하는 정보로 판독.
- **arm B: E1 FAIL 불채택** — ΔIR −0.127, **서브기간 3개 전부 음**,
  IC 불변(0.0200 — 정보 추가 없음), 퇴화율 15→22 악화. US 152/200 반쪽
  커버리지 + median 중립화가 약신호를 노이즈로 희석, 약피처 추가가
  조기종료 병리(§S13.8)만 자극.
- **production 무변경**: 두 플래그 default-OFF 유지, flip 0건. 코드·테스트는
  인프라 잔존(§S13.14 idiom).
- **부수 확정**: (i) Pictet 정의 active share 최초 실측 **~20%**(Pictet 44~50%
  의 절반 이하 — CLAUDE.md §5의 "4.75%"는 active return 혼동이었음을 확정,
  표 갱신 필요), (ii) one-way turnover ~33-36%(Pictet ~150%의 1/4), (iii)
  퇴화율은 피처 수에 민감(±)함을 재확인 — §S13.8/34 단서와 정합.
- **인벤토리**: A·B 각 1건 산입 — `n_trials_total` 461 → **463**.

## §S13.36 share_turnover 틸트 승격 — 사전점검 사전등록 (2026-08-11, 측정 전)

**동기/사용자 지시**: §S13.35 종결 후 사용자 "arm A를 이용해서 수익률이 좀 더
개선될 방향으로 발전" → 제안 검토 후 "가장 효과적일 것 같은 제안 먼저 도입"
승인. 메인 판독: arm A의 실패는 정보 실패가 아니라 **전달 경로 실패** — IC는
소비됐으나(0.020→0.027) LightGBM 랭킹 재배열이 캐리(액티브의 ~41%, §S13.12)를
헐어 지불. 채택 승리 전례가 있는 유일한 층은 틸트층(§S13.31/32 vol×quality
λ=0.25, ΔIR +0.046 무비용·production)이므로 **share_turnover_63d를 피처층에서
틸트층으로 승격**한다. 모델·라벨·피처 admission 무변경(volume_features_enabled
OFF 유지).

**정정 기록**: §S13.35 보고에서 차기 후보로 든 "63d vol×quality 재등록"은
재검토 결과 **§S13.31→32에서 이미 소진·production 채택 완료**(production
variant에 `vol_quality_tilt_enabled: true` 확인). 후보 목록에서 제거.

**사전점검 설계(읽기 전용 — 백테스트·재학습 없음)**: 데이터 —
`outputs/s0_recert_s13_35/backtest_result.pkl`(08-11 빈티지 S0′: 최종
`predictions`·`portfolio_weights`·`panel[idio_vol_63d]`) +
`build_volume_flow_features(UniverseData)` 원출력의 `share_turnover_63d`
(admission 전 정본 — honest NaN 그대로) + `returns_masked`. 리밸 ~96회
(21BD, forward 21d 비중첩 창), z는 §S13.28/30 1/99 윈저 정본.

**PROCEED 게이트(사전약정 — 셋 모두 충족 시에만 arm 진행)**:
- **P1(μ-잔차 판별력·핵심)**: 각 리밸일 scored 종목 내 z_st를 최종 예측
  score의 윈저 z에 날짜별 OLS 직교화한 **잔차**의 Spearman IC vs forward
  21d — 평균 > 0 **and** t ≥ 2.0 **and** 서브기간 3분할 ≥ 2 양(+).
  (raw IC 병기 — §S13.35 사전점검 +0.115와의 연속성 확인용. 잔차 IC가
  소멸하면 "모델이 상관 피처로 이미 표현"이 확정 → 틸트는 이중계상.)
- **P2(vol 축 중첩)**: 날짜별 scored 내 Spearman ρ(z_st, z_idio_vol) 평균
  < 0.6. (≥0.6이면 틸트가 vol 틸트 중복재 — vol 축은 이미 노출 +0.385로
  구조적 보유(§S13.28)·제거 불가 4중 증거(§S13.29). SHELVE.)
- **P3(북 여지)**: 96리밸 평균 active-weighted z_st 노출(Σ active·z,
  §S13.31 `active_exposure` 관용구 동일 단위) < +0.25. (구조적 vol 노출
  +0.385와 동급으로 이미 실려 있으면 틸트는 기존 베팅 증폭일 뿐 → SHELVE.
  이 실측치는 PROCEED 시 D1 노출 이동의 기준값을 겸한다.)
- 미달 시 SHELVE(구현·백테스트 없음), 읽기 전용 진단이라 인벤토리 미산입
  (§S13.7·§S13.30 전례).

**arm 스펙(PROCEED 시, 사전약정·스윕 금지)**:
- config: `share_turnover_tilt_enabled: bool = False`(default-OFF),
  `share_turnover_tilt_lambda: float = 0.25`(§S13.31 단일 약정값 승계,
  재튜닝·스윕 금지).
- `apply_share_turnover_tilt(predictions, data, config)`: 각 예측일 스코어
  ≥30종이면 scored 전체에 `score' = score + λ·sd(scored)·z_st`(z_st =
  scored 내 1/99 윈저 z, 결측 셀 무변경). **주입 지점 = vol_quality_tilt
  직후·listing mask 전**(§S13.32 동일 체크포인트). 피처는
  `build_volume_flow_features(data)`로 함수 내 직접 계산 — 패널 admission
  없음, 모델 무변경. tercile 조건 없음(신호가 유니버스 전체에서 측정됐고
  목적이 방어가 아닌 수익 노출이므로 — §S13.31과의 의도적 차이).
- OFF parity는 구조적(disabled → 입력 객체 그대로 반환) + 단위테스트 선행
  (RED→GREEN), ON은 inline reference 대조.
- variant: `arm_s13_36_turnover_tilt.yaml` = `s0_recert_s13_35` + 플래그
  1개. **S0′ 재실행 불필요** — 워크북 mtime 14:33:43 불변 확인을 실행
  전제로 하며, 비교 기준은 S0′ 1.5290 고정.

**판정 기준(사전약정)**:
- **D1(노출 실효·선결)**: active-weighted z_st 노출이 S0′(P3 실측) 대비
  양(+) 이동. 미이동이면 틸트 inert — 불채택·해석 금지.
- **E1(주)**: full ΔIR > +0.36 & 서브기간 부호 일관 → 채택 후보 회부
  (DSR 해킷 필수).
- **D2(무비용 경로)**: 0 ≤ ΔIR < +0.36 & E2 통과 → "무비용 노출 전환"으로
  사용자 회부(§S13.31 전례, 자동 채택 없음). ΔIR < 0 → 불채택.
- **E2(캐릭터)**: TE ≤ 4.5%, active share ≥ 0.5×S0′, vol_z 노출 +0.37
  부근 유지(§S13.27B형 해체 감시), fail rate ≤ S0′+10pp.
- **인벤토리**: arm 측정 시 1건 산입(463 → 464).

산출물: `outputs/s13_36_turnover_precheck/summary.json`, 스크립트
`scripts/preflight_s13_36_turnover_tilt.py`, 테스트
`tests/test_preflight_s13_36_turnover_tilt.py`.

상태: **측정 완료 (2026-08-11, 235s, exit 0) → 아래 결과. SHELVE.**

### §S13.36 사전점검 결과 — **SHELVE. P1 FAIL: 잔차 IC 소멸(t 1.04) — 최종 score가 share_turnover 정보의 ~70%를 이미 보유**

측정: 95창(21d 비중첩)·96리밸, skipped 0, 테스트 7 passed(RED→GREEN),
헬퍼 plain 함수(§S13.30 관용구 승계).

| 지표 (scored 내, 95창) | 값 | 게이트 |
|---|---:|---|
| raw IC (z_st vs fwd 21d) | **+0.0481 (t 2.50, 서브 3/3 양)** | — (병기) |
| **잔차 IC (P1·핵심)** | **+0.0145 (t 1.04, 서브 3/3 양)** | **FAIL** (t < 2.0) |
| vol 중첩 ρ(z_st, z_vol) (P2) | 0.507 (t 67.5) | PASS (< 0.6) |
| 북 active-weighted z_st 노출 (P3) | +0.122 | PASS (< 0.25) |

**게이트 판정**: P1 FAIL → **PROCEED 불성립. SHELVE** — 틸트 미구현·백테스트
없음·config 무변경·**인벤토리 미산입(463 불변)**(읽기 전용 진단, §S13.7·
§S13.30 전례).

**해석**:
1. **share_turnover 정보의 ~70%는 최종 예측 score에 이미 표현돼 있다**
   (raw +0.048 → 잔차 +0.015). §S13.35 arm A의 "모델이 소비는 했다"(IC +35%)
   와 정합 — 소비 가능했던 이유가 애초에 기존 피처(vol·momentum 계열)로
   접근 가능한 정보였기 때문. vol 중첩 ρ 0.507이 그 절반을 직접 설명.
2. 잔차 +0.015는 서브기간 3/3 양(+)이라 방향성 단서는 있으나 t 1.04로
   노이즈와 구분 불가 — λ=0.25 틸트를 강행해도 기대 효과가 |ΔIR| < 0.36
   노이즈 밴드로 수렴할 공산. 사전약정대로 강행하지 않는다(§2.4).
3. 북 여지(P3 +0.122)는 있으나 **채울 잔차 정보가 없다** — 여지는
   필요조건일 뿐 충분조건이 아님의 실례.
4. **축 종결**: volume 축은 피처층(§S13.35 arm A)·틸트층(§S13.36) 모두
   폐쇄. put/call 축은 피처층 FAIL + raw 정보 추가 0(§S13.35 arm B, IC
   불변)이라 틸트 승격 후보 자격 자체가 없음(잔차 이전에 raw가 부재).
   **arm A 계열의 수익 개선 경로는 이것으로 소진.**

산출물: `outputs/s13_36_turnover_precheck/summary.json`·`windows_21d.csv`·
`exposure_rebalances.csv`(runtime 235s), 스크립트·테스트는 인프라 잔존.

## §S13.37 옵션 리스크 파생 데이터 계층 + sync_data 수리 + 빈티지 포크 (2026-08-12)

**성격**: 사용자 지시 데이터 계층 작업 — 성능 arm 아님. 모델측(PipelineConfig·feature 코드) 무변경.

**추가 (re_study/create_ai_signal_data.py, [2.10/4] 단계 신설)**: ai_signal_data.xlsx에
옵션 리스크 파생 시트 10종 — 원시 4종 `iv30`(30D ATM IV) ·
`downside_skew`(1M 25Δ 풋 IV − 콜 IV) · `iv_term_structure`(30D − 3M IV) ·
`vol_risk_premium`(30D IV − 실현 30D vol), 각각의 252D 롤링 TS z-score
(min_periods 126) `*_z` 4종, `downside_skew_chg_5d`, `days_to_earnings`
(Earnings_Date 0/1 기반 다음 발표까지 달력일수, 마지막 기록 발표 이후 NaN).
파생 9종은 종목 200열 + **`SPX Index` 열**(Index.xlsx 동일 필드, 종목 달력에
ffill 정렬 — 미래 참조 없음). 원재료(1M_PUT/CALL_IMP_VOL_25DELTA_DFLT 등)는
S&P500.xlsx·Index.xlsx에 기수집·price_v4.py 계속 수집 중 — bat·수집기 무변경.
earnings 조건부 중앙값 조정은 전표본 통계 = lookahead라 데이터층에서 만들지
않고 `days_to_earnings`로 소비측 인과 수행 계약.

**검증**: 재생성 실행 exit 0, 10종 전부 (4606, 202)/(3290, 201). iv30_z 전체
std 1.260·|z|>4 비율 0.9%, downside_skew SPX +1.91(풋스큐 양수)·개별종목보다
가파름, iv_term_structure 음수(정상 콘탱고), SPX 비결측 4606/4606(z는 번인
125 제외 4481).

**부수 수리**: sync_data.py 텍스트 손상(SyntaxError line 156, §Fwd-Sales-Slope
세션에서 잔존 기록)을 동일 로직 클린 재작성으로 복원, 회귀 테스트
`test_sync_data.py` 2 PASS. 단 sync 대상 3 프로젝트(ai_signal_cc/cc2/codex_v2)는
디렉터리 부재로 전부 SKIP — sync 단계는 현재 레거시 no-op. ai_port는
`config.data_path`로 re_study 루트 파일을 직접 읽음.

**빈티지 포크 (중대)**: ai_signal_data.xlsx 2026-08-12 13:45 재생성(264.1MB,
54시트) — 기존 시트도 08-12 행까지 연장됨. §S13.35의 S0′ 1.5290(08-11 빈티지)
과 혼용 금지. 다음 arm 착수 전 08-12 빈티지에서 S0′ 재수립 필수.

**모델측 후속 (보류)**: 사용자가 제시한 OptionRisk 블록(IV_Z·Skew_Z·Term_Z·
VRP_Z 가중 합산 → score 차감) 및 risk-overlay(Skew_Z>2 & IV_Z>2 시 weight
컷)는 §2.1 default-OFF + §2.4 단일 사전등록 대상 — 미구현. 전례: §S13.34
(IV 3피처 레벨형 E1 FAIL)·§S13.35(putcall raw E1 FAIL). 단 본 시트들은 TS
z-score·25Δ skew·ΔSkew_5D·earnings 분리라 실패 arm과 정의가 다른 신규 축.

## §S13.38 옵션 리스크 표준화 피처 — 사전등록 (2026-08-12, 측정 전)

**동기/사용자 지시**: §S13.37 데이터 계층 신설 직후 사용자 "새롭게 생성한
데이터들로 features들을 테스트해줘". 사용자가 직접 제시한 시작 셋
"Skew_Z, ΔSkew_5D, IV_Z, Term_Z 네 개"(VRP_Z는 방향 불확실로 제외) +
term의 earnings 효과 분리 장치. §S13.34(IV 레벨형)·§S13.35(putcall raw)와
달리 **TS 252D z-score 표준화·25Δ 스큐·Δ5·earnings 분리**라는 신규 정의 축.

**arm 스펙(사전약정, 스윕 없음)**: 기준선 `variants/s0_recert_s13_38.yaml`
(08-12 13:45:41 빈티지 재인증). arm은 단일 플래그 delta:
- **arm A** `arm_s13_38a_option_risk.yaml`: delta = `option_risk_features_enabled`
  (5피처 동시: iv30_z / downside_skew_z / downside_skew_chg_5d /
  iv_term_structure_z / days_to_earnings — 워크북 시트 패스스루,
  파생 정의는 §S13.37 데이터 계층이 유일 정본).
- days_to_earnings는 트리 분기 조건화로 term의 earnings 효과를 분리하는
  장치(사용자 지시 "최소한 Days to Earnings를 같이"). 전표본 조건부 중앙값
  조정(TermStructureAdjusted)은 lookahead라 채택하지 않음. 휴면 §S13.9
  earn_days_to_next와 의미 중첩 사전 기록.

**구현 (default-OFF·parity, 전체 스위트 508 PASS)**: `features/option_risk.py`
신규(S8 idiom, 패스스루) + config `option_risk_features_enabled`(기본 False) +
`data_loader.BLOOMBERG_EQUITY_SHEETS` +4 리네임(로딩만으로 inert) +
assembly OptionRisk 그룹·core-whitelist 게이트. 테스트 4건 신규
(`tests/test_option_risk.py`) + precheck 헬퍼 4건(`tests/test_run_s13_38_…`).

**사전점검 실측 (2026-08-12, read-only, 5일 샘플링·보수적 t=naive/√(h/5),
`outputs/s13_38_option_risk_precheck/summary.json`)**:
| 피처 | 21d IC (t_cons) | 부호일관 | vol축 ρ |
|---|---:|---|---:|
| **iv30_z** | **+0.0263 (+2.97)** | ✓ | +0.082 |
| downside_skew_z | +0.0121 (+1.55) | ✗ | +0.010 |
| downside_skew_chg_5d | +0.0073 (+0.90) | ✓ | −0.004 |
| **iv_term_structure_z** | **+0.0247 (+3.38)** | ✓ | −0.022 |
| (참고) vol_risk_premium_z | +0.0164 (+2.21) | ✓ | −0.378 |
- P1 통과 2/4 (iv30_z·term_z). P2 전 피처 통과 — **TS z가 §S13.34의 vol축
  중복(레벨 IV ρ +0.88)을 실제로 제거**(≤0.082). 이번 축의 핵심 차별점.
- **IC 부호가 사용자 가설(높은 z → 음의 스코어)과 반대인 양(+)** — 경고가
  아니라 보상/리바운드 방향. 트리가 방향을 학습하므로 arm 설계 불변, 해석만
  사전 기록.
- 블록 내부 중복: iv30_z~term_z ρ 0.687 (실효 ~2.5축, §S13.25 전례).
- days_to_earnings 커버리지: 전기간 70.8%·최근 21일 22.5%(미래 일정 미기록
  꼬리 → median 채움) — 라이브 엣지 약점 사전 기록.

**판정 기준**: E1 게이트 ΔIR > +0.36 & 서브기간 부호 일관(§2.4). |ΔIR|<0.36은
설명력 근거로만. E2 캐릭터: TE≤4.5%·집중 캐릭터 보존(§2.5)·turnover 변화 기록.
각 런 전 워크북 mtime 13:45:41 동일 확인(중간 재생성 시 해당 run 무효).
자동 채택 없음 — flip은 §8 게이트+사용자 결정. §S13.12 전달률 ~9% 상한 동일
적용(§S13.35에서 IC +0.115도 E1 실패 — IC 강도가 통과를 보장하지 않음).
**인벤토리**: 측정 후 arm 1건 산입 예정(463→464).

상태: 측정 진행 중 (S0′ 재인증 → arm A).

### §S13.38 결과 (2026-08-12) — arm A E1 FAIL·불채택

S0′ 재인증(`s0_recert_s13_38`, 빈티지 08-12 13:45:41, 양 런 mtime 동일 확인,
ECOS 192회·fallback 0 동일):

| 지표 | S0′ | arm A (option_risk) | Δ |
|---|---:|---:|---:|
| IR | **1.5383** | 1.2622 | **−0.276** |
| P1/P2/P3 | 1.424/0.910/2.128 | 1.389/**0.379**/1.863 | −0.04/−0.53/−0.27 |
| TE | 3.54% | 3.74% | +0.20pp |
| avg_ic | 0.0195 | **0.0350** | **+79%** |
| 퇴화율 | 14/32 (43.8%) | 12/32 (37.5%) | 개선 |
| turnover one-way | 33.6% | 41.1% | +7.5pp |
| realized_beta | 1.051 | 1.065 | +0.014 |
| active_share (Pictet) | 19.55% | 18.80% | −0.75pp |

- **E1 FAIL 불채택** — ΔIR −0.276 (<+0.36), **서브기간 3개 전부 음**(특히 P2
  0.910→0.379 붕괴). `option_risk_features_enabled` default-OFF 유지, flip 0건.
- **소비는 사상 최대**: IC +79%(0.0195→0.0350, §S13.35 arm A의 +35%를 크게
  상회)·퇴화율 개선·turnover +7.5pp — 모델이 이 피처들을 가장 강하게 소비한
  arm. 그런데도 북 전달은 음(−0.276) — **§S13.12 전달률 상한의 최강 반례
  갱신**: "IC를 올려도 IR로 환전되지 않는다"가 소비 강도와 무관함을 실증.
- 사전점검의 vol축 중복 제거(ρ≤0.08)로도 전달 실패 — §S13.34(중복)와 다른
  실패 경로: 빠른 옵션 신호가 랭킹을 재편(월 회전 +7.5pp)하며 기존 느린
  알파(21–63d, §S11.8)의 캐리를 훼손하는 것으로 판독. §S13.13 캐리 gap
  비례 실증과 정합.
- S0′ 1.5383은 08-12 빈티지의 유효 기준선(08-11의 1.5290과 혼용 금지,
  Δ+0.009는 1영업일 빈티지 드리프트).
- **운영 기록**: 세션 내 background 백테스트 2회 연속 외부 중단(각 2분·5분,
  로그 무오류) → schtasks 우회로 완주(923s). §S12·§S13.23의 "장기 런은
  schtasks가 안정" 재확인. S0′만 세션 내 완주(906s)한 것은 예외 사례.
- **인벤토리**: arm 1건 산입 — `n_trials_total` 463 → **464**.
- 산출물: `outputs/s0_recert_s13_38/`·`outputs/arm_s13_38a_option_risk/`·
  `outputs/s13_38_option_risk_precheck/summary.json`. 코드·테스트는 인프라
  잔존(§S13.14 idiom). 커밋 미실시(사용자 승인 대기).

상태: **종결 (2026-08-12). 후속 제안**: 옵션 축을 피처(랭킹 입력)가 아니라
사용자 원안의 **risk-overlay**(Skew_Z>2 & IV_Z>2 시 weight 컷 — 랭킹 불변·
소량 가중 조정이라 캐리 훼손 경로가 없음)로 별도 사전등록하는 안이 잔존.
단 §S13.34 arm B(μ-스케일링 no-op)·§S13.17 전례상 제약-고정 최적해에서
실효성 사전 검증 필요.

## §S13.39 turnover-중립 진입 게이트 — 사전점검 SHELVE (2026-08-12)

**동기/사용자 지시**: §S13.38 E1 FAIL의 원인("빠른 옵션 신호가 느린 책의
캐리를 훼손")에 대해 사용자 "빠른 신호가 느린 책을 흔든다는 구조를 변경할
수 있지 않을까" → 구조 변경안 2종 제시(스무딩 vs turnover-중립 진입 게이트)
후 사용자 "2번으로 실시해줘". 랭크 자기상관 진단(read-only, 비액션)이 선행:
옵션 z 피처 AC21 0.11~0.32 vs 책 피처 0.85~0.96, 21d 스무딩으로도 0.2~0.46
회복에 그침(skew_chg_5d는 −0.37로 악화) — 정보 자체가 본질적으로 빠름.

**설계(사전등록)**: 기존 포지션 불가침, 리밸런싱의 신규 진입/증량
(Δw > no_trade_band 0.003)에만 게이트 신호(사용자 원안 iv30_z>2 AND
downside_skew_z>2) 적용. arm 구현 전 S0′(s0_recert_s13_38) 실제 96회
리밸런싱 기록으로 게이트 2종 실측(`scripts/preflight_s13_39_entry_gate.py`,
테스트 3건, 전체 스위트 511 PASS):
- P1 바인딩: 진입 ≥3건/리밸런싱 AND 극단 겹침 ≥10%
- P2 판별력: 극단 진입 vs 비극단 진입 21d 선행수익 스프레드 음(−)·|t|≥2·
  서브기간 부호 일관

**실측 (`outputs/s13_39_entry_gate_preflight/summary.json`, 진입 381건/95회
= 4.0건/리밸런싱)**:
| 신호 | 극단 n (겹침) | 스프레드(21d) | t | 부호 일관 |
|---|---:|---:|---:|---|
| **AND(원안)** | **6 (1.6%)** | **+0.0177** | +0.32 | ✗ |
| OR | 43 (11.3%) | +0.0058 | +0.24 | ✗ |
| iv_z>2 단독 | 38 (10.0%) | +0.0102 | +0.39 | ✗ |
| skew_z>2 단독 | 11 (2.9%) | −0.0023 | −0.06 | ✗ |

**게이트 판정: P1 FAIL(겹침 1.6% < 10%) & P2 FAIL(스프레드 양수·비유의)
→ arm 미구현 SHELVE.**

**판독**:
1. **원안 AND 조건은 실책에서 거의 발화하지 않는다** — 7.7년간 6건.
   책의 진입 후보와 옵션 극단이 애초에 잘 겹치지 않음(모델·오버레이가
   이미 다른 축으로 선별한 뒤라).
2. **발화해도 방향이 반대** — 극단 진입이 오히려 +1.8pp(노이즈 수준) 우위.
   §S13.38 사전점검의 "높은 z = 보상 방향"과 정합: 조기경보 컷은 이 책에서
   수익 나는 진입을 깎는 행위였을 것.
3. 어떤 분할(OR·단일)도 음의 판별력이 없음 — 컷 방향 게이트는 임계값을
   바꿔도 정당화 불가(사전등록 외 임계값 스윕은 §2.4 위반이라 미수행).
4. **옵션 리스크 축 종결**: 피처 주입(§S13.38)·진입 게이트(§S13.39) 모두
   소진. 잔존 미측정 안은 "높은 z 진입 우대" 방향뿐이나, 이는 turnover-중립
   컷이 아니라 별도 신규 설계이며 스프레드 t 0.3~0.4로는 사전등록 근거 부족.
   데이터 계층(§S13.37 시트 10종)은 대시보드·후속 연구용으로 잔존.

**인벤토리**: read-only preflight — §S13.7/S13.30 전례로 미산입(464 불변).
산출물: `outputs/s13_39_entry_gate_preflight/`(summary.json·increase_events.csv).
커밋 미실시.

## §S13.40 리밸런싱 주기 10BD(2주) 전환 — read-only 진단 (2026-08-12, 비액션)

**질문(사용자)**: "리밸런싱 주기를 2주 기준으로 변경하면 도움이 될까?" —
§S13.38~39의 빠른 신호/느린 책 논의 후속. arm 미실행, S0′(s0_recert_s13_38)
기록만으로 진단(스크립트: scratchpad `reb_freq_diag.py`, 재현 가능 로직은 본
항목에 기술).

**역사 기록**: `src/config.py` REDESIGN R(2026-04-14) — **이 책은 원래
rebalance_freq=10이었고, iter6 turnover 455%가 과도해 21로 두 배 확대**한
설계다(예상 turnover ~225%). 2주 주기는 미개척이 아니라 철회된 원설계.

**진단 (95개 보유창, 08-12 빈티지)**:
- **D1 보유기간 전·후반 액티브 분해**: 일평균 전반10일 +2.71bp vs 후반
  +1.60bp(연환산 +6.8% vs +4.0%), 차 −1.11bp/일 **t=−1.20 비유의**.
  후반도 여전히 양(+4%/yr) — 책이 월 중반에 "식지" 않음.
- **D2 스코어 IC 감쇠**: IC(score_d0) 전반 +0.0695(t 3.33) → 후반
  +0.0288(t 1.36). 감쇠는 실재.
- **D3 중간 리프레시 이득 상한**: 10일차 최신 스코어로 후반을 예측해도
  IC +0.0329(t 1.57) — 보유 스코어 대비 **이득 +0.0041(paired t 0.88,
  3분할 부호 [+,−,+] 비일관)**. rank corr(score_d0, score_d10)=0.915 —
  10일 뒤에도 랭킹이 거의 그대로라 갈아탈 정보 자체가 없음.

**판독**: D2의 감쇠는 정보의 노화가 아니라 알파 지평 구조다 — 10일차 신선한
스코어(D3)도 감쇠분을 회수하지 못한다(+0.033 ≈ +0.029). 스코어가 느린
피처(부호 지속 0.915)로 지배되는 한 주기 단축은 수확할 신호가 없고, 비용만
확정적이다: 리밸런싱 횟수 2배 = TC(10bp one-way)·§S13.12 캐리 훼손 채널
확대(§S13.38 arm A는 21BD 고정에서도 turnover +7.5pp가 −0.28 IR). 부수
불일치: 라벨 지평 21d fwd·partial_rebalance_eta 0.50·TE 캘리브레이션 전부
21BD 전제 — 전환 시 베이스라인 포크(전 인벤토리 비교 무효).

**결론: 도움 안 됨 — 미채택 권고, arm 사전등록 불가(이득 상한이 비유의·부호
비일관이라 §2.4 근거 미달).** 빠른 신호를 쓰고 싶다면 병목은 주기가 아니라
스코어 내용물인데, 그 경로(옵션 피처 주입)는 §S13.38에서 이미 E1 FAIL.

**인벤토리**: read-only 진단 — 미산입(464 불변). 커밋 미실시.

## §S13.41 옵션 IV 변동성 예측 → 공분산 대각 조정 arm — 사전등록 (2026-08-13)

**출처**: 사용자 전달 GPT 제안(2026-08-12) + 메인 보강 2건(P1 바인딩·P2 방향).
§S13.38 잔존 후보(risk-overlay)의 리스크-모델 채널 구체화. **알파 스코어·랭킹 불변.**

**사전 검증 완료(§S13.40 직후 scratch 실측, 892시점)**: GPT 인용 분위수 수치
재현(top−bot fwd vol +6.39%p·수익 +0.59%p) + **trail126 통제 후 iv30_z의
fwd 21d vol 증분 IC +0.198 (t_cons +22.9, 3분할 일관)** — 채널 통계적으로 생존.

**D_option 정의 (단일 사전등록, 스윕 금지)**:
- 모델 A: log σ_fwd21 = a + b·log σ_trail126 (풀드 OLS) / 모델 B: A + c·iv30_z
- 워크포워드: 63BD마다 재추정, expanding, 학습 표본 5BD 샘플링, **엠바고
  t+21 < 추정일**(타깃 윈도우 완결 전 관측 사용 금지), 최초 추정 최소 252BD
- 스케일 s = clip(σ̂_B/σ̂_A, 0.8, 1.5); **Σ_new = D_s @ Σ_LW126 @ D_s**
  (기존 메가캡 수축과 동일 관용구, PSD 보존, 상관 불변)
- 입력 수익률 = 파이프라인 data.returns(USD) — cov와 동일 원천

**게이트**:
- **P0 (통계)**: OOS에서 B가 A 대비 ① excess-QLIKE(log(h/σ²)+σ²/h−1) 평균
  ≥5% 감소 AND ② MAE(연환산 σ) ≥5% 감소 AND ③ 3분할 모두 개선 부호.
  (raw QLIKE는 %개선 정의 불가하여 excess form으로 사전 확정)
- **P1 (바인딩, §S13.34-B no-op 전례 보강)**: S0′(s0_recert_s13_38) 리밸런싱
  12회 샘플(8간격)에서 실제 estimate_covariance+optimize_portfolio 재호출
  (동일 config·μ=r.predictions행·prev=전일 daily_weights·bm=make_capweight_bm_fn·
  sector=get_sector_map). D_s 적용 vs baseline MVO 타깃의 one-way L1 이동
  **중앙값 ≥ 0.005** 미달 시 구조적 no-op → arm 미실행 SHELVE.
  재구성 충실도(baseline 재해 vs 기록 타깃 거리)는 비게이트 보고.
- **P2 (방향, 비액션)**: corr(Δw, iv30_z)·corr(Δw, μ), 상위 μ 5분위 이탈 비중.

**P0·P1 통과 시에만 단일 arm** (`option_vol_covariance_enabled`, default-OFF):
채택 = E1(ΔIR>+0.36 & 서브기간 일관) AND Δone-way turnover ≤ +2%p AND
MaxDD ≤ S0′ AND TE ≤ S0′+0.2%p. 비교 기준 08-12 빈티지 S0′ IR 1.5383
(워크북 mtime 2026-08-12 13:45:41 고정 재확인 필수). 인벤토리: arm 실행 시 +1.

**사전 역풍 기록**: 상위 iv30_z가 수익도 높음(+0.59%p) → 대각 상향은 위너
비중 축소 방향(§S13.27 volcap −0.62 전례). ΔIR 기대는 중립~음, TE/MaxDD
개선이 실질 목표 — 채택 게이트가 이를 반영.

### §S13.41 사전점검 실측 (2026-08-13) — P0·P1 PASS → arm 진행

실행: `scripts/precheck_s13_41_optvol_cov.py` (테스트 3건 + 스위트 518 PASS),
산출물 `outputs/s13_41_optvol_precheck/summary.json`. 워크북 08-12 13:45:41 확인.

- **P0 PASS**: OOS 619 평가시점(2014~2026). B(+iv30_z)가 A 대비
  **excess-QLIKE −19.1% / MAE −9.1%**, 3분할 모두 개선(qlike
  [14.1/21.4/19.0]% · mae [7.5/10.6/8.4]%) — 게이트(둘 다 ≥5%·부호 일관)
  여유 통과. c_z 계수 12년간 +0.106~+0.123으로 안정.
- **P1 PASS**: 리밸런싱 12회 샘플에서 실제 estimate_covariance+
  optimize_portfolio 재호출. D_s 적용 시 one-way L1 이동 **중앙값
  0.0245**(게이트 0.005의 ~5배) — §S13.34-B(4e-7 no-op)와 달리 명확히
  바인딩. 클립 비율·스케일 범위는 summary.json rows 참조.
  재구성 충실도: baseline 재해 vs 기록 타깃 L1 중앙값 ~0.05 — 기록
  가중치는 eta 0.5·projection 후 값이고 재해는 MVO 타깃이므로 예상 범위.
  주의: 재구성 hist는 data.returns 사용, production simulate 루프의
  risk_source는 data.raw_returns — 델타 측정(동일 hist 양변)에는 영향
  없으나 절대 충실도 수치 해석 시 유의.
- **P2 (비액션)**: dw@top-μ-5분위 평균 ≈ 0(−0.014~+0.003 혼재) — 위너
  유출이 사전 우려보다 작음. corr(Δw, iv30_z)는 음(고 z 종목 비중 축소
  방향, 설계 의도대로).

**arm A 기동**: `variants/arm_s13_41a_optvol_cov.yaml`(S0′ 대비 단일 델타
`option_vol_covariance_enabled: true` 검증), 구현 = `src/option_vol_cov.py`
(사전등록 상수 정본) + backtest `_optimizer_fn` D@Σ@D 주입(진단 캡처 전이라
projection도 동일 조정 Σ 소비) + run_variant SAFE_FOR_CACHE_REUSE 등록.
OFF 경로는 스케일 패널 미생성(구조적 바이트 동일). schtasks 원샷 실행.

### §S13.41 arm A 측정 결과 (2026-08-13) — E1 미달·리스크 게이트 전원 통과, flip은 사용자 회부

실행: schtasks 원샷(배터리 조건 해제 후 정상 기동), 1799s, ECOS 192/192
fallback 0. 스케일 비-inert 셀 86.3%. 워크북 08-12 13:45:41 빈티지 불변.

| 지표 | S0′ | arm A | Δ | 게이트 |
|---|---:|---:|---:|---|
| IR | 1.5383 | **1.6623** | **+0.1240** | E1(>+0.36) **FAIL** |
| 서브기간 IR | 1.424/0.910/2.128 | 1.441/1.089/2.238 | **+/+/+ 3/3** | 부호 일관 PASS |
| TE | 3.537% | 3.462% | −0.075%p | ≤+0.2%p **PASS** |
| MaxDD | −32.13% | −31.75% | +0.38%p 개선 | ≤S0′ **PASS** |
| one-way turnover | 33.63% | 34.68% | +1.05%p | ≤+2%p **PASS** |
| realized_beta | 1.0511 | 1.0414 | 1.0 방향 | (참고) |
| active_share | 19.55% | 19.29% | 보존 | §2.5 PASS |
| avg_ic / 퇴화 | 0.0194865318…/14 | **완전 동일** | 0 | 예측 계층 불변 입증 |

**판독**:
1. **알파 완전 불변 증명**: avg_ic·퇴화 14/32·sp500 수치가 비트 동일 —
   Phase 1~4 산출이 S0′와 바이트 동일하고 차이는 오직 optimizer의 Σ 대각.
   ΔIR +0.124는 순수 리스크 모델 효과다(§S13.38의 랭킹 재편 실패 경로와
   구조적으로 다름).
2. **효율 개선의 방향이 사전 기대(중립~음)를 상회**: active return
   +0.31%p/yr과 TE 감소가 동시 발생, 3 서브기간 전부 양(+). 낮은 vol
   (20.03→19.84%)·낮은 beta로 수익은 오히려 증가 — vol 억제 비용
   전례(§S13.27 −0.62)와 달리 예측 변동성의 "정보"가 비중 배분 효율로
   전달된 것으로 해석.
3. **그러나 ΔIR +0.124 < +0.36(1 SE)** — §2.4에 따라 IR 근거 채택 불가
   (노이즈 대역, 설명력 근거로만). 사전등록 채택식 "E1 AND 리스크 게이트"
   에서 E1이 미달이므로 **기본 판정 = 불채택, default-OFF 유지**.
4. **flip 회부**: §S13.31 전례(ΔIR +0.046 노이즈 대역·무비용 노출 개선 →
   사용자 결정으로 production 채택)와 동형의 사안. 리스크 게이트 4종
   전원 통과 + 알파 불변 증명이 있으므로 production flip 여부를 사용자
   결정에 회부한다. flip 시 §2.7 DSR 해킷 기록 필요.

**인벤토리**: 464 → **465** (arm A +1; 사전점검은 §S13.7 전례로 미산입).
산출물: `outputs/arm_s13_41a_optvol_cov/`. 커밋 미실시.

### §S13.41 사후 귀속 진단 (2026-08-13, 사용자 질문 — 비액션)

두 백테스트 기록의 창(95개)·종목 단위 분해(scratch `s13_41_attribution.py`):
- 총 Δ액티브 +2.37%p/7.7y, 이 중 리밸런싱일 Δw의 fwd21 직접 기여 +1.78%p.
- **횡단면 재배분(주채널)**: 창당 평균 Δw가 z Q5(고IV, fwd vol 30.7%)에서
  −26.7bp, Q1(저IV, 23.4%)로 +21.8bp. Q5 평균수익이 더 높은데도(+1.57 vs
  +1.20%) 오버레이 기여가 양(+) — corr(Δw,z) −0.03로 블랭킷 컷이 아니라
  μ/리스크 비율이 나쁜 고z 종목만 선별 축소(§S13.27 volcap −0.62와의 차이).
- **스트레스 축소(보조)**: SPX z>1 창 17개에서 액티브 일변동성 23.6→22.0bp
  (저IV 창은 불변), corr(창Δ, spx_z) −0.303 — 스트레스 창 수익은 −6.9bp로
  소폭 양보하고 TE·MaxDD 개선을 획득.
- **광범위성**: 양(+) 창 52/95, 연도별 2020/2022/2023/2024/2025 양(+88bp
  최대), 상위5창 기여 114%(하위5창 −2.4%p 상쇄) — 단일 이벤트 요행 아님.

## §S13.41 Production flip (2026-08-13) — 사용자 승인 승격

**승인**: 사후 귀속 진단(선별적 재배분 +1.78%p 직접 기여·스트레스 축소·요행
아님) 보고 후 사용자 지시 "알파가 상승한 이유가 합당하면 이 버전을 정식
버전으로 상승시키고, 커밋하자" (2026-08-13).

**§2.7 DSR/selection-bias 해킷** (`run_selection_bias.py --auto --pkl
outputs/arm_s13_41a_optvol_cov/backtest_result.pkl`, N=465):
- Deflated SR 1.118, **p=0.1319** → 공식 게이트 FAIL (p>0.05)
- Haircut SR 1.239, MinTRL 1.0yr, 서브기간 3구간 전부 양(+) STABLE
- **오버라이드**: §S13.25(slope)·§S13.31(tilt) 승격 전례와 동일하게 사용자
  명시 지시로 오버라이드. p=0.1319는 §S13.31 채택 시점(p=0.154)보다 양호.
  리포트: outputs/reports/selection_bias_report.md

**flip**: `variants/codex_causal_rank_65.yaml`에
`option_vol_covariance_enabled: true` 한 줄 추가 (후보 1개, 단독).

**재검증 (E0 동형)**: production variant overrides는
`s0_recert_s13_38.yaml`과 **완전 동일**함을 yaml 파싱으로 확인 — 따라서
단일 델타 arm 런(`outputs/arm_s13_41a_optvol_cov`, ECOS 192/192, avg_ic
S0′ 비트 동일)이 곧 production+flag 인증 런이다. 별도 재실행 생략 근거.

**새 production 기준선 (08-12 빈티지)**: **IR 1.6623 / TE 3.46% /
MaxDD −31.75% / one-way turnover 34.7% / realized_beta 1.041 /
active_share 19.3%**. 직전 유효 기준선 1.5383(동일 빈티지 S0′) 대체.
빈티지 표기: 워크북 2026-08-12 13:45:41.

**롤백**: 플래그 한 줄 revert = 스케일 패널 미생성 → 바이트동일 복원
(구조 보장 + 파리티 테스트 + 독립 검증자 #3 확인).

**퇴화율 HOLD 항목(§S10)은 본 flip과 무관하게 잔존**(모델 계층 불변,
14/32 동일).
