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
