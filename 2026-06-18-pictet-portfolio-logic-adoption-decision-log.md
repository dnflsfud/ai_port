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
