# Pictet Quest 포트폴리오 로직 → cc2_rl 반영 — 설계 (Design Spec)

- **Date**: 2026-06-18
- **Status**: Approved (design); pending spec review → writing-plans
- **Source 자료**: `c2/ai_port/Pictet_Quest AI-driven strategies_Knowledge_20260531.pdf`
- **대상 코드베이스**: `machine/re_study/c2/ai_signal_cc2_rl`
- **운용 베이스라인**: `variants/iter15_65tkr_reb21_vtg.yaml` (cap-weighted S&P65, core-satellite, IR≈1.30 docs)

---

## 1. 목표 (Goal)

Pictet Quest AI-Driven(enhanced index) 전략에서 **포트폴리오 구성 레이어의 리스크-규율/설명력 로직**을 cc2_rl에 이식한다. 사용자 결정에 따라:

- **집중 캐릭터 유지**: 65종목 고알파(active ~4.75%, TE ~3.2%) 성격을 보존. **유니버스 확장 없음.**
- **수동 오버레이 검증 포함**: value-trap gate / growth tilt / PEAD boost를 crystal-box 관점에서 on/off 측정.
- **프로덕션까지**: 백테스트 검증 통과 후 variant/config 반영.
- **솔버**: ECOS 설치 완료 (아래 §3.1).
- **범위**: P0~P3 **전부** 적용 (factor_neutral 포함). 단 시퀀스상 P3는 마지막, 강한 anti-p-hacking 가드레일 하에.

성공의 정의: 모든 변경이 **OFF-default 플래그**로 게이트되고, **단일 솔버/단일 프로토콜**의 ablation으로 측정되며, 채택은 사전등록된 통계 바를 통과할 때만 이뤄진다. attribution이 모든 후보를 판정하는 공통 언어가 된다.

---

## 2. 현재 구조 요약 (검증됨)

- **옵티마이저** (`src/portfolio_optimizer.py`): long-only benchmark-aware MVO.
  `maximize(mu@w − risk_aversion·risk − turnover_penalty·turnover)`, `risk=quad_form(active, cov)`, `active=w−bm`.
  제약(`_build_mvo_constraints` ~250–357): `sum(w)==1`, `w≥0`, `w≤max_weight(0.15)`, `risk≤max_daily_te_var(TE 0.045)`, `turnover≤max_single_turnover(0.15)`, `w≥bm·bm_weight_floor(0.02)`, per-name OW/UW caps(`max_active_per_stock 0.12`→core-satellite `satellite_max_per_stock 0.04`), 옵션 score-gated-OW·mega-cap funding, L1 active-share≤`max_active_share`(0.50→core-satellite `2·satellite_budget=0.45`), sector deviation ±0.10. 솔버 ECOS→SCS, 실패 시 `bm_weights` fallback.
- **beta 제약 없음 / 스타일-팩터 노출 중립화 없음** (sector 그룹만). `beta_63d`는 모델 피처일 뿐 제약 아님.
- **타깃** (`src/target_engine.py`): 20d forward specific return = fwd cum return − PCA(remove PC1,PC2 over 252d) 재구성, 후 CS z-score.
- **모델**: 단일 LightGBM, 63d 재학습, walk-forward 1260d, daily 예측 + EMA blend α=0.5.
- **수동 오버레이** (`src/backtest.py` ~1250–1600, 점수에 pre-MVO 적용): value-trap gate(점수 0), growth tilt(+0.25z), PEAD boost(+0.30z, ~7d decay, 21d cutoff).
- **메트릭**: IR(primary, SE(IR)≈0.36/7.8y), TE, turnover, TC 10bps one-way. (`src/backtest.py`)
- **설정 SSOT**: `src/config.py` `PipelineConfig`(+`__post_init__`/`derive_max_active_share`), `variants/*.yaml` override.

---

## 3. 검증으로 드러난 사실 (작업 전제)

### 3.1 ECOS 솔버 — 설치 완료
- 설치 전: `installed_solvers() = CLARABEL/HIGHS/OSQP/SCIPY/SCS` → ECOS 부재 → `_solve_problem`(`portfolio_optimizer.py:182`)의 `ECOS` 시도가 매 리밸런스 SolverError → **전 구간 SCS로 풀림** (docs IR 1.30이 SCS였을 가능성).
- **조치 완료(2026-06-18)**: `pip install ecos` → `installed_solvers()`에 `ECOS` 포함 확인.
- **함의(반드시 반영)**: 이제 `_solve_problem`이 의도대로 **ECOS를 1순위 선택** → 실현 책이 기존 SCS-fallback과 달라진다. 따라서 **S0 = ECOS로 baseline 재인증**이 이후 모든 비교의 단일 기준이며, 과거 SCS 기반 수치와 직접 비교 금지.

### 3.2 Alpha attribution — 코드는 있으나 죽어 있음
- `li_three_component_attribution` / `run_attribution`이 정의돼 있으나 `run_backtest`/`run_variant`/`run_variant.py` 어디서도 호출되지 않음(self-reference만).
- 즉 P0는 **신규 알고리즘이 아니라 ~90% 배선(wiring)** 작업 + 정직성 라벨링.

---

## 4. 후보 설계 (P0 → P1 → P2 → P3)

> 공통 원칙: 전부 **OFF-default 플래그**. OFF일 때 메트릭 **바이트 동일**(off-path parity) 단위테스트. config.py는 default-OFF SSOT 유지, 활성화는 variant yaml override로만.

### P0 — Alpha-source attribution (가중치 불변, 즉시)
- **Pictet 매핑**: 액티브수익 3분해 — 선형(linear) / 상호작용(interaction, AI-specific) / 포트폴리오구성(construction). (Daul-Jaisson-Nagy, JFDS 2022.)
- **구현**:
  - config: `alpha_attribution_enabled=False`, `alpha_attribution_n_dates=8` (Attribution 블록).
  - 배선: `src/harness.py:run_variant`의 `metrics=result.compute_metrics()` 직후(line ~133, `sub_period_irs` 부착 방식) **및** `run_variant.py` CLI 양쪽. 가드 안에서 **lazy-import**(주입 config를 읽도록; `N_GRID_POINTS`가 import 시점 `DEFAULT_CONFIG`를 읽는 문제 회피). `SAFE_FOR_CACHE_REUSE`에 추가.
- **정직성 제약(필수, 미준수 시 과장)**:
  - (a) 레그 A/B는 **신호-분산 점유율**, 레그 C는 **실현수익 델타** → 별도 명명 필드, **합산 100% 아님**.
  - (b) `interaction_ratio`는 `RandomState(42)`·200행·8일 서브샘플의 clamp(≥0) 잔차 → **upper bound로만 라벨**.
  - (c) 레그 C(construction): `result.predictions`가 post-overlay라 "싼" counterfactual은 **블로커**. → **harvest-once/re-MVO-many** 경로(`scripts/run_dr_ablation.py` 클론, `precomputed_*` kwargs)로 overlay-OFF re-MVO arm 구성. `construction_active = annualize(full active) − annualize(overlay-OFF re-MVO active)`, 동일 MVO/TC/제약으로 단위 일치. **annualized active-return 델타로만** 보고(uncosted book의 construction_ir 금지).
- **검증**: OFF→메트릭 바이트동일(‘alpha_attribution’ 키 없음); ON→IR/TE/turnover/beta 불변, linear+marginal_nl+interaction ≈ 1.0; 레그 C round-trip identity(overlay off + score-비례 가중 ⇒ naive==real active).

### P1 — 수동 오버레이 정리 (crystal-box)
- **Pictet 매핑**: 모델+옵티마이저만으로 알파(수작업 휴리스틱 배제).
- **구현**: 오버레이 함수 **코드 수정 없음**. config 토글 `value_trap_gate_enabled`/`growth_tilt_enabled`/`pead_boost_enabled`(+`signal_stability_lambda`)로 ablation. 신규 `scripts/run_overlay_ablation.py`(`run_dr_ablation.py` 클론).
- **confound 2건 제거(필수)**:
  - (1) **EMA**: overlay-OFF로 한 번 harvest한 **EMA-blended overlay-free `base.predictions`**를 `precomputed_predictions`로 주입(raw 아님). `base.raw_predictions`는 stability anchor용 `precomputed_raw_predictions`.
  - (2) **이중 오버레이**: 재사용 경로가 overlay 블록을 무조건 재실행 → post-overlay 주입 시 2번 적용. 각 arm이 production EMA base에 자기 오버레이를 **한 번만** 재적용하도록.
  - 추가: `data.earnings_timeline`+Factset revision 시트 로드 assert(미로드 시 ‘on’ arm이 무음 ‘off’가 됨); per-overlay affected-cell 카운트 로그; 명시적 `overlays_on_baseline.yaml`(`value_trap_gate_enabled:true`, config 기본은 False).
- **결정규칙(사전등록 do-no-harm)**: 임계치가 전구간 피팅 → full-sample IR은 **순환**. **OOS holdout**(`enforce_oos_holdout=True`, `train_cutoff_date`)에서 marginal dIR이 **명확히 음(−)이고 서브기간 부호 일관**일 때만 제거. 그 외 KEEP/형식화. **IR승부 아닌 설명력·provenance.** on-baseline arm이 ~1.30 재현 후에만 델타 신뢰.

### P2 — Beta-중립 타게팅 (soft penalty)
- **Pictet 매핑**: beta=1.0 고정(2025 AI 기반 ex-ante beta 추정 강화).
- **구현**: `_build_mvo_constraints` sector 루프 뒤 **soft penalty**. cov-implied `beta_vec=(cov@bm_w)/(bm_w@cov@bm_w)`(기존 입력만, ex-ante, 누수 없음). 3-tuple→4-tuple 반환, **caller 2곳(437/511)만** 수정(grep 확인, blast radius 한정). objective(522) 및 projection objective(448)에 penalty 추가. 가드: `denom>1e-12 AND all(isfinite(beta_vec)) AND max(abs(beta_vec))<~5`, degenerate cov ⇒ penalty=0(fallback 아님).
- **config**: `beta_neutral_enabled=False`, `beta_neutral_penalty` 기본 **~1.0**(주의: 제안된 25는 "soft"가 아니라 active_beta를 ~0.0025로 **하드핀** → {1,5,10,25} 스윕), `beta_active_band=0.10`(declared-unused, 하드 band 보류).
- **헤드라인 진단**: cov-implied beta는 mega-cap vol-shrinkage 편향 → **실현 252d OLS 회귀 beta(on r_bm)**를 헤드라인으로 보고(penalty가 최적화하는 cov-implied 수치가 아니라 리스크위원회가 보는 수치가 실제 움직이는지). 누수 assert: `returns[t:]` 교란이 `beta_vec` 불변.
- **프레이밍**: 알파가 아닌 **리스크-규율/설명력**. EV neutral 가정.

### P3 — 스타일-팩터 노출 중립화 (마지막, 강한 가드레일)
- **Pictet 매핑**: 팩터 노출 ≈ 인덱스(알파는 stock-specific/factor-neutral).
- **구현**: active 스타일 노출(value/growth/momentum/quality/size/low-vol)을 기존 CS 피처 z-score를 loading으로 써 **soft penalty**로 0 근처 band. `loadings_by_date` 빌더를 closure로 스레딩.
- **가드레일(p-hacking 표면이 가장 큼 — CS-DR-Alpha/codex_rl 실패 전례)**:
  - **단일 penalty 사전약정**(스윕-최대-IR 선택 금지).
  - **growth+momentum 축 a priori 제외**(in-scope인 growth_tilt/PEAD와 직접 충돌하므로).
  - **non-IR 선택규칙**: "mean |active style exposure|를 고정 임계 아래로 내리는 최소 λ".
  - per-date **non-finite loading impute-to-0**(유일한 load-bearing anti-fallback 방어), applied-dates 진단.
- **예상**: TE-var 제약이 이미 체계 분산을 흡수해 **non-binding 가능성 높음**. 그 경우 "TE가 이미 중립화함"을 **유효한 crystal-box 결론**으로 기록.

---

## 5. 검증 프레임워크

### 5.1 Ablation 매트릭스 (단일 솔버=ECOS, 단일 프로토콜)
| run | 변경 | 비교 대상 | 메트릭 |
|---|---|---|---|
| **S0** baseline-resolve | production variant, ECOS, 모든 플래그 production 기본 | docs IR≈1.30(SCS) | IR, TE(≤0.045 확인), turnover, ex-post beta(cov-implied + 252d OLS) — **단일 기준** |
| **S1** attr-off-parity | `alpha_attribution_enabled=False` | S0 | IR/TE/turnover/beta **바이트동일**, ‘alpha_attribution’ 키 없음 |
| **S2** attr-on | attribution 켬(레그 A/B + 레그 C re-MVO) | S0 | 가중치 불변; attribution dict 존재, linear+marginal_nl+interaction≈1.0, construction=annualized 델타 |
| **S3** overlay grid | 2³ + all-off 순수 + on-baseline | S3-on-baseline | dIR(full+P1/P2/P3+OOS), TE, turnover, beta; per-overlay marginal |
| **S4** beta sweep | `beta_neutral_penalty∈{1,5,10,25}` | S0 | **252d OLS 회귀 beta(헤드라인)** + cov-implied; IR; TE; turnover; **fallback율(불변이어야)** |
| **S4** beta off-parity | `beta_neutral_enabled=False` | S0 | 바이트동일(penalty term int 0) |
| **S5** factor | 단일 사전약정 penalty, growth+momentum 제외 | S0 | **mean|active style exposure|/축(바인딩 체크)**, IR(guardrail only), TE, turnover, fallback율 |

### 5.2 성공 바 (채택 기준)
- **솔버 패리티**: 모든 수치 단일 문서화 솔버(ECOS), baseline·전 arm 동일 솔버.
- **attribution 패리티**: OFF→오늘과 바이트동일·키 없음; ON→메트릭 불변, 점유율 합≈1.0, interaction=upper bound.
- **IR 바(통계)**: full-period **dIR > +0.36(1 SE) & 서브기간 부호 일관**일 때만 IR 사유 채택. |dIR|<0.36=노이즈("설명력으로만"). **스윕-최대-IR 선택 금지(p-hacking)**, 후보당 단일 사전등록 penalty.
- **오버레이 do-no-harm**: OOS holdout marginal dIR 명확히 음(−)일 때만 제거.
- **TE 가드**: 실현 TE ≤ 0.045 (SCS optimal_inaccurate 드리프트 없음 — 단 ECOS 사용으로 완화).
- **fallback 가드**: fallback율이 S0 +1~2%p 초과 금지. soft-penalty 후보는 **불변**이어야(증가=NaN/Inf loading 버그 신호).
- **beta 실제 이동**: 실현 252d OLS 회귀 beta가 1.0 쪽으로 측정 가능하게 상승. penalty=25에도 안 움직이면 음의 결과로 shelve.
- **factor 바인딩**: penalized 축의 mean|active exposure| 측정 가능하게 하락. 안 움직이면 "TE가 이미 중립화"로 결론.
- **집중 캐릭터 보존**: active share ~4.75%·TE ~3.2% 부근 유지. 벤치마크 붕괴(무음 fallback/과강 penalty)는 IR 무관 **FAIL**.

---

## 6. 시퀀싱

0. **STEP 0 (완료/검증)**: ECOS 설치 확인. **S0 = production variant를 ECOS로 재인증** → 단일 비교 기준 확립. (docs 1.30과 직접 비교 금지.)
1. **STEP 1 (P0 레그 A/B)**: config 플래그 + harness/CLI 양쪽 배선 + lazy-import + SAFE_FOR_CACHE_REUSE. off-path 바이트동일 단위테스트.
2. **STEP 2 (P0 레그 C)**: re-MVO counterfactual 경로로 construction 레그. round-trip identity 검증.
3. **STEP 3 (P1 overlay)**: `run_overlay_ablation.py`. EMA·이중오버레이 confound 제거, on-baseline ~1.30 재현 후 OOS 판정. 각 arm에 attribution 부착.
4. **STEP 4 (P2 beta)**: soft penalty(4-tuple, caller 2곳). 스윕 {1,5,10,25}, 회귀 beta 헤드라인.
5. **STEP 5 (P3 factor)**: 가드레일 하 단일 penalty·growth+momentum 제외·non-IR 규칙·impute-to-0. overlay 결과를 본 뒤 노출 보유 여부 반영.
6. **STEP 6 (프로덕션)**: 바 통과 후보만 `variants/iter15_65tkr_reb21_vtg.yaml` overrides에서 플래그 flip(한 번에 1개), flip마다 S0 대비 재검증. SAFE_FOR_CACHE_REUSE 갱신.

---

## 7. 롤백 / 프로덕션 롤아웃

- **롤백**: 전 변경 OFF-default 플래그 → variant yaml(또는 config) 한 줄 revert = 바이트동일 복원(off-path-parity 테스트로 보증). beta의 4-tuple 시그니처는 penalty=0이면 objective 이미 동일 → 플래그 revert만으로 충분(시그니처 revert는 cosmetic). 후보별 독립 commit/branch.
- **롤아웃**: `config.py` SSOT는 default-OFF 유지(`bm_proportional_cap_enabled` 선례). 활성화는 variant yaml override로만(`daily_update.py:64`/`update_and_deploy.py:61`이 가리키는 프로덕션 포인터). 한 번에 후보 1개, flip 후 S0 동일 솔버 재검증. attribution은 bit-identical이므로 즉시 가능(연구/ablation은 ON, daily prod는 비용 따라 ON/OFF).

---

## 8. 실행 중 해소할 결정 지점 (블로커 아님)

- **baseline beta 전제**: 현재 책의 실현 252d OLS 회귀 beta(on r_bm)를 **착수 초기 1개 진단**으로 측정. 이미 ~1.0이면 P2는 코드 작성 전 shelve.
- **저-beta 틸트 = 알파 vs 잡음**: 2018–25 저변동성 프리미엄이면 중립화가 IR을 깎음 → ablation 점추정이 노이즈 내일 가능성 높아 **판단**이 결정.
- **OOS 윈도 검정력**: holdout이 overlay marginal-IR에 충분한 power를 주는가(sub-period SE ~√3배 ≈0.6). 부족하면 overlay 연구는 explainability-only.
- **DSR/deflation 회계**: N_trials≈403에서 DSR 통과 전무 전례 → 새 arm을 `experiment_inventory.json`+`run_selection_bias.py` 해킷에 포함할지, sub-haircut delta는 비액션 간주 입장 일관 적용할지.

---

## 9. 비범위 (Out of scope)
- 유니버스 확장(65종목 유지). ESG/Article-8 틸트(데이터 없음). 하드 beta/factor band(soft penalty만; band는 declared-unused로 예약). 모델/피처 레이어 변경(다중-GBM 앙상블 등은 별도 검토).
