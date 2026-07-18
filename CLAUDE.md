# CLAUDE.md — Pictet 포트폴리오 로직 → cc2_rl 구현 운영 계약서

이 파일은 **에이전트(Claude)가 이 계획을 실제로 구현할 때 따르는 governing 운영 계약서**다.
상세 설계·코드는 spec/plan에 있고, 이 파일은 "**절대 틀리면 안 되는 규칙 · codex 리뷰로 해소된 결정 · 게이트 · 실행 루프**"만 담는다.

> **우선순위(Precedence)**: 충돌 시 이 CLAUDE.md > `README.md`(codex 리뷰) > `…-design.md`(spec) > `…-adoption.md`(plan).
> spec/plan에 codex가 지적한 모순(아래 §4)이 남아 있으면 **이 파일의 해소안이 권위(authoritative)**다.
> 단, 상위 레포의 일반 규칙(`venv_vf_new/CLAUDE.md`, `cc2_rl/CLAUDE.md`)과 사용자 직접 지시는 항상 우선한다.

> **STATUS (2026-06-23)**: 2026-06-19 벤더링 이후 이 프로젝트의 코드 정본은
> `c2/ai_port/src`다. 모든 실행은 `c2/ai_port`에서 `PYTHONPATH=.`로 수행한다.
> 과거 `CC2` 절대경로와 `PYTHONPATH=<CC2>` 지시는 superseded다.

> **STATUS (2026-07-16)**: 유니버스 100종·unhedged USD 회계로 전환(결정 로그 §S9).
> 65종 시절 인증 수치와 직접 비교 금지. 데이터 정확성 계층(마스킹·FX 변환)은 §2.1 예외로
> default-ON이다.

---

## 0. 문서 세트와 읽는 순서

| 역할 | 파일 (이 폴더 = `c2/ai_port/`) | 정본(canonical) |
|---|---|---|
| 출처 PDF | `Pictet_Quest AI-driven strategies_Knowledge_20260531.pdf` | — |
| 설계 spec | `2026-06-18-pictet-portfolio-logic-adoption-design.md` | `cc2_rl/docs/superpowers/specs/…` |
| 구현 plan | `2026-06-18-pictet-portfolio-logic-adoption.md` | `cc2_rl/docs/superpowers/plans/…` |
| codex 리뷰 | `README.md` | — |
| **결정 로그** | `2026-06-18-pictet-portfolio-logic-adoption-decision-log.md` | `ai_port` 로컬 정본 |

읽는 순서: PDF는 **매핑 주장 검증용으로만** → spec(연구 계약) → 이 CLAUDE.md(규칙·해소) → plan을 Task 단위로 실행. 측정마다 결정 로그를 갱신한다.

---

## 1. 환경 (Environment) — 착수 전 1회 확인

- **PY** = `C:/Users/westl/PycharmProjects/pythonProject/venv_vf_new/Scripts/python.exe`
- **코드 정본(편집 대상)** = `C:/Users/westl/PycharmProjects/pythonProject/venv_vf_new/machine/re_study/c2/ai_port/src`.
- **WD(작업 디렉터리) = `ai_port`(현재 폴더)**: **모든 명령(단위테스트·백테스트·ablation 포함)을 `ai_port`에서 실행**한다.
  - 임포트: `PYTHONPATH=.` 설정. 스크립트·`--variant`·테스트 파일은 `ai_port` 로컬 경로를 지정한다.
  - 산출물(결정 로그·metrics 사본 등)은 `ai_port/outputs`에 둔다.
  - `cc2_rl` 체크아웃은 벤더링 원천 기록일 뿐이며, 현재 수정 대상이 아니다. 재동기화가 필요할 때만 `ENGINE_PROVENANCE.md` 절차를 따른다.
- **ECOS**: 설치·검증 **완료**(`<PY> -c "import cvxpy as cp; print('ECOS' in cp.installed_solvers())"` → `True`). 따라서 `_solve_problem`이 이제 ECOS를 1순위로 선택한다.
- **좀비 hang 주의**: python을 background로 반복 spawn 금지. 백테스트/ablation은 **단일 foreground**.
- **경로 정합(codex #1 해소)**: `ai_port`는 2026-06-19부터 self-contained 벤더링 엔진이다. 코드·테스트·variant·scripts는 `ai_port` 사본을 기준으로 한다.

---

## 2. 절대 불변식 (Non-negotiable invariants)

이 7가지를 어기면 결과를 **신뢰하지 말고 중단**한다.

1. **OFF-default + parity**: 모든 신규 동작은 `PipelineConfig`에서 default-OFF. OFF일 때 메트릭이 baseline과 **바이트 동일**해야 하며, 이를 단위테스트로 먼저 통과시킨다(`tests/test_*` — fixture 없는 plain 함수, `_inline_reference`/`np.allclose(...fillna(0)...atol=1e-6)` 관용구). **예외(2026-07-17 개정, §S9)**: 데이터 정확성 계층 — 상장 전 마스킹(`listing_mask_enabled`)과 FX→USD 변환(`convert_returns_to_usd` 계열) — 은 default-ON을 허용한다. OFF면 100종·다통화 유니버스에서 결과 자체가 틀리기 때문이다. OFF-default+parity는 성능 개선 목적의 연구 arm에만 적용한다.
2. **단일 ECOS 프로토콜**: baseline(S0)과 모든 arm을 **동일 ECOS 솔버·동일 경로**로 실행. 과거 SCS 기반 수치(docs IR≈1.30 포함)와 **직접 비교 금지**. 모든 보고 수치에 사용 솔버를 명시.
3. **평가지 적용이 아님(codex #3 해소)**: 본 작업은 "P0~P3 **전부 평가(evaluate)**, **게이트 통과 후보만** 프로덕션 활성화"다. "P0~P3 전부 적용(apply)"이라는 표현은 폐기. beta/factor는 shelve 게이트가 있다.
4. **IR 채택 바**: full-period **ΔIR > +0.36(=1 SE) & 서브기간 부호 일관**일 때만 IR을 근거로 채택. `|ΔIR| < 0.36`은 노이즈 → "설명력 근거로만 판단". **스윕에서 최대-IR arm 고르기 금지(= p-hacking)**; 후보당 **단일 사전등록 파라미터**.
5. **집중 캐릭터 보존**: active share ~4.75% · TE ~3.2% 부근 유지. 무음 fallback-to-bm이나 과강 penalty로 책이 벤치마크로 붕괴하면 IR과 무관하게 **FAIL**.
6. **결정 로그 우선**: baseline·ablation 결과를 프로덕션 변경 근거로 쓰기 전에 **결정 로그에 먼저 기록**(§6). 기억으로 재구성 금지.
7. **DSR/selection-bias 게이트(codex #7 해소)**: 성능에 영향 주는 후보(오버레이 제거, beta/factor penalty)는 활성화 전 `experiment_inventory.json` + `run_selection_bias.py` 해킷을 결정 로그에 기록. sub-haircut ΔIR은 비액션(설명력으로만)으로 간주하는 기존 입장을 일관 적용.

---

## 3. S0 상태 — 아직 미완 (codex #2 해소)

> **(2026-07-16 갱신, 2026-07-18 재인증)** S0의 유니버스·회계 기준은 100종·unhedged USD로
> 재정의되었다(결정 로그 §S9). §S9의 production 수치(IR 1.599/TE 4.42%)는 ex-ante TE 캡
> 0.045 시절 산출물 — 커밋된 config(캡 0.035) 기준 유효 baseline은 §S9.1: production
> IR 1.570/TE 3.72%/beta 1.054, challenger IR 1.011. 아래 65종 서술은 역사 기록으로 유지한다.

> **ECOS 설치 ≠ S0 확정.** 설치는 끝났지만 **S0(baseline 재인증)은 PENDING**이다.

S0는 다음을 실행하고 결정 로그에 기록해야 비로소 "확정"된다:

```
<PY> run_variant.py --variant variants/iter15_65tkr_reb21_vtg.yaml
```

기록 항목: `information_ratio`, `tracking_error`(≤0.045 확인), `avg_annual_turnover`, **`realized_beta`**(plan Task 0.1–0.2에서 추가). 이 S0가 이후 **모든 arm의 단일 비교 기준**이다.

**게이트(매우 중요)**: S0 `realized_beta`가 **~1.0**이면 **P2(beta-neutral)는 코드 작성 전 shelve**한다. 0.90~0.93 부근일 때만 P2 진행.

---

## 4. codex가 짚은 모순 — 권위 있는 해소 (authoritative resolutions)

구현 시 아래 해소안을 따른다. spec/plan 본문이 이와 다르면 **이 절이 이긴다**.

### 4.1 Beta 구현 (codex #5) — **inline `optimize_portfolio`가 권위**
- 채택: **`optimize_portfolio` objective에 inline soft penalty**(plan Phase 3). `_build_mvo_constraints` **시그니처 변경(4-tuple)·projection objective 편집은 폐기**(spec §4-P2의 4-tuple 서술은 superseded).
- 근거: 기존 risk/turnover penalty가 이미 objective에만 있고 projection엔 없다 — 동일 패턴 유지가 더 작고 일관적. disabled 시 penalty term=`0`(int) → objective 바이트 동일.
- helper: `_beta_vec_cov_implied(cov, bm)` = `(cov@bm)/(bm@cov@bm)`, 가드(`denom>1e-12`, finite, `max|beta|<5`) 미충족 시 zeros(=inert). 헤드라인 진단은 **실현 252d OLS 회귀 beta**(`metrics.realized_beta`), cov-implied 수치가 아님.

### 4.2 Overlay-free 예측의 단일 정식 명칭 (codex #4) — `pre_overlay_ema_predictions`
- **정의**: 오버레이를 모두 OFF로 둔 harvest의 **EMA-blend 후·오버레이 전** 예측 패널.
- **단일 객체로 통일**: alpha attribution 레그 C와 overlay ablation **둘 다** 이 객체를 `precomputed_predictions`로 주입한다. `base.raw_predictions`가 이 의미(pre-overlay·post-EMA)인지 **첫 실행 로그로 검증**하고, 아니면 overlays-OFF harvest를 별도로 수행해 그 `base.predictions`를 쓴다. 어떤 코드도 pre/post-overlay나 pre/post-EMA를 혼동해선 안 된다.
- **이중 오버레이 금지**: post-overlay 예측을 재주입하지 않는다(재사용 경로가 오버레이를 또 적용함).

### 4.3 Factor-neutral 사전 점검 의무 (codex #6)
- `factor_neutral_loadings`의 컬럼명은 **추정값**이다. ablation 전에 **반드시**:
  1. 설정된 모든 컬럼이 실제 feature 패널(`src/features/assembly.py` CORE_FEATURE_WHITELIST)에 **존재**함을 확인,
  2. **applied-date 수 > 0**,
  3. **결측/비유한 impute 비율**을 보고.
- 이 점검 없이 penalty가 inert하면 "TE가 이미 스타일을 중립화함"으로 **오독**된다. 점검 통과 전 P3 결론 금지.

### 4.4 Repo 위생 (codex #8)
- 커밋/패키지에서 **제외**: `.claude/settings.local.json`(로컬 절대경로·권한), `bash.exe.stackdump`(크래시 아티팩트). 디버깅 목적이 명시되지 않으면 포함하지 않는다.

---

## 5. 출처-로컬 의도적 편차 (Pictet PDF vs cc2_rl)

이 프로젝트는 **Pictet 복제가 아니라, 기존 cc2_rl 포트폴리오에 Pictet식 리스크 규율·설명력을 입히는 것**이다. 65종목 집중 성격은 **의도적으로 유지**한다.

| 차원 | Pictet PDF | cc2_rl 의도 | 처리 |
|---|---:|---:|---|
| 보유종목 | 400–500 | 100 (2026-07-16 확장) | 의도적 비복제로 명시 |
| Beta | 1.0 | S0 먼저 측정 후 결정 | P2를 코딩 전 게이트(§3) |
| Tracking error | ≤2% | ex-ante 캡 3.5%(production)·실현 ~4.4%, 가드 4.5% | 로컬 리스크 예산으로 명확화 |
| Active share | ~50% | 문서상 ~4.75% | 단위/정의 확인 |
| 종목 active weight | ±1% | 로컬 캡 상이 | 적응된 제약으로 표기 |
| 국가/산업 노출 | ±2% | sector deviation만 | 누락/비범위로 표기 |
| 팩터 노출 | 인덱스 유사 | soft factor penalty 제안 | loading 검증 후 ablation(§4.3) |
| ESG / Article 8 | PDF에 존재 | 데이터 없어 비범위 | 명시적 제외 유지 |

---

## 6. 결정 로그 — 첫 측정 전에 생성

`2026-06-18-pictet-portfolio-logic-adoption-decision-log.md`를 **S0 실행 전에** 만든다. 최소 섹션:

```
## S0 (ECOS baseline)        — 실행일/커밋, IR/TE/turnover/realized_beta, P2 게이트 판정
## S1 attribution parity     — OFF 바이트동일 여부, ON 시 점유율 합≈1.0
## S2 leg-C construction      — annualized active 델타, round-trip identity
## S3 overlay ablation        — 2^3 + all-off + on-baseline, per-overlay OOS ΔIR, do-no-harm 판정
## S4 beta sweep              — penalty {0,1,5,10,25}, realized_beta 이동, fallback율
## S5 factor                  — 컬럼 존재/applied-date/impute율, 노출 바인딩 여부
## DSR / selection-bias       — experiment_inventory + run_selection_bias 해킷
## Production flips           — 후보별 활성화 결정·근거·롤백 확인
```

각 변경의 채택/보류는 이 로그의 게이트로만 정당화한다.

---

## 7. 실행 루프 (plan Phase 0→5 요약 — 상세는 plan)

각 단계: **실패 테스트 → 실패 확인 → 최소 구현 → 통과 → (승인 시) 커밋**. 단계 사이 게이트를 결정 로그에 기록.

| Phase | 내용 | 핵심 게이트 / 검증 |
|---|---|---|
| **0** | `compute_beta` 추가 → `compute_metrics`에 `realized_beta` 부착 → **S0 ECOS 재인증** | §3 베타 게이트. S0가 단일 기준 |
| **1 (P0)** | attribution config → `compute_alpha_attribution` → harness+CLI attach(cache-safe) → leg C 재-MVO(`pre_overlay_ema_predictions`) | OFF 시 `alpha_attribution` 키 없음·바이트동일. interaction=**upper bound**, 레그 C=**델타**(A/B와 합산 금지) |
| **2 (P1)** | `run_overlay_ablation.py`(harvest-once, EMA·이중오버레이 confound 제거, 2³+all-off+on-baseline) | on-baseline이 S0 재현 후에만 신뢰. **OOS do-no-harm**일 때만 제거 |
| **3 (P2)** | config → **inline** `optimize_portfolio` soft penalty(§4.1) → 스윕 {0,1,5,10,25} | realized_beta가 1.0 쪽 이동 + 단일 사전약정 penalty. 안 움직이면 shelve |
| **4 (P3)** | config(growth+momentum 제외) → penalty + per-date loadings → 단일 penalty | **§4.3 사전점검 필수**. 노출 바인딩 시에만 결론, 아니면 shelve |
| **5** | 바 통과 후보만 `variants/iter15_65tkr_reb21_vtg.yaml` overrides flip(한 번에 1개), S0 재검증, 한 줄 롤백 | §8 production rule |

검증 명령(공통): `<PY> -m pytest tests/<file> -v` (단위), `<PY> run_variant.py --variant …` (백테스트), `<PY> scripts/run_*_ablation.py` (ablation).

---

## 8. 프로덕션 활성화 규칙 (Production rule)

`PipelineConfig` 기본은 **항상 default-OFF 유지**. 활성화는 **production variant override로만**, 결정 로그가 해당 게이트를 기록한 뒤에:

- **attribution**: parity 통과 + 비용 수용 가능 → (가중치 불변이므로) 즉시 가능.
- **overlay 변경**: OOS marginal harm이 **명확·부호 일관**일 때만.
- **beta-neutral**: realized_beta가 1.0 쪽으로 이동하고 fallback/리스크예산 손상 없을 때만.
- **factor-neutral**: penalized active 노출이 **실제로 바인딩되어 하락**할 때만.

한 번에 후보 1개씩 flip → S0 대비 동일 솔버 재검증 → 롤백(플래그 한 줄 revert = 바이트동일 복원) 확인. 후보별 독립 커밋.

---

## 9. 현실이 문서와 어긋나면

코드/데이터가 spec·plan·이 파일의 가정과 다르면(예: `base.raw_predictions`가 pre-overlay가 아님, factor 컬럼 부재, S0 IR이 1.30과 크게 다름) — **그대로 진행하지 말고** 차이를 결정 로그에 기록하고 사용자에게 보고한 뒤 가정을 수정한다. 추정으로 메우지 않는다.
