# 포트폴리오 구조 리뷰 루프 — 결과 (2026-06-24)

대상: `c2/ai_port/src` 전체 (~8,900 LOC). 5개 read-only 리뷰어 병렬 + 통합 트리아지.
기준선: **39 tests pass** (수정 전·후 동일). 모든 수정은 baseline 바이트동일 유지가 증명/검증된 것만 적용.

판정 규칙: critical/medium이라도 **수정 시 certified baseline(패널·예측·메트릭)이 바뀌면** invariant #1/#2 위반 →
사용자 루프 규칙("invariant 위반해야만 고쳐지는 이슈는 보고만") 및 CLAUDE.md §6/§8에 따라 **report-only**.

---

## A. 적용한 수정 (baseline-preserving, 검증 완료) — 4건

| # | 위치 | 등급 | 내용 | 왜 안전한가 |
|---|---|---|---|---|
| 1 | `portfolio_optimizer.py:236` | CRITICAL | 광범위 `except ValueError`를 `"NaN or Inf"` 메시지로 한정, 그 외 re-raise | baseline에선 해당 예외 없음 → 바이트동일. 숨겨진 구성 버그가 bm-fallback으로 은폐되던 silent-fallback 제거 |
| 2 | `attribution.py:148` | MEDIUM | all-NaN feature 컬럼에서 `np.percentile` 예외 가드 | OFF-default 모듈. 정상 컬럼엔 finite 값 존재 → 미발동. 예외→`(0.33,0.67)` 왜곡 경로 차단 |
| 3 | `attribution.py:661` | MEDIUM | 실패 시 하드코딩 `(0.33,0.67)` → `(np.nan,np.nan)` | `harness.py:88`의 `l==l and nlr==nlr` NaN-skip이 해당 날짜를 평균에서 정확히 제외 → 헤드라인 share 왜곡 제거 |
| 4 | `features/conditioning.py:162` | CRITICAL(증명상 inert) | `days_to.bfill()` 제거 | `searchsorted(side='left')` NaN은 마지막 실적일 이후 **후행 구간에만** 발생 → bfill이 끌어올 값이 없어 **no-op**. 제거는 바이트동일 + 데이터 형태 변화 시 미래 끌어옴 방지 |

검증: `pytest tests/ -q` → **39 passed**.

---

## B. Report-only — 수정 시 baseline 변경 (invariant #1/#2) → 사용자 결정 필요

> 이 항목들은 default 동작/패널/예측을 바꾼다. CLAUDE.md §8 "한 번에 1개 flip → S0 재검증 → 롤백 확인" 절차와
> 결정 로그 기록이 선행돼야 한다. 임의 수정 금지.

| 위치 | 등급 | 이슈 | 권고 |
|---|---|---|---|
| `config.py:79` `macro_cross_enabled=True` | CRITICAL | 2026-04-22 신규 5피처가 ON-default. 순수 `PipelineConfig()`가 pre-feature baseline과 다름 | baseline_v2에 포함된 의도라면 결정 로그에 명시. 후보 arm이면 `False`로. **확인 필요** |
| `config.py:30` `data_path` 절대경로 | CRITICAL | 개발자 홈 절대경로 하드코딩(이식성·재현성). 데이터는 ai_port 트리 밖 `re_study/`에 위치 | 경로 변경 시 실제 실행이 깨짐. env-override + repo-relative fallback 스킴을 **합의 후** 도입 |
| `model_trainer.py:192` lgbm | MEDIUM | `deterministic=True`/고정 thread 미설정 → 멀티스레드 FP drift가 1e-6 parity 바를 흔들 수 있음 | `deterministic:True` 추가 시 예측 수치가 이동 → S0 재인증 동반 필요. 결정 로그에 기록 후 진행 |
| `backtest.py:1268-1280` IC 이중정의 | CRITICAL | `targets`에 날짜 있으면 target, 없으면 raw 20d fwd-sum으로 **조용히 대체** → 게이트 메트릭 `avg_ic` 왜곡 | `targets` 미커버 시 `realized=None`(IC skip)로 통일. **단** `avg_ic` 수치가 바뀔 수 있어 baseline 영향 확인 후 적용 |
| `features/short_interest.py:55` | MEDIUM | 무제한 `.ffill()` → SI가 무한정 stale, `si_tsz_chg_63d`가 stale 구간에서 ~0 | `limit≈21` 추가. baseline 패널 변경 → 결정 로그 |
| `features/factor.py:35-36 등` | MEDIUM | broadcast factor가 ffill 없이 intersection만 → 캘린더 불일치일이 NaN→0 | `regime.py`처럼 `.reindex(dates).ffill()`. baseline 패널 변경 |
| `features/price.py:55` `drawdown_63d` | MEDIUM | `min_periods=1`로 day-1 drawdown 구조적 0 (타 롤링은 `min_periods=w`) | `min_periods=63`. baseline 패널 변경 |
| `features/conditioning.py:106`, `factor.py` warm-up | MEDIUM | 롤링 median NaN 구간에서 `>NaN`→False로 플래그가 워밍업 내내 silent 0 | 워밍업을 NaN 마스킹. baseline 패널 변경 |
| `features/assembly.py:663-665` 최종 `.fillna(0.0)` | MEDIUM | broadcast(raw) 피처에서 0이 "평탄커브/제로금리"의 의미값 → 가짜 레짐 주입 | skip_zscore 피처는 source에서 ffill. baseline 패널 변경 |

---

## C. Report-only — 그 외 (의도 의존 / 미호출 코드 / §2 충돌 / 동작특성)

| 위치 | 등급 | 이슈 | 처리 |
|---|---|---|---|
| `target_engine.py:377` `add(piece, fill_value=np.nan)` | MEDIUM | `fill_value=np.nan`은 no-op → 멀티호라이즌 블렌드가 교집합으로 붕괴 | multi-horizon은 OFF-default(미실행). **올바른 의도(0.0 vs 교집합)가 불명확 → 추측 금지**, 스펙 확인 후 수정 |
| `attribution.py:485-498` Ridge 미정규화 | MEDIUM | macro 스케일 차이로 importance가 스케일에 지배 | `explain_period`는 **외부 호출자·테스트 없음**(미배선) → 지금 수정은 surgical 근거 없음. 배선 시 standardize |
| `attribution.py:276-291` interaction 정규화 | MEDIUM | 비공통 subsample로 `total_var-total_marginal` 계산 → upper-bound가 0으로 clamp/은폐 가능 | OFF-default 진단. 동일 subsample 재사용으로 수정(출력 변동) → 진단 정확도 개선, 배선/필요 시 |
| `analytics.py:87-91` regime 단일관측 | MEDIUM | idx<2에서 std→NaN→"Low Volatility" 오분류 | 진단 출력(첫 rebal일). min-window 가드 시 라벨 변동 → 필요 시 |
| `portfolio_optimizer.py:455/529` `np.all(isfinite([]))==True` | MEDIUM | 빈 universe(n==0)가 finite 통과 | **프로덕션 미발생 시나리오** → §2(불가능 시나리오 핸들링 금지)와 충돌, 미적용 권고 |
| `analytics.py:475-478` 빈 dict `max()` | MEDIUM | 빈 sector/style dict에서 `max()` ValueError + 동점 비결정성 | 동상. 빈 dict는 프로덕션 미발생 → §2 충돌 |
| `harness.py:122-124` `inject_config` 전역 변이 | MEDIUM | 모듈 전역 `DEFAULT_CONFIG` 재바인딩 → 병렬 variant 시 교차오염 위험 | 현재 직렬 실행이라 무해. 향후 병렬 스윕 도입 시 config threading/try-finally 복원 |
| `backtest.py:1155` fallback 탐지 비일관 | MEDIUM | diagnostics 유무에 따라 fallback 판정 상이 → `optimizer_failure_rate` 인터페이스 의존 | diagnostics `used_fallback`로 표준화(동작특성 변경) |
| `backtest.py:906/1086` first-rebal cadence | MEDIUM | rebalance가 첫 성공시점에 앵커 → 데이터 의존 그리드 시프트 | 의도면 문서화, 캘린더 앵커가 의도면 플래그 처리 |
| `model_trainer.py:298-303` 센티넬 모호성 | MEDIUM | 명시 인자가 모듈 default와 같으면 config로 덮임 | 현재 호출자 미발현(latent). `Optional=None` 센티넬 권고 |
| `dr_walkforward.py:104,159` val 0개 | MEDIUM | early-stopping 무력화가 무로그 | dr_alpha OFF-default. 로그 추가 권고 |
| `sellside.py:91-92` `shift(-5)` | CRITICAL(조건부) | timeline이 realized 날짜면 t+5 정보로 t 마스킹=누수 | 기본 `timeline=None`(미발동). as-of 스케줄 계약 assert/문서화 |

### LOW (사용자 규칙: 보고만)
- `config.py:121` `revision_clean_mode="reversion_gated"` vs sellside.py 다수 docstring "down_only" — **코드는 자기일관**, docstring이 stale (문서 정합만 필요)
- `model_trainer.py` 다수 docstring stale (3년/756일 → 실제 1260일; 3-tuple → 4-tuple)
- `attribution.py` 미사용 import/param, ticker bucket 중복(`analytics.get_asset_rotation_buckets` 미재사용)
- `metadata.py`/`assembly.py` `LEVEL_SKIP_SHEETS` 중복 정의, feature 컬럼 순서가 dict 순서 의존(정렬 권고)

---

## 요약

- **수정 4건** (critical 2 + medium 2) 적용, **39 tests 유지**.
- **"critical/medium = 0" 미달성** — 잔여 항목은 전부 (a) baseline 변경 필요(§1/§2 invariant), (b) §2 충돌하는 방어코드,
  (c) 의도/스펙 의존, (d) 미배선 코드라 **auto-fix가 invariant/규칙 위반**. 의도적으로 보고만 함.
- **다음 액션(사용자 결정)**: B 섹션 항목을 결정 로그에 등재 → §8 절차(1개씩 flip → S0 재검증 → 롤백 확인)로 처리.
  특히 `config.py:79`(macro_cross default)와 `backtest.py:1268`(IC 이중정의)은 게이트 메트릭에 직접 영향.
