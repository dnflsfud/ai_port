# ai_port 구조 감사 (§S16 리뷰 트랙) — 2026-08-31

> 멀티에이전트 리뷰(모듈 10군 + 횡단 5렌즈) → 원시 61건 → 중복 제거 51건 → 발견마다 반박 우선
> 검증자 2~3인(코드 추적 · production 영향 정량 · 경험적 재현) → **확정 47건 / 반박 1 / 기지 중복 3**.
> 총 138 에이전트 · 897 도구 호출. §S15(2026-08-27) 확정 34건은 검증 단계에서 중복 제외.
>
> **기준선**: S0′ IR 1.5356 / TE 3.73% / β 1.052 / 퇴화 14/33 / avg_ic 0.019378 (빈티지 08-25 20:47:55 · 16:37:36)
>
> 아티팩트 판본: https://claude.ai/code/artifact/63c20f22-4b30-4448-b2b7-e0f3ec1ab421

**표기 규약**
- **직접 재현** = 메인 모델이 `outputs/s15_2_optvol_scale_fix/backtest_result.pkl` 및 pytest 실행으로 독립 관측한 수치.
- **검증됨** = 발견자와 독립된 반박 우선 검증자 2~3인이 확인.
- 모든 수정 제안은 §2.1(default-OFF + parity)과 §8(한 번에 1개 flip)을 전제로 한다.
- 이 문서는 결정 로그가 아니다. 어떤 항목도 §8 절차(사전등록 → 측정 → 사용자 결정) 없이 production에 적용되지 않는다.

---

## 계기판 (전부 직접 재현)

| 지표 | 값 | 의미 |
|---|---:|---|
| 라이브 모델 나이 | **354일** | 2025-08-28 적합본을 4회 연속 재사용 중 |
| HEAD 실패 테스트 | **5건** | S15.2 flip 이후 acceptance 5 failed / 179 passed |
| 의도한 정보를 담지 못하는 코어 피처 | **8 / 65** | 단위 4 + 이중 z-score 중복 4 |
| MVO 리턴항 ÷ 리스크항 | **약 18만배** | `risk_aversion=1.0`이 사실상 무효 |
| 실제 캘린더 밀도 | **260.9 행/년** | 연율화 상수는 252 |
| production 게이트 | **HOLD** | 종목 44.0% > 35% · Technology 94.4% > 85% |

---

## 한눈에

1. **지금 당장 손봐야 하는 것은 알파가 아니라 운영이다.** production이 오늘 쓰는 모델은 1년 전 적합본이고(4회 연속 재훈련 퇴화), 레포는 S15.2 flip 이후 테스트가 빨간 상태이며, 대시보드·리스크 번들은 은퇴한 pre-flip 런(IR 1.3811)을 읽고 있다. 셋 다 수치를 바꾸지 않고 고칠 수 있다.
2. **코어 65피처 중 8개가 의도한 정보를 담고 있지 않다.** 통화·단위 불일치 4개와 이중 z-score로 기존 피처의 부호반전 복제본이 된 매크로 교차항 4개. §S13.12가 규명한 전달 상한과 별개로, 이 8개는 *애초에 IC를 만들 재료가 아닌 상태*다.
3. **리스크 계층에 구조적 여유가 있다.** MVO 목적함수의 리스크항은 리턴항보다 18만배 작아 사실상 무효이고, 공분산은 N=250 > T=126인데 수축이 걸리지 않는 경로로만 돈다. §S13.41(옵션 IV→Σ 대각)이 이 프로젝트에서 성공한 유일한 리스크 채널이었던 점을 감안하면 남은 개선 여지가 가장 크다.

---

## P — production 수치에 들어가는 결함 (8건)

전부 §S15 확정 34건과 중복되지 않는 신규 발견이다.

### P1. 런던 상장 8종의 `tg_upside`가 12년 내내 −3.87σ 상수 — **직접 재현**
`src/features/sellside.py:341` · `src/data_loader.py:1344`

`upside = tg / local_prices − 1`은 목표주가와 주가가 같은 단위라는 전제인데, Bloomberg PX_LAST는 런던 상장분을 **펜스(GBp)** 로 호가하고 FactSet 목표주가는 파운드로 들어온다. 비율이 ≈0.01로 고정되어 upside ≈ −0.99, 횡단면 z ≈ −3.87이 된다. 로더 어디에도 거래소별 가격 단위 정규화가 없고, 레포 전체에 `GBp`/`pence` 언급이 0건이다.

```
직접 측정 (backtest_result.pkl, 티커별 tg_upside 중앙값 최저 8개 = LN 8종 전부)
  RR/ −3.89 · LSEG −3.89 · REL −3.88 · AZN −3.88
  ULVR −3.88 · HSBA −3.88 · SHEL −3.87 · RIO −3.86
  9위 TSLA −0.69 — 단절 폭 3.2σ
  대조군 AAPL −0.25 · NESN +0.22 · 7203 +0.37 · 000660 +0.46
```

랭커는 이 상수를 알파가 아니라 **국가 더미**로 학습한다(모델별 SHAP 부호가 임의로 뒤집히는 것이 증거). `tg_upside`는 제거 시 IR −0.48을 낸 기록이 있는 핵심 피처인데 8종에서는 정보량이 0이다. 8종의 벤치마크 비중 합은 약 1.9%라 영향은 국지적이지만, 매일 횡단면 std를 부풀려 나머지 242종의 z를 압축하는 부수효과가 있다. 휴면 `fwd_opcf_yield`도 같은 계약을 쓴다.

**처방** — 로더에 거래소별 `PRICE_UNIT_SCALE`(LN: 0.01)을 도입해 `local_prices`와 목표주가의 단위를 맞춘다. 방어로 티커별 median(TG/PX)이 [0.2, 5] 밖이면 fail-fast. 워크북의 목표주가 단위를 1회 감사해 스케일을 확정해야 한다.

### P2. FCF·CAPEX 레벨 z-score가 사실상 2종목 통화 더미 — **직접 재현**
`src/features/accounting.py:50`

`best_calculated_fcf_level_z`·`best_capex_level_z`는 BEST_CALCULATED_FCF·BEST_CAPEX의 **현지통화 절대금액**을 250종 횡단면에서 그대로 표준화한다. 워크북 생성기는 CUR_MKT_CAP만 USD로 변환하고, 로더 `_apply_usd_conversion`도 PX_LAST·Daily_Returns·raw_returns만 변환한다. 원화(×1,350)·엔화(×150) 종목이 평균과 표준편차를 완전히 지배한다.

```
직접 측정 (최종일 횡단면)
  상위: 000660 +5.00(클립) · 005930 +5.00(클립) · 285A +0.14 · 7203 +0.03
  하위: ORCL −0.10 · AMZN −0.10 · META −0.10
  KR·JP 제외 234종의 표준편차 = 0.001   ← 잔여 정보량 사실상 0
```

두 피처는 core whitelist의 'Quality' 블록에 있어 33개 모델 전부가 소비한다. 모델이 보는 것은 현금흐름의 질이 아니라 "이 종목이 한국 종목인가"다.

**처방** — 단위 자유(unit-free) 정의로 교체: FCF는 `FCF / CUR_MKT_CAP`(FCF yield), CAPEX는 이미 존재하는 `capex_intensity_z`(총액/총액이라 단위 일관)로 대체. 단순 USD 환산만 하면 여전히 size proxy이므로 비율화가 의도('quality')에 부합한다.

### P3. `cash_conversion_z` = 총액 ÷ 주당 — 발행주식수 프록시 — **직접 재현**
`src/features/accounting.py:116`

`z(FCF_총액 / |EPS_주당|)`는 분자가 총액, 분모가 주당 값이라 결과가 `주식수 × (FCF/순이익)`이 된다. 통화는 상쇄되지만 주식수는 상쇄되지 않아 횡단면 순서를 주식수가 지배한다.

```
직접 측정 (최종일 상위)
  9432 NTT +5.00 (약 900억 주) · NVDA +4.78 (240억) · AAPL +3.29 (150억)
  7203 +2.23 (130억) · RR/ +1.98 (85억) · IBKR +1.50
  실제 FCF/순이익은 NVDA ≈0.8 · NTT ≈1.0 — +4.8σ가 될 수 없는 값
```

EWMA 프루닝에서 한 번도 탈락하지 않고 33/33 모델에 존재하며 gain share 평균 2.06%(65피처 중 14위).

**처방** — 분자·분모 단위 일치: `FCF / BEST_NET_INCOME`(총액/총액). 필요 시트가 워크북에 없으면 §9 원칙대로 추정으로 메우지 말고 시트를 신설한 뒤 진행.

### P4. 매크로 교차항 5개 중 4개가 기존 피처의 부호반전 복제본 — **직접 재현**
`src/features/assembly.py:787` · `src/features/macro_cross.py`

`macro_cross.py`는 이미 횡단면 z-score된 티커 레그에 **날짜별 스칼라**(금리 z, 커브 기울기, VIX z, DXY z)를 곱해 `mc_*`를 만든다. 그런데 `assembly.build_all_features`의 `skip_zscore`는 Conditioning·Factor·Regime 그룹만 제외하므로 MacroCross 그룹은 **다시** 횡단면 z-score를 통과한다. 날짜별 상수 `c`에 대해 `z(c·x) = sign(c)·z(x)`이므로 매크로의 *크기*는 전부 소거되고 *부호*만 남는다.

```
직접 측정 (3개 날짜 샘플)
  |corr(mc_rate_x_eps_rev, eps_rev)| = 1.0000  (2015-12-25 / 2019-10-25 / 2023-08-25 전부)
  |corr(mc_vix_x_mom252, momentum_252d)| = 1.0000  (부호는 날짜마다 −1 / −1 / +1)
  대조: corr(mc_vol_x_mom63, momentum_252d) = 0.22 ~ 0.40  (두 레그 모두 횡단면 변수 — 정상)
```

모듈 docstring이 약속한 "VIX 3σ와 0.1σ를 구분하는 매크로 크기 × 종목 신호"는 모델에 전혀 전달되지 않는다. 65피처 중 4개가 `eps_rev`·`momentum_252d`의 부호반전 사본이며, `colsample_bytree=0.8` 하에서 분할 선택과 EWMA importance 배분을 왜곡한다.

**처방** — `skip_zscore |= set(feature_groups["MacroCross"])` 한 줄. 레그가 이미 z-score이고 매크로는 63일 시계열 z라 스케일이 호환된다. ±5 클립 영향은 함께 점검.

### P5. fix-pack #5 리비전 클리너가 진짜 붕괴를 무기한 동결 — **검증됨 · 3렌즈**
`src/features/sellside.py:116-136`

§S15 fix-pack #5는 "롤오버 아티팩트가 1일만 지연되는" 결함을 고치려고 지속 연장 마스크를 넣었다. 그런데 연장 해제 조건이 `|row−ref| ≤ threshold` 또는 `|row| ≥ |ref|·0.5` 뿐이고 **길이 상한도, 아티팩트 판별 조건도 없다**. 실적발표 직후 컨센서스가 진짜로 붕괴해 중간 대역에 머무르면 그 기간 내내 붕괴 이전 값이 ffill로 복원된다.

```
합성 재현 (production 파라미터: threshold 15 · extreme 50 · ratio 0.5)
  rev 80 → 10 후 200일 지속: OFF는 t+1부터 10.0, ON은 200일 전부 80.0으로 동결
  80 → 5 후 39까지 실제 회복: 39 < 40이라 끝까지 해제 없음
  회복 시 해제 스텝이 threshold(15)의 1.7~3.3배로 한 번에 유입 — 재정제되지 않음
```

소비처가 넓다: 코어 피처 `eps_rev`·`eps_rev_ma_63d`·`eps_rev_trend`·`sales_rev_ma_63d`, 매크로 교차 3종, 그리고 **PEAD 오버레이의 rev_quality와 growth tilt**. 결과적으로 "리비전이 방금 무너진 종목"에 PEAD 부스트가 붙는다 — 오버레이 의도와 정반대다. 기존 테스트는 30행·회복값 60 케이스만 검사해 이 경로를 잡지 못한다.

**처방** — 연장 최대 길이(예 10BD) 상한, 또는 "같은 날 base_mask 발화 종목 비율 ≥20%"(롤오버는 횡단면 동시 발생) 조건, 또는 earnings_timeline ±5BD 창 제외 중 하나. fix-pack 8건에서 #5만 분리해 페어드 재측정하고, 연장 셀 수를 `[RevisionClean]` 로그에 노출.

### P6. Ledoit-Wolf 수축 분기가 production에서 도달 불가 — **검증됨 · 2렌즈**
`src/portfolio_optimizer.py:88`

`estimate_covariance`는 lookback 창에 NaN이 하나라도 있으면 무조건 `_pairwise_covariance`(수축 없음)로 가고, NaN이 전혀 없을 때만 Ledoit-Wolf를 쓴다. production의 리스크 원천은 `raw_returns`(상장 마스킹·임퓨트 없음)이고 2019~2025년에 24종이 순차 상장했으므로 **97회 리밸 전부에서 126×250 창에 NaN이 존재**한다. 즉 LW 분기는 죽은 코드다.

결과적으로 250×250 표본 공분산을 126개 관측으로 추정하므로 rank ≤ 125이고, 나머지 125개 고유값은 `fallback_var × 1e-4`(연 변동성 약 0.3%)로 바닥 처리된다. MVO는 이 Σ로 TE 제약을 평가하므로 **바닥 처리된 부분공간 방향의 액티브 베팅은 사실상 공짜**가 된다. 실현 TE 3.73%가 ex-ante 캡 3.5%를 넘는 방향과 일치한다.

**처방** — pairwise 경로에도 수축을 넣는다(pairwise-complete 모멘트로 LW 강도 추정 → 상수상관/단일팩터 타깃), 또는 최소한 고유값 바닥을 중위 분산의 5~10%로 올려 "공짜 방향"을 없앤다. docstring("via Ledoit-Wolf shrinkage")이 production 경로를 오도하므로 정정 필요.

### P7. 메가캡 '변동성 수축'이 실은 양방향 평균 회귀 — **검증됨 · 2렌즈**
`src/portfolio_optimizer.py:110`

`cov_megacap_vol_shrink_enabled`(기본 ON)는 `bm_i > 2/n`(=0.8%)인 종목의 변동성을 `(0.5·평균 + 0.5·σ_i)`로 옮긴다. `min(1.0, ·)` 클램프가 없어 **저변동 대형주는 부풀려지고 고변동 대형주는 할인**된다. 250종 cap-weighted에서 이 집합은 21종·벤치마크 비중 51%다.

메가캡 펀딩 모드가 이 종목들에 −10%까지(다른 종목의 2.5배) UW를 허용하므로, TE 캡이 **가장 큰 단일 액티브 포지션을 가장 부정확한 분산으로** 평가한다.

**처방** — 설계를 유지하더라도 리밸마다 수축 전 Σ 기준 ex-ante TE를 diagnostics에 병기해 실현 TE와의 비율을 추적. 플래그 OFF arm 1회 측정으로 리스크 예산 왜곡 정도를 결정 로그에 기록할 가치가 있다.

### P8. `subsample: 0.8`이 조용히 무시됨 (배깅 미적용) — **검증됨 · 2렌즈**
`src/model_trainer.py:442` · `src/config.py:366` · `variants/codex_causal_rank_65.yaml:41`

LightGBM은 `bagging_fraction`이 `bagging_freq > 0`일 때만 동작한다. production은 `subsample: 0.8`만 지정하고 `subsample_freq`를 지정하지 않아 sklearn 래퍼 기본값 0이 적용된다. pkl booster params에서 `subsample_freq: 0` 확인.

현재 수치에 오류를 만들지는 않지만 **설정이 서술하는 모델과 실제 학습된 모델이 다르며**, 누군가 "설정대로" freq를 켜면 S0′ 기준선이 통째로 바뀌는 숨은 자유도다.

**처방** — 수치 불변 선택지: 키 제거 또는 `# inert: subsample_freq=0` 명시. 실제 배깅을 원하면 단일 사전등록 arm으로.

---

## O — 운영·측정 결함 (9건, 산출물 대부분 불변)

### O1. 라이브 모델이 354일 스테일인데 게이트는 PASS — **직접 재현**
`src/model_trainer.py:793-799` · `scripts/validate_portfolio_bundles.py:548`

```
직접 측정 (models 딕셔너리 33개 객체 동일성 추적)
  2025-08-28  trees=132  ← 적합
  2025-11-25  trees=132  ← 재사용
  2026-02-20  trees=132  ← 재사용
  2026-05-20  trees=132  ← 재사용
  2026-08-17  trees=132  ← 재사용   = 라이브 모델, 나이 354일
  stale_depth = 4 · 게이트 한도 7 → PASS
  고유 모델 19/33 · 과거 최장: 2021-01-26 적합본이 2022-01-13까지 5회 사용
```

**게이트가 시간 축을 갖고 있지 않은 것이 핵심이다.** `MAX_CONSECUTIVE_STALE_RETRAINS = 7`은 재훈련 *슬롯* 수이고, retrain_freq=63BD에서 한도 7은 약 1.75년의 모델 나이를 허용한다. 전체 퇴화율(14/33 = 42%)은 이미 report-only로 강등되어 있어 어떤 게이트도 "모델이 1년 묵었다"를 잡지 못한다.

**처방** — 게이트에 **일수 기준** 추가(예: 라이브 모델 나이 > 189BD면 breach). 근본 치료는 Tier 1 T1-1.

### O2. S15.2 flip 이후 acceptance 테스트 5건 실패 중 — **직접 재현**
`tests/acceptance/test_s13_{6,10,13,14,15}_*.py`

```
직접 실행: pytest tests/acceptance -q
  5 failed, 179 passed
  AssertionError: assert not ['option_vol_scale_fix_enabled']
```

§S15 flip 때는 6건을 allowlist로 흡수했는데(650 PASS) S15.2 flip에서는 그 절차가 누락됐다. 이 상태로는 다음 arm의 "OFF parity" 인증이 신뢰를 잃는다.

**처방** — allowlist 흡수 또는 arm variant 갱신. 근본적으로 **flip 절차에 "핀 테스트 갱신"을 체크리스트 항목으로 고정**.

### O3. 운영 export의 공분산이 production 경로와 갈라짐 — **검증됨 · 3렌즈**
`scripts/export_operating_data.py:204-215`

production은 S15.2 flip 이후 옵션 IV 스케일을 **raw_returns + pre-impute 관측 마스크**로 만든다. 그런데 export의 `_load_optvol_scale`은 여전히 `build_option_vol_scale(data.returns[tickers], iv_sheet)` — 임퓨트 dense 패널, 마스크 없음, 플래그 미참조 — 즉 S15.2 이전 레시피다. 두 flip 커밋 모두 이 스크립트를 건드리지 않았다.

이 스케일 패널은 `risk.json`(가드 3종), `monitoring.tracking_error_constraint`(97 리밸 ex-ante TE 감사), `expected_rebalance.json`(what-if 목표비중) **셋 모두의 Σ에 곱해진다**. §S15.2 실측으로 임퓨트 은닉 셀 3.77%, 관측률 50% 미만 20종(285A 0%, 6146 8%, COF 10%…).

**처방** — export를 backtest와 동일 분기로. 더불어 `expected_rebalance`가 리밸일일 때 what-if 가중치와 실제 `portfolio_weights`의 L1 차이를 기록해 1e-8 초과면 에러로 강등하는 **E0 자기검증** 추가.

### O4. 대시보드·리스크 번들이 은퇴한 pre-flip 런을 읽고 있다 — **직접 재현**

```
직접 측정
  operating_codex_causal_rank_65/performance.json → IR 1.3811 (은퇴한 pre-fixpack 값)
  인증 기준선 s15_2_optvol_scale_fix/metrics.json → IR 1.5356
  production_gate: HOLD — name_active_risk_ok false (44.0% > 35%)
                          sector_active_risk_ok false (Technology 94.4% > 85%)
```

Technology 액티브 리스크 점유는 라운드 종료 시점 90.6%에서 94.4%로 *악화*되었다.

**처방** — 번들 export에 **run_dir의 resolved config와 현재 variant yaml의 diff 검사**를 넣어 불일치 시 경고.

### O5. 연율화 상수 252가 실제 캘린더 260.9행/년과 불일치 — **직접 재현**
`src/backtest.py:899` · `src/utils.py:27`

```
직접 측정
  panel 인덱스 2014-01-24 ~ 2026-08-25 · 3,283행 · 12.58년 → 260.9행/년
  portfolio_returns 2,022행 · 7.75년 → 261.1행/년
```

캘린더는 순수 **평일 인덱스**다(주말만 제거, 휴장일은 0-수익률 행으로 잔존). IR·TE는 √(261/252) − 1 ≈ **1.7% 과소**, 연수익·회전율은 3.5% 과소. 모든 arm에 일관 적용되므로 ΔIR 비교는 유효하지만, "TE 3.73% vs 캡 3.5%" 같은 **절대 임계 판정**에서는 편향이 실재한다.

**처방** — 런의 실제 행/년으로 연율화하거나, 최소한 metrics.json에 사용된 연율화 상수와 실측 행 밀도를 병기. 수정 시 모든 역사 수치가 이동하므로 §8 절차 대상.

### O6. DSR 게이트의 생존편향 항목이 구조적으로 영구 FAIL — **검증됨 · 2렌즈**
`run_selection_bias.py:398` · `:158`

생존편향 검사가 PIT 상장 마스크로 NaN이 된 종목을 "늦은 진입"으로 세기 때문에, 마스크가 정상 작동할수록 verdict가 FAIL로 고정된다. 추가로 기대 최대 SR이 Bailey–López de Prado 공식이 아니라 점근 상한 `√(2 ln N)`으로 계산되어 p-value가 체계적으로 과대 추정된다.

**처방** — 생존편향 검사에서 상장 마스크 종목 제외 + 기대 최대 SR을 Bailey–LdP 정식으로 교체. 진단 전용이라 산출물 불변.

### O7~O9. 나머지 운영 결함 — **검증됨**

| ID | 내용 | 결과 |
|---|---|---|
| O7 | `metrics.json`·manifest에 데이터 빈티지 지문(워크북·Index.xlsx의 mtime/size/hash)이 기록되지 않음 | "동일 빈티지 비교"라는 §S13.47 이후의 핵심 규율이 산출물만으로 **검증 불가** |
| O8 | 유니버스 교집합이 "워크북에 존재하는" 필수 시트로만 계산 — 13개 중 10개는 통째로 없어도 `expected_universe_size` 250/250 통과 | ENOSPC 등으로 시트가 빠진 채 재생성되면 63피처로 학습된 결과가 **실패 없이** metrics.json으로 나감 |
| O9 | 진단 스크립트군(alpha attribution, overlay ablation, factor ablation, seed ensemble, dr_alpha)이 `raw_predictions`를 재주입 — 이미 lag 1BD가 적용된 패널이라 **2일 지연** | 이 스크립트들의 IR/TE는 S0와 직접 비교 불가. 라운드트립 게이트 전에 summary.json이 먼저 기록되는 문제도 동반 |

---

## D — 휴면 경로 (12건, 향후 arm 보호용)

| 위치 | 결함 | 발동 조건 |
|---|---|---|
| `data_loader.py:437` | Daily_Returns를 레벨 시트처럼 무제한 ffill — 수익률 시계열이 끝난 종목의 마지막 일간 수익률이 매일 반복 | 상폐·흡수 종목이 유니버스에 남는 순간 유령 복리 수익(현재 고정 250 생존 유니버스라 잠복) |
| `target_engine.py:222` | regime-weighted PCA 경로가 eligibility-aware가 아님 | `regime_pca_weighted_enabled` ON 시 타깃 거의 전량 손실 |
| `target_engine.py:335` | `regime_aware_pca_lookback` 외 4개 필드가 무배선 | 켜도 조용히 baseline — 실험이 no-op인 줄 모르고 판정 |
| `features/peer_earnings.py:98` | 상장 전 종목을 '침묵 피어'로 산입 | peer earnings arm 재도전 시 |
| `features/option_risk.py:66` | 의도적 NaN('마지막 발표 이후')을 로더 ffill이 0(=오늘 발표)으로 변환 | option risk arm 재도전 시 |
| `features/implied_vol.py:55` | IV 스프레드가 로더 임퓨트 셀을 관측값처럼 소비 | IV 서피스 arm 재도전 시 |
| `features/index_eps.py:55` | S15 fix-pack의 팩터 캘린더 수정을 받지 못해 미 휴장·테일 날짜에 literal 0.0 | index EPS·revision arm 재도전 시 |
| `data_loader.py:588` | Universe_Meta 날짜가 정수(20200930)면 epoch-ns로 해석 → 상장일 1970-01-01 | 워크북 편집으로 날짜 서식이 바뀌는 순간 |
| `harness.py:152` | `satellite_budget`·`satellite_max_per_stock`을 YAML로 *완화*하면 조용히 무시 | satellite 제약 arm 설계 시 |
| `residual_sleeve.py:945` | 슬리브 arm의 optimizer_fn이 production 공분산 경로를 재현하지 않음 | 잔차 슬리브 재측정 시 |
| `attribution.py:282` · `:635` | Li 3-성분 분해가 균등 grid 분산과 경험 분산 혼합 · 상장 전 유령 행을 SHAP에 포함 | attribution을 의사결정 근거로 쓸 때 |
| `portfolio_optimizer.py:178` | `score_based_weights`: 예측 하나가 NaN이면 북 전체 동결 | `use_score_based` ON 시(현재 미사용) |

---

## 성과 개선 후보 (44건 → 4계층)

6축 리서치가 44건을 냈고, 그중 5건은 리서처 자신의 사전점검에서 이미 탈락했다(추정치 분산 잔차 IC t<2, 공매도비율 t<2, 통화블록 demean 라벨, 호라이즌 분리 이중 랭커, 섹터 상대화 peer_rel). 남은 것을 실제 채택 이력 — 정확성 수정 +0.166, ES 지표 +0.203, Σ 대각 +0.124, 피처 블록 +0.266 vs 알파 피처 arm 10연속 FAIL — 을 사전확률로 삼아 정렬했다.

### Tier 0 — 정확성 수정 팩 (최우선 · 인벤토리 비계수)

P1~P4를 단일 default-OFF 플래그 뒤에 묶는다. §S13.33·§S15 선례대로 **채택 기준은 정확성**이며 ΔIR은 관측 기록이다(선택 이벤트가 아니므로 DSR 인벤토리 비계상). 알파 경로를 수정하므로 `avg_ic`가 바뀐다 — §S15의 "avg_ic 비트 불변" 구조 불변식 대신 do-no-harm으로 판정.

코어 65피처 중 7개가 전 종목에서 의도한 정보를 담지 못하고(P2·P3·P4), 1개가 8종목에서 상수 더미(P1)인 상태를 해소한다. 피처 *추가*가 아니라 이미 있는 피처를 의도대로 되돌리는 것이라 §S13.12의 전달 상한(새 정보 가중치 전달률 ~9%)이 적용되지 않는 드문 개입이다.

```
게이트
  사전등록: 측정 전 단독 커밋 · 채택 기준 = 정확성(ΔIR 부호 무관)
  E0: OFF 경로 바이트 동일 parity 단위테스트 선행
  E1: 페어드 동일 빈티지 · 관측 = ΔIR / ΔTE / Δbeta / Δ퇴화율 / Δturnover
  E2: do-no-harm — TE ≤ 4.5% · active share 캐릭터 보존 · ECOS fallback 0
  단일 파라미터: 없음(플래그 1개, 하위 4수정 동시)
```

### Tier 1 — 구조적 치료

| ID | 후보 | 기전 | 단일 파라미터 |
|---|---|---|---|
| T1-1 | **퇴화 폴백을 '이전 모델 재사용'에서 '고정 용량 신선 재적합'으로** | O1의 근본 치료. 조기종료가 10그루 미만이면 재사용 대신 사전등록 고정 용량(고유 19모델 best_iteration 중앙값 67)으로 조기종료 없이 재적합. §S10.2·§S11.5·§S11.10·§S13.8·§S13.47은 전부 "퇴화율을 낮추는" 하이퍼파라미터 변경이었고 폴백 *정책* 자체를 바꾼 arm은 0건 | `degenerate_fallback_mode: fresh_fixed` (용량 67 고정) |
| T1-2 | **pairwise 공분산에 수축 복원** | P6의 치료. 성공한 유일한 리스크 채널(§S13.41 +0.124)의 핵심은 **알파 비트 불변 증명 가능**이었고 Σ 수정도 같은 성질 → 채택 바가 낮다. 실현 TE 3.73% > 캡 3.5% 이상을 직접 겨냥 | 수축 타깃 1종 또는 고유값 바닥 비율 1값 |
| T1-3 | **risk_aversion 단위 정합 (λ* ≈ 26,000)** | 리턴항 ≈ 0.88 vs 리스크항 상한 4.86e-06 → 약 18만배. 현 MVO는 "제약 하 μ·w 최대화"이고 리스크 통제는 전부 하드 TE 캡. §S13.50의 "변위 80%가 제약 강제"의 구조적 원인 | `risk_aversion: 26000` (사전약정 1값) |
| T1-4 | **섹터 액티브 분산 하드 SOC 제약 (ρ=0.85)** | Technology 94.4% HOLD. 현 통제는 ±10% *가중치* 밴드뿐이라 리스크 점유를 제어 못함. SOC라 MVO·투영 양쪽 동일 적용, §S11.5 λ 스케일 불일치 원천 제거. **IR 후보가 아니라 거버넌스 레버** | ρ = 0.85 (기존 가드값) |

T1-1~T1-3 순서 근거: T1-1은 실측된 라이브 리스크의 치료라 우선순위 최상, T1-2는 알파 비트 불변 증명이 가능해 채택 바가 낮으며, T1-3은 T1-2로 Σ가 신뢰할 만해진 *다음*이라야 의미가 있다(부정확한 Σ에 큰 λ를 곱하면 오차 증폭).

### Tier 2 — 라벨·학습·Σ (전례상 성공 채널)

| ID | 후보 | 기전 요약 | 단일 파라미터 |
|---|---|---|---|
| T2-1 | 라벨 지평 20 → 63BD | 액티브 포지션 평균 보유 = active share 0.199 ÷ one-way turnover 0.349 ≈ **144BD**로 라벨 지평의 7배. §S11.8 실측 IC h20 0.053 → h63 0.080. §S11.9 블렌드 실패는 라벨을 *섞어서*였고 순수 63d 단일 라벨은 잔존 후보로 명시됨 | `forward_horizon: 63` |
| T2-2 | 조기종료 지표를 선형 top-k 평균으로 | §S13.45-D: 북 OW의 69.3%가 top-5 밖, 60.4%가 랭크 6–50. ndcg@20은 지수 이득·위치 할인으로 최상위 과집중. §S13.47이 k 하나로 +0.203을 낸 계보의 다음 단계 | ES 지표 함수 1종 (다중성 3회째 선언) |
| T2-3 | 학습 쿼리 5BD 스트라이드 | 인접일 라벨 순위 lag1 Spearman 0.93 — 1,260일 창의 독립 정보는 ~63창인데 행 수는 25만. `min_child_samples=60`이 실제로는 독립 관측 3개를 의미 | `train_query_stride: 5` |
| T2-4 | 시간 감쇠 표본 가중 | 5년 창 균등 가중 → 2019년이 2025년과 동일 무게. §S13.47(이득 2024+ 집중)·§S13.23(후반 IC 감쇠)가 관계 표류를 시사 | `sample_weight_halflife: 504` |
| T2-5 | HAR 단기 실현변동성 항 → Σ 대각 | 모델 B의 실현변동성 정보가 126BD 창 하나뿐이라 급등 초기 지연. `log σ_trail21` 추가(Corsi HAR-RV 단기 성분) | 회귀변수 1개 |
| T2-6 | 어닝 이벤트 분산 항 → Σ 대각 | 21BD 창에 발표일이 들어가면 창 σ +33%. 어닝 캘린더는 랭커 경로에서 3번 실패(§S13.9·10·38)했지만 **Σ 대각 arm은 0건**이고 랭킹 불변이라 turnover 증가 없음 | 이벤트 더미 1개 |
| T2-7 | UW측 μ 평탄화 | score gate로 이미 OW 불가인 μ≤0 종목의 objective μ만 0으로 → "가장 낮은 μ"가 아니라 "가장 좋은 헤지"로 선택 | 플래그 1개 |
| T2-8 | 종목·일자별 거래비용 | 균일 10bp → `spread/2 + k·σ·√(참여율)`. 측정 IR은 하락만 가능 → **성과 후보가 아니라 capacity 게이트**. §S13.12의 "비용 7bp 무시 가능" 전제 자체를 검증 | k 1값 |

### Tier 3 — 새 데이터가 필요한 축

| 후보 | 근거 | 커버리지 위험 |
|---|---|---|
| **EPS 기간구조 slope** (BEST_EPS 1FY/2FY) | 유일하게 채택된 피처 블록(§S13.25, gain 6.09% · ΔIR +0.266)의 자매축. 현 워크북은 BEST_SALES만 1FY/2FY가 있고 EPS는 블렌디드 단일 시트라 **존재하지 않는 축** | 낮음 — 기존 시트 확장 |
| **순자사주매입** (shares_out 변화) | `CUR_MKT_CAP / PX_LAST`로 **지금 유도 가능**. Pontiff–Woodgate 순발행 효과. 연 단위 신호라 turnover 불변. 사전점검 잔차 IC t 2.84 (§S13.36의 t 1.04와 대조) | 없음 — 즉시 사전점검 가능 |
| **뉴스 주목도 조건부 PEAD** | 드리프트는 주목이 낮을 때 강함(Hirshleifer–Lim–Teoh). §S8 실패는 *톤*·*피처 주입*이었고 이건 *발행량*·*오버레이 조건자* | 중간 |
| **CDS / DRSK → Σ 대각** | 신용시장이 주식 변동성 선행(Acharya–Johnson). §S13.41 계보의 새 입력 | 높음 |
| **실적 콜 트랜스크립트 톤** | PEAD `rev_quality`는 발표 *전* 정보만 씀 — 발표 자체의 정성 정보가 0 | 높음 |

---

## 제안하는 실행 순서

1. **레포를 초록으로 되돌린다 — O2.** allowlist 흡수 또는 arm variant 갱신 + flip 체크리스트에 "핀 테스트 갱신" 고정.
2. **운영 계층을 production과 다시 맞춘다 — O3·O4·O7·O8.** 전부 산출물 불변이라 사전등록 없이 즉시 적용 가능. expected_rebalance E0 자기검증 동반.
3. **Tier 0 정확성 팩 사전등록 + 페어드 측정.** 다음 모든 arm의 새 비교 기준선이므로 다른 어떤 후보보다 먼저 확정.
4. **P5를 fix-pack에서 분리해 재설계.** 정확성 트랙이지만 새 파라미터가 생기므로 별도 사전등록. Tier 0 측정 후 새 기준선 위에서 단독.
5. **Tier 1을 하나씩 — T1-1 → T1-2 → T1-3.**
6. **T1-4(섹터 SOC)는 위반 만성화 실측 후 결정.** §S11.5 사전등록 발동 조건(최근 24회 위반율 ≥50%) 실측 선행. IR 후보가 아니라 거버넌스 결정.
7. **Tier 2는 새 기준선 확정 후.** T2-1·T2-2가 기대값 최대이나 §S13.8의 ES 지표 다중성 규율 명시 필요.
