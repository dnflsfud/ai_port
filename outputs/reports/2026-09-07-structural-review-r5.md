# ai_port 구조 감사 5차 (§S18 리뷰 트랙) + 성과 개선 후보 — 2026-09-07

> **방법**: 메인 단독 수행(에이전트 0). ① 코드 렌즈 — `src/` 핵심 9파일(data_loader·backtest·target_engine·model_trainer·portfolio_optimizer·assembly·sellside·price·accounting·conditioning·factor)을 전량 읽고 룩어헤드·정렬·수치 안정성 관점으로 점검(§S17 비평가가 남긴 미검정 3건을 첫 항목으로). ② 데이터 프로브 3종(읽기 전용, 백테스트 0) — A 원시 워크북(커버리지 갭 임퓨트·스핀오프·정지 수익률), C 백필 상수 접두, B 인증 S0′ pkl(`outputs/s17_5_nominal_price`) 진단. 스크립트·결과는 `outputs/s18_prechecks/`(probe_a_raw.py·probe_b_pkl.py·probe_c_flat.py + JSON/CSV).
> 레포·production·인벤토리(472) **무변경**. 이 문서는 결정 로그가 아니다(정본 = 결정 로그 §S18).

**기준선 주의**: 인증 S0′ IR 1.7633(`outputs/s17_5_nominal_price`, 워크북 2026-09-03 15:51:22 / Index 09-03 14:26:46)은 **APH 목표주가 기저가 깨진 빈티지**에서 측정됐다(아래 P1). 09-04 14:50 재생성 워크북 이후의 production 런(09-07 12:36)은 **IR 1.6106 / TE 3.65% / avg_ic 0.0166 / 퇴화 11/33**(APH 정상) — 1.7633 과 −0.153 포크. 이후 모든 arm 은 새 빈티지 쌍에서 S0′ 재인증 후 비교(§S13.47 규율).

---

## 계기판 (전부 직접 측정)

| 지표 | 값 | 의미 |
|---|---:|---|
| 인증 S0′ 패널 `tg_upside` z 가 +4.99 클립인 셀 | **APH 3,178행(2014~26 전 구간 z=5.0)** · DELL 728(2019~21 z=5.0) · BE 199 · APP 111 | APH 는 09-03 빈티지에서 TG/가격 2.33(로더 진단), 09-01·09-04 빈티지는 1.16 — **빈티지 고유 데이터 결함**이 인증 기준선에 들어감 |
| `tg_upside` 횡단면 z 표준편차 | 2014~21 **0.81~0.89**, 2026 0.96 | 클립 outlier 가 타 249종 z 를 11~19% 압축 |
| 명목가 분모(§S17.3) 이후 남은 벤더 기저 불일치 | RTX 2014~19 TG/UNADJ **0.63~0.69**(패널 z −3.2~−4.0) · T 2014~21 0.93 vs 이후 1.16(z −1.9~−2.6) · DELL 2017~21 1.95~2.72(z +5) · DHR 2014~15(z +3.1/+2.6) | FactSet TG 는 스핀오프를 어떤 건은 조정(RTX·T·MRK — 블룸버그는 특별배당으로 분류, UNADJ 미조정)하고 어떤 건은 미조정(DELL·DHR — 블룸버그 자본변경, UNADJ 조정) → 두 방향의 아티팩트 |
| vol-quality 틸트 × 음(−)자본 ROE | **157 종목-리밸 쌍 / 78/97 리밸일**, 점수 이동 평균 **−0.80 sd** — DELL 30·FICO 21·HCA 16·MSCI 15·RR/ 11·PM 9·LNG 8·ABBV 7(zq −4.4)·ADSK 4(−4.9)·ORCL 3(−4.3)·LOW 2(−5.2)·CL 1(−5.7) | §S17 M3 는 모델 채널만 반박 — **틸트 채널은 결함 실재**(비평가 #1 확증) |
| 집행 신뢰도 | raw 스프레드 3.05~3.61 vs `spread_scale` 0.20 → spread leg **100% 포화**; 신뢰도 바닥 0.2 가 **34%** 리밸일; eta 0.224~0.50(중앙 0.42) | IC lag-1 자기상관 **−0.087**, 최근 6회 평균→다음 IC 상관 **−0.17** → IC 기반 속도 조절은 잡음 |
| Σ 추정 분기 | **pairwise 85 / Ledoit-Wolf 10** (95 리밸), cond 1.2e6 vs 322; 실현/ex-ante TE 0.96 vs 1.16 | 창에 NaN 열 1개(중앙 11개)만 있어도 전체가 무수축 표본 Σ 로 전환 — 편향은 없으나(§S16.4 재확인) 설계와 문서가 다름 |
| Fwd_Sales_Slope 실커버리지 | 2018 **17%** · 2020 34% · 2021 58% · 2023 90% · 2024 97%(나머지는 per-date median 임퓨트) ; 블록 gain 2018~20 모델 4.4~9.1% | 벤더 고정회계연도 히스토리가 종목별 2017~2023 부터만 존재(생성기 마스킹은 정확, 99.6% 일치). §S13.25 채택 근거는 사실상 2021+ |
| 커버리지 갭 CS-median 임퓨트(비평가 #2) | TG 965셀(§S17.1-A 로 해소) · EPS_Rev 1,219(VRT 586·VST 222) · Sales_Rev 1,012 · **BEST_* 17시트·CUR_MKT_CAP·EQY_REC·NEWS 전부 0** | 비TG 시트는 무해(revision 중앙값 ≈ 중립) — 벤치마크 가중치 영향 0 |
| 백필 상수 접두(NaN 아님) | LIN 1,262행(2014-01→2018-10, EPS·SALES·ROE·PE·REC) · VRT 420(2018-08→2020-02 SPAC) · VST 157 · LSEG 130 | 자동 상장 추론(PX_LAST 평탄)이 못 잡는 "전신 히스토리"(TKO/HWM 선례) |
| 스핀오프(비평가 #3) | 14건 모두 PX_LAST 조정 완료(이벤트 ±7일 최소 일수익률 −0.6~−5.6%, 가짜 폭락 0) | Daily_Returns·모멘텀·Σ 는 정합 — 문제는 TG 기저만 |

---

## P — production 수치에 들어가는 결함

### P1. 목표주가(FactSet) 와 명목가(Bloomberg UNADJ) 의 기업행사 기저 불일치 — **high · 직접 측정**
`src/features/sellside.py:418-424`(`upside = tg/px − 1`, px = `local_prices_nominal`) · `src/data_loader.py:519-577`(`_apply_nominal_price`) · 워크북 `PX_LAST_UNADJ`(price_v4 `adjustmentSplit=True, Normal/Abnormal=False`)

**세 가지 클래스**:
1. **빈티지 고유(APH)**: 09-03 15:51 부분 재수집 빈티지에서 APH 의 TG/가격 = 2.33(트레일링 252d 중앙), 09-01·09-04 빈티지는 1.16. 인증 S0′(s17_5)와 그 OFF 짝(s17_2)이 **둘 다** 이 빈티지 → 두 런의 패널에서 APH `tg_upside` z 는 12년 내내 +5.0(3,178행). 로더 가드 `_check_target_price_unit_ratio` 의 허용대 [0.2, 5] 를 통과해 **무음**. 09-04 재생성 워크북으로 자동 해소됐고, 그 결과 production IR 1.7643(09-04 12:41, 같은 데이터 빈티지) → **1.6106**(09-07 12:36) 포크.
2. **자본변경형 스핀오프(블룸버그 `Split` 플래그, UNADJ 도 조정) + FactSet TG 미조정**: DELL/VMW 2021-11(내재 주식수 스텝 −0.68 → TG/UNADJ 2017~21 1.95~2.72, 패널 z +5.0 2019~21, 2020 평균 액티브 +0.73%p OW), DHR/Fortive 2016-07(−0.28, 패널 z +3.1/+2.6 2014~15), WDC/SNDK 2025-02(−0.28, TG/UN 1.35→1.17 부분), GE/GEHC 2023-01(1.15→1.04 부분).
3. **특별배당형 스핀오프(블룸버그 `Abnormal`, UNADJ 미조정) + FactSet TG 조정**: RTX/Carrier·Otis 2020-04(UNADJ/PX_LAST 스텝 −0.528 → TG/UN 2014~19 0.63~0.69, 패널 z **−3.2~−4.0** 6년), T/WBD 2022-04(−0.296 → 0.93 vs 1.16, z −1.9~−2.6 8년), MRK/OGN 2021-06(−0.048, 소폭). **이 클래스는 §S17.3 명목가 flip 이 만든 것** — 조정가(PX_LAST) 분모에서는 FactSet 와 기저가 같았다.

**해석**: `tg_upside` 는 gain 상위 피처(라이브 1.27%)이고 횡단면 z 이므로 소수의 ±5 아티팩트가 전체 분포를 압축한다(z 표준편차 0.81~0.89). 1·2·3 어느 것도 룩어헤드는 아니지만 학습기에 "종목 고정효과"를 심는다(RTX 6년 최하위, T 8년 하위, DELL 3년 최상위). 인증 S0′ 1.7633 은 클래스 1 이 포함된 수치다.

**처방(전부 사전등록 대기)**:
- (A, 즉시·비계수) **빈티지 가드 강화** — `_check_target_price_unit_ratio` 허용대를 트레일링 252d 중앙 [0.6, 1.7] 로 좁히고, 직전 런 metrics.json 의 `tg_px_ratio_median` 대비 종목별 |Δlog| > 0.25 이면 경고+HOLD 게이트(체크 7종째). 산출물 불변.
- (B, 정확성 arm) **TG 기저 정규화 이벤트 테이블** `tg_basis_events`(default 빈 dict): (ticker, event_date, factor) 를 데이터로 도출 — 클래스 2 는 내재 주식수 스텝(`Δlog(CUR_MKT_CAP/UNADJ)` < −0.08)에서, 클래스 3 은 `Δlog(UNADJ/PX_LAST)` < −0.03 스텝에서 factor 를 읽어 **이벤트 전 TG 를 factor 로 재스케일**. 데이터 게이트: 대상 6종(DELL·RTX·T·DHR·MRK·WDC) 의 연도별 median(TG/UN) 이 [0.85, 1.35] 로 진입 · 패널 |z| ≥ 4.99 셀이 APH 제외 연 ≤ 100 · `tg_upside` z 표준편차 연 ≥ 0.95. E2 do-no-harm. 대안 A′ = `EQY_SH_OUT` 인출 후 nominal = CUR_MKT_CAP/주식수(클래스 2·3 동시 해소, 사용자 Bloomberg 작업).
- (C, 즉시) **S0′ 재인증** — 09-04 14:50 워크북 빈티지(APH 정상)에서. 09-07 12:36 production 런(1.6106)이 config 동일·`--no-cache` 라 그대로 새 S0′ 후보 — 일별 런이 덮어쓰기 전에 metrics/manifest/pkl 을 `outputs/s18_s0recert/` 로 동결할지 사용자 결정.

### P2. vol-quality 틸트가 음(−)자본 고ROE 종목을 "최하 품질"로 벌점 — **high · 직접 측정 · 비평가 #1 확증**
`src/backtest.py:501-553`(`apply_vol_quality_tilt`: `panel["best_roe_level_z"]` 를 1/99 winsor z 로 재사용) · production `vol_quality_tilt_enabled: true`, λ 0.25

Bloomberg BEST_ROE 는 자기자본이 음이면 순이익이 양이어도 크게 음수다(자사주 매입 누적 기업: ABBV·ORCL·LOW·CL·MO·PM·HCA·MSCI·FICO·ADSK·SBUX·VRSN·DELL). 상위 idio-vol 터실에 이들이 들어오면 zq −2~−6 → `λ·sd·zq` = 평균 **−0.80 sd** 점수 이동(모델 z 표준편차 1 기준). 인증 S0′ 97 리밸 중 78일, 157 쌍. §S17 M3 는 "LightGBM 분할 90%가 본체 내부"라 모델 채널을 반박했지만, 틸트는 분할이 아니라 **선형 곱**이라 극단 z 가 그대로 전달된다.

**처방** — 사전등록 단일 플래그 `vol_quality_tilt_negative_equity_mask`(default-OFF): `BEST_ROE < 0 ∧ BEST_EPS > 0` 셀의 품질 입력을 NaN → 함수 계약대로 해당 셀 바이트 불변(틸트 미적용). 기전 게이트 = 쌍 157 → 0, 타 셀 바이트 동일. E2 do-no-harm + IR 관측(부호 무관, 정확성 트랙 §S16.1 선례). 대안(부호 인식 ROE) 은 §S17 대안 IC t 0.63 으로 이미 약함 — 마스크가 최소.

### P3. 집행 신뢰도 스케일링은 잡음 — **medium · 직접 측정**
`src/backtest.py:1216-1277`(`compute_signal_confidence`·`apply_dynamic_execution`), `confidence_spread_scale` 0.20

raw 예측이 횡단면 z 라 상하위 십분위 스프레드는 항상 3.0~3.6 → spread leg 는 **1.0 에 포화(97/97)**. 남는 것은 최근 6 리밸 IC 평균의 leg 인데 IC 는 지속성이 없다(lag-1 −0.087, 6회 평균→다음 −0.17, n 96). 결과: 34% 리밸일에 신뢰도 바닥 0.2 → eta 0.224(기본 0.50 의 45%), 밴드 1.5%. 즉 **무작위 시점에 거래 속도를 절반으로 줄인다**. 턴오버 캡 0.15 가 97/97 바인딩(§S17 T-05)인 상태에서 실행 스텝은 eta×15% ≈ 3.4~7.5%(실측 두방향 중앙 5.8%).

**처방** — G2-01(§S17 Tier 2) 을 **Tier 1 로 승격**: `static_execution_enabled`(신뢰도 ≡ 1 → eta 0.50·밴드 0.003 고정). §S13.11 은 eta 1.0(상향) 이 느린 신호를 해친다는 결과이지 **무작위 감속 제거**와는 다른 처치다. 기대: 회전율 +10~20%, IR 소폭 +; E1 표준(ΔIR>0.36 & 3분할) 대신 §S16.7 식 리스크 규율 프레임(IR no-harm) 로도 채택 가능한지는 사용자 결정.

### P4. `fwd_sales_slope` 4피처의 실효 히스토리가 짧다 — **medium · 데이터 한계(코드 결함 아님)**
`create_ai_signal_data.py:678-716`(`_mask_backfilled_prefix`, 정확) · `src/features/fwd_sales_slope.py`

BEST_SALES_2FY 고정회계연도 히스토리가 종목별로 2017~2023 부터만 실측(그 전은 벤더 평탄 백필, 중앙 1,810 캘린더일)이라 slope 시트 실커버리지는 2018 17%·2020 34%·2021 58%·2023 90%. 비커버 종목은 per-date median 에 고정 → 모델은 "값이 중앙값과 다름"(=커버 여부) 을 분리 가능. 커버 지표의 타깃 IC 는 유의하지 않아(2016 만 t −2.56) 커버리지가 알파를 위장한 증거는 없으나, 블록 gain 4.4~9.1%(2018~20 모델) 은 42~88종 표본에서 학습된 것이다. **§S13.25 의 +0.266 은 2021+ 커버리지에 기대는 수치**로 재해석. 처방: 문서화 + (선택) 검증 arm "slope 블록을 커버리지 ≥ 50% 시점(2021-06) 이후만 admit" 으로 이득 재현 여부 확인(성과 아닌 타당성 검사).

### P5. Σ 추정기가 사실상 무수축 표본 공분산 — **medium · 설계 불일치**
`src/portfolio_optimizer.py:88-98`

250종 × 126일 창에 NaN 열이 하나라도 있으면(상장 마스크·결측) 전체가 `_pairwise_covariance`(rank ≤ 125, 고유값 floor) 로 전환 — 95 리밸 중 **85**. Ledoit-Wolf 는 10회만. 실현/ex-ante TE 는 0.96(pairwise) vs 1.16(LW) 로 과소예측은 없고(§S16.4 무편향 재확인) §S13.48b 가 floor 방향 액티브 분산 0% 를 보였으므로 성과 피해는 작다. 처방: 통일(완전 열 LW + 결측 열 pairwise 채움) 은 §S16.4 결론상 IR 기대 0 → **문서·가드만**(추정기 분기 카운트를 metrics 에 기록).

### P6. 전신(前身) 히스토리 — VRT SPAC 셸·LIN Praxair 기간 — **medium-low**
`src/config.py:151-155`(`"VRT": "2018-08-01"` "genuine SPAC-predecessor history" 로 의도 등록)

VRT 2018-08~2020-02 는 GSAH SPAC($10 근처, 펀더멘털 BEST_EPS/SALES 420행 상수, 목표주가 갭, de-SPAC 일 내재주식수 ×3.8) — 경제적 실체가 Vertiv 가 아니다. LIN 은 2014-01~2018-10 BEST_* 1,262행 상수(가격은 Praxair 계열로 변동). TKO/HWM "무빙 전신" 선례에 따라 `VRT: 2020-02-10`, `LIN: 2018-10-31` 오버라이드 후보. 영향 2종·소폭(정확성 트랙, 비계수).

### P7. 저심각 관찰
- 금융주의 정의 불가 펀더멘털이 NaN 아닌 **상수**로 존재(GM 24종 >756행, EV/EBITDA 12, FCF 14, CAPEX 15, PEG 9): chg=0·tsz→NaN→median 으로 중립화됨. §S13.6(NaN 이 median 보다 해로움)과 일관 — 무변경.
- OPER_MARGIN 은 전 종목 첫 ~111행 상수(연 1회 갱신 필드의 첫 구간) — 무해.
- 거래비용 편도 10bp 균일: 영국 인지세 50bp(매수)·KR/FR/IT 거래세 미반영. 연간 TC 6.8bp 규모라 IR 영향 1~2%.
- IC 게이트·미 휴장일 리밸(P5 §S17)·IR 정의 3종(T-12) 은 기존 항목 유지.

---

## 성과 개선 후보 (우선순위, 전부 사전등록·사용자 결정 후 실행)

| 순위 | 후보 | 유형 | 단일 파라미터 | 결정 게이트 | 근거·사전확률 |
|---|---|---|---|---|---|
| **0** | **S0′ 재인증**(09-04 빈티지) + 빈티지 가드(P1-A) | 운영 | — | 09-07 production 런 동결 여부 | 없으면 이후 모든 비교가 무효(§S13.47) |
| **1** | **P2 틸트 음자본 마스크** | 정확성 | `vol_quality_tilt_negative_equity_mask` | 기전(157→0)·E2·IR 관측 | 전례 정확성 팩 6/6 채택; ABBV·ORCL·LOW·CL 벌점 해소 |
| **2** | **P1-B TG 기저 이벤트 정규화**(또는 A′ EQY_SH_OUT) | 정확성 | `tg_basis_events` | 6종 TG/UN [0.85,1.35]·클립 ≤100/yr·z sd ≥0.95·E2 | RTX 6년·T 8년·DELL 3년 종목 고정효과 제거 |
| **3** | **P3 정적 집행**(신뢰도 ≡ 1) | 실행 | `static_execution_enabled` | E1 또는 no-harm 프레임(사용자 결정) · turnover ≤ 1.25× | IC 무지속성 실측(−0.09/−0.17); 34% 리밸 감속 제거. §S13.11 과 처치 상이 |
| 4 | G4-01 63BD 라벨(§S17 Tier 2) | 알파 | `forward_horizon: 63` | §S17 게이트 그대로 | §S11.8 느린 알파(21~63d IC 0.069) |
| 5 | G5-02 메가캡 vol 수축 OFF(§S17) | 리스크 | `cov_megacap_vol_shrink_enabled: false` | QLIKE 5분 사전점검 | 코드 0 |
| 6 | P6 전신 오버라이드(VRT·LIN) | 정확성 | `listing_dates` 2줄 | 데이터 게이트(상수 접두 0) | 소폭 |
| 7 | P4 slope 커버리지 타당성 arm | 검증 | admit 시점 게이트 | 이득 재현 여부 | 성과 목적 아님 |
| — | G5-01 Σ 5일 겹침·G3-04 섹터 리스크 몫 캡 | §S17 잔여 | — | §S17 | 변경 없음 |

**종결 재확인(재도전 금지)**: 커버리지 갭 CS-median 임퓨트의 비TG 확장(BEST_*·시총 0셀) · 스핀오프 가격 조정(전부 정합) · Σ 수축(§S16.4) · eta 상향(§S13.11) · 음자본 ROE 의 **모델** 채널(§S17 M3).

---

## 제안 실행 순서
1. 사용자 결정: 09-07 production 런을 새 S0′ 로 동결(`outputs/s18_s0recert/`) 또는 별도 재인증 런. 가드 P1-A 는 산출물 불변이라 즉시 적용 가능.
2. 정확성 arm 2건 사전등록·측정: 후보 1(틸트 마스크) → 후보 2(TG 기저). 각각 default-OFF + 패리티 테스트 + 단독 커밋 + schtasks 단일 런.
3. 후보 3 정적 집행은 새 S0′ 위에서 E1/no-harm 프레임을 사용자가 비준한 뒤.
4. 후보 4~7 은 §S17 순서 유지.

## 부록 — 산출물
- `outputs/s18_prechecks/probe_a_results.json`(A1 커버리지 갭·A1 시총 bm 영향·A2 14 스핀오프·A2 내재주식수 점프 106건·A2 TG/UN 이상 종목-연도·A3), `probe_c_results.json`(시트별 상수 접두), `probe_b_results.json`(B1 틸트·B2 신뢰도·B3 회전율·B4 Σ 분기·B5 slope 커버리지·B6 클립·B7 lag 검증 0.0), `probe_b_cov_branch.csv`(리밸일별 분기·ex-ante/실현 TE).
- 비교 런: `outputs/s17_5_nominal_price/metrics.json`(S0′, APH 2.329), `outputs/codex_causal_rank_65/metrics.json`(09-07, APH 1.166, IR 1.6106), git `d62e7ba`/`8489798` 의 production metrics(09-01 빈티지 1.7330 / 09-03 빈티지 1.7643).
