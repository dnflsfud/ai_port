# ai_port 구조 감사 4차 (§S17 리뷰 트랙) + 성과 개선 후보 탐색 — 2026-09-02

> **방법**: 3단 워크플로우 + 메인 직접 재현. ① 코드 렌즈 6종(S16.6 이후 변경분·시간 정합·옵티마이저/실행·데이터/피처 단위·학습/설정·측정/판정, 읽기 전용) → 원시 20건 → 트리아지 18건(기지 중복 0) → 반박 우선 검증 8건 → **확정 7 / 반박 1 / 저심각 미검증 10**. ② 메인이 인증 S0′ pkl 감사 팩 + 워크북 원시 7시트 프로브로 리드 M1~M5 도출 → **독립 경험적 검증자 4인 + W1 배치 검증자(10건)** → M1·M2·M4 확정, **M3 반박**, T-08 기각. ③ 개선 후보: 5관점 생성기 → 관점별 회의적 심사(결정 로그 grep·불변식·사전점검 실행가능성) → 후보 24건 중 **사전점검 진행 가능 14 / 종결축 1 / 기각 9**.
> 총 32 에이전트(W1 16 · W2 10 · W3 6) · 1,143 도구 호출. 레포·결정 로그·production 무변경 상태에서 수행(스크래치는 레포 밖). 인벤토리 471 불변(read-only).
>
> 아티팩트 판본: https://claude.ai/code/artifact/48c4b07c-6a6c-436b-a7b1-25eed6f9709e
>
> **기준선**: 인증 S0′ IR 1.7596 / TE 3.71% / avg_ic 0.01818 / 퇴화 14/33 / ECOS 194·fallback 0 (`outputs/s16_7_name_risk_cap`, 워크북 08-25 20:47:55 · Index 09-01 빈티지 쌍). **주의**: 09-02 11:30 스케줄 런은 워크북 09-01 14:05 KST 리프레시 + Index 09-02 빈티지에서 **IR 1.7052 / TE 3.64% / avg_ic 0.01631** — 데이터 리프레시만으로 −0.054. 이후 모든 arm 은 새 빈티지 쌍에서 S0′ 재인증 후 비교(§S13.47 규율).

**표기 규약**
- **직접 재현** = 메인이 `outputs/s16_7_name_risk_cap/backtest_result.pkl`·워크북 원시 시트로 독립 관측한 수치. **검증됨(n렌즈)** = 발견자와 독립된 반박 우선 검증자가 확인. **경험 확정** = W3 검증자가 pkl/시트로 정량 재판정.
- 어떤 항목도 §8 절차(사전등록 → 측정 → 사용자 결정) 없이 production 에 적용되지 않는다. 이 문서는 결정 로그가 아니다(결정 로그 §S17 이 정본).

---

## 계기판 (전부 직접 재현 + 경험 확정)

| 지표 | 값 | 의미 |
|---|---:|---|
| PX_LAST 2014-06-30 / 명목 종가 | MO 0.466 · T 0.373 · VZ 0.527 · AAPL 0.876 · **무배당 TSLA/AMZN/ADBE/NFLX/ISRG 1.000(4자리 일치)** | 워크북 가격이 **배당 재투자 소급 조정(Bloomberg DPDF)** — 분할이 아니라 배당 |
| 배당락 서명 | log(CUR_MKT_CAP/PX_LAST) 일별 스텝: MO 음수 74건·양수 0, 배당락일과 일치, −1.1%/건 | 시총은 명목, 가격만 조정 → 내재 주식수 MO 2014 2.15배·T 2.68배 과대 |
| 같은 조정을 받은 시트 | BEST_PE_RATIO·BEST_PX_BPS_RATIO·BEST_PEG_RATIO(배당락 스텝 비 ≈0) / EV_EBITDA·CUR_MKT_CAP 명목 | **PE×EPS ≡ PX_LAST(98.6%가 1% 이내) → 명목가 프록시 무효**; 2014 MO 시트 fwd PE 7.7 vs 실제 ~16 |
| 고배당 12종 median(TG/PX) | 2014 **2.02** → 2018 1.78 → 2022 1.47 → 2026 1.10 (무배당 1.14→1.31 평탄) | `tg_upside` 가 미래 배당 누적분만큼 부풀려짐 → 라이브로 갈수록 소멸 |
| 고배당군 패널 tg_upside z | 2014~19 중앙 **+1.44** → 2025~26 **−0.12** (백분위 0.95 → 0.30; MO 2.34→−1.10, RIO 4.71→−0.98) | 학습기와 라이브의 **순위 반전** — 2026-08-25 라이브 12종 전부 음수 |
| tg_upside 의 IC (21일 격자 144일) | 현행 −0.0006(t −0.05) vs 탈조정 **+0.028(t 2.78)**; 아티팩트항 IC −0.026(t −2.5) | 아티팩트는 성과를 부풀리지 않고 **학습기 신호를 지웠다**; 라이브 4모델에서 tg_upside gain **0.0** |
| 목표주가 커버리지 갭 임퓨트(T-01) | VRT 409행·VST 127행 +5.0 상수, 갭창 타 249종 \|z\| q95 0.777 vs 1.875/1.924 (**2.4× 압축**) | \|z\|≥4.99 클립이 3,283일 중 1,335일에 존재(DELL 297·APP 118·RIO 114행은 M1 기전) |
| 아시아 16종 vs US(주중 달력) | 동시 corr 0.124 < 전일 **0.283**, Dimson 3.82; 주간 corr 0.382(**3.44× 회복**), 주간 β 0.73 | 시차 비동기 거래 확정 — T-08(JP T+1 스탬프)은 **기각**(피크 lag1 16/16, 08-05 폭락 같은 행) |
| Σ 아시아×미국 블록 corr(97 리밸일) | 일별 0.044 vs 5일 겹침 0.201, **97/97 날짜에서 3.1×** (EU×US 1.27×, US×US 0.94×) | 아시아 순 OW +1.5%(2026년 +2~3%)의 ex-ante 리스크 점유 10.9%→13.6% 과소; 북 TE 영향 +2% |
| 코어 65 피처 중 모델이 소비 못 하는 것 | **9** (bcast 7 = 33/33 모델 split 0 · 완전 중복 2 = 820,750셀 max diff 0.0) | EWMA 드롭 집합이 31/31 재훈련에서 죽은 피처 3개로 고정 → 피처 선택층 inert; §S15 fix-pack #2 는 모델 no-op |
| 음(−)/0 자본 ROE·P/B (M3) | 압축은 실재(극단셀 제외 z std −75%)이나 예측 백분위 0.513 vs 실현 0.500, 액티브 +0.0001, 대안 변환 IC 차 t 0.63 | **반박** — 설계 관찰(low)로 강등 |
| 라이브 production (09-02) | IR 1.7052 / TE 3.64% | 인증 S0′ 1.7596 대비 새 빈티지 쌍 효과 −0.054 |

---

## 한눈에

1. **가장 큰 구조 결함은 코드가 아니라 입력 계약에 있다.** 워크북의 PX_LAST 와 가격 파생 비율 시트(PE·P/B·PEG)가 배당 재투자 소급 조정본이라, 명목 목표주가를 나누는 `tg_upside`(제거 시 IR −0.48 기록의 핵심 피처)와 §S16.1 P3 의 주식수 유도·PE/PB 레벨 z 등 **코어 10/65(gain 합 ~14%)**가 12년 내내 미래 배당을 담고 있었다(M1, critical). 효과는 "성과 과대"가 아니라 **학습기 신호 소거 + 학습→라이브 순위 반전**이며, 그 궤적대로 라이브 4개 모델은 tg_upside 를 이미 전혀 쓰지 않는다. 같은 피처에 FactSet 커버리지 개시 전 구간의 타 종목 가격 임퓨트(T-01, high)가 겹쳐 있다.
2. **명목가 복원은 코드로 불가능하다** — PE×EPS 프록시는 PE 시트가 같은 조정을 받아 무효. 워크북에 배당 미조정 `PX_LAST_UNADJ` **별도 시트**(또는 EQY_SH_OUT) 재인출이 필요하고, PE/PB/PEG 는 `X × PX_UNADJ/PX_LAST` 로 역산 가능. 사용자 Bloomberg 작업 + 새 빈티지 + S0′ 재인증이 선행 조건.
3. **아시아 종목은 시차 때문에 Σ와 β에서 체계적으로 잘못 계측된다(M2, medium).** 실체는 리스크 채널(Σ 블록 3.1× 과소, 아시아 액티브 리스크 점유 ~25% 상대 과소)이고, 리드가 의심한 idio_vol_63d 오염은 +4~5%에 그쳐(아시아의 고 idio 터실 편입은 대부분 진짜) 피처 채널 가치는 작다.
4. **개선 후보 중 기대값이 가장 높은 것은 여전히 정확성 수정 팩이다**(전례 5/5 채택). 알파·실행·구성 축 12건은 전부 탈락 가능한 사전점검 게이트로 설계됐고 사전확률은 대부분 낮다. 음의 자본 가드(G1-02)는 M3 반박으로 SHELVE.

---

## P — production 수치에 들어가는 결함

### P1. `tg_upside`·PE/PB/PEG·주식수의 분모가 배당 재투자 조정 가격 — 미래 배당 룩어헤드 + 라이브 순위 반전 (M1) — **critical · 직접 재현 · 경험 확정(0.92)**
`src/data_loader.py:1438`(`local_prices = sheets["PX_LAST"]`) · `src/features/sellside.py:368`(`upside = tg/px − 1`) · `src/features/accounting.py:160`(`shares = market_cap/prices`) · `assembly.py:564-573`·`accounting.py:104-105`(PE/PB/PEG level_z, 시트 자체가 조정) · 원천 `re_study/create_universe_data.py:368`(Daily_Returns = PX_LAST.pct_change) ← `Data/S&P500.xlsx` Bloomberg 인출(DPDF 배당 조정 ON)

**증거**: (1) 2014-06-30 PX_LAST / 명목 종가(분할 조정): T 0.373 · MO 0.466 · VZ 0.527 · PFE 0.557 · XOM 0.603 · KO 0.688 · PG 0.708 · JPM 0.723 · MSFT 0.837 · AAPL 0.876 — 배당수익률이 높을수록 낮다. 무배당 TSLA 16.004/16.004 · AMZN 16.239/16.239 · ADBE · NFLX(2025-11 10:1 반영) · ISRG(3:1) 는 **4자리 정확 일치** → 분할이 아니라 배당. (2) CUR_MKT_CAP 은 명목(MO 83,312M 등 6종 0.1% 내) → mktcap/PX 내재 주식수가 MO 2.15배·T 2.68배·VZ 1.89배 과대. (3) **배당락 서명**: log(CUR_MKT_CAP/PX_LAST) 일별 스텝(|Δ|>0.3%) MO 74건 전부 음수, 2014-03-12/06-12/09-11/12-22 … = 배당락일, 크기 −1.1% ≈ 분기 수익률; T·KO·XOM·VZ·PFE 각사 달력과 일치; TSLA·AMZN 음수 0. (4) 같은 스텝이 BEST_PE_RATIO(−0.02)·BEST_PX_BPS_RATIO(−0.014)·BEST_PEG_RATIO(−0.016)에는 없고 EV/EBITDA(−0.85≈−1)에는 있다 → 비율 시트 3종도 조정본, **PE×EPS ≡ PX_LAST**. (5) 날짜 정렬 후 고배당 12종 median(TG/PX) 2014 2.02 → 2018 1.78 → 2022 1.47 → 2026 1.10 vs 무배당 1.14→1.31; AAPL/NVDA/TSLA 분할일에 TG·PX 모두 연속(TG 는 분할만 조정된 명목).

**정량**: 21일 격자 144일 IC — 현행 tg_upside −0.0006(t −0.05) vs 탈조정 +0.028(t 2.78), 아티팩트항 −0.026(t −2.50); 리밸일 96일에서는 0.014 vs 0.012(차이 무의미). 두 버전 날짜별 스피어만 연도 중앙값 2014 −0.18 → 2019 0.00 → 2024 0.30 → 2026 0.94. 패널 고배당 12종 tg_upside z 2014~19 +1.44 → 2025~26 −0.12(기울기 −0.148 z/년; 백분위 0.95 → 0.30; 무배당 −0.92 → +0.60 역전). cash_conversion_z 고배당군 −0.234 → +0.021. 모델 gain: tg_upside 중앙 1.5%(2019~20 3%·순위 7~9 → 2025-11 이후 4개 모델 **0.0**·순위 59), 영향 10피처 합 13.5~14.5%.

**해석**: 룩어헤드(조정계수 = t 이후 배당)이자 횡단면 오염(2014~21 tg_upside 로그 분산의 10~16%가 배당 더미)이며, IC 를 부풀린 것이 아니라 학습기 신호를 지웠다. 과거 "tg_upside 제거 시 IR −0.48" 기록은 오염 피처 기준. pre-causal 시절부터 모든 baseline·arm 에 공통이라 상대 순서는 대체로 보존되나 **절대 수치는 전부 재인증 대상**. 성과 방향(수정 시 IR 상승/하락)은 백테스트 없이 미정 — 채택 근거는 정확성.

**처방** — (A, 권장) 워크북에 배당 미조정 `PX_LAST_UNADJ` **별도 시트**(기존 PX_LAST 는 Daily_Returns 파생원이라 교체 금지; 총수익 모멘텀·52주고가·Σ·수익률은 현행이 정합). 로더에 `local_prices_nominal` 부착(§S16.1 P1 LN×0.01 동일 적용) → sellside.py:368/432·accounting.py:160·backtest.py:362 사용; PE/PB/PEG 는 `X × PX_UNADJ/PX_LAST` 역산. 검증: 무배당 5종은 PX_LAST 와 바이트 동일, MO 2014-06-30 ≈ 41.94. (A′) EQY_SH_OUT 인출 후 nominal = CUR_MKT_CAP/EQY_SH_OUT. (B) PE×EPS 프록시 — **무효**. (C) 시총 스텝 복원 프록시 — 바이백 오염(앵커 |오차| 중앙 0.15 log, AAPL 0.44) → 진단 전용. 사전등록 단일 파라미터 `nominal_price_source ∈ {off, PX_LAST_UNADJ}`(default OFF·바이트 parity; 데이터 정확성 계층으로 §2.1 예외 default-ON 후보). 채택 기준 = 데이터 정확성 게이트: 무배당 5종 |Δtg_upside|<1e-9 · 고배당 12종 2014~16 TG/PX 2.0→1.0~1.25 · 내재 주식수 MO 2014 1.98bn±3% · 고배당군 학습기-라이브 z 격차 1.56 → <0.5; 새 S0′·turnover·퇴화율·avg_ic 보고, DSR 해킷 기록, 사용자 승인 후 flip(§8 체크리스트). 부수: `_check_target_price_unit_ratio` 에 "배당락일 내재 주식수 음수 스텝(연 4회+)" 가드 추가.

### P2. `_fill_missing` 횡단면 median 이 목표주가 커버리지 갭을 타 종목 가격으로 채움 (W1 T-01) — **high · 검증됨 2렌즈 · 경험 확정**
`src/data_loader.py:443-451` → `preprocess_sheets`(:659-668) → `sellside.py:368`

ffill 후 잔여 NaN(커버리지 개시 전)을 `df.median(axis=1)` = 타 종목 목표주가 레벨로 채운다. 상장 마스크는 상장일만 다루므로 상장 후 커버리지 갭은 어느 패스에서도 재마스킹되지 않는다. **실측**: VRT 2018-08-01~2020-02-24 **409행 전부 +5.0**, VST 127행, PLTR 16·CRWD 15·LITE 4행 +5.0, BE 평균 3.97; 갭 종료 후 63BD `tg_mom_63d` VRT −4.93(≤−4.99 비율 93.8%); 갭창 타 249종 |z| 95분위 0.777 vs 직전 1.875·이후 1.924(**2.4× 압축**, 중앙값 0.247 vs 0.571). 학습창 1,260일의 31%.

**처방** — 가격 레벨 시트(Factset_TG_Price·Fwd_OpCashflow)의 선행 결측은 NaN 유지 → 패널 per-date median(z≈0), §S13.6 교훈대로 네이티브 NaN 경로 아님. 코드만으로 가능 — P1 데이터 확보 전에 **단독 선행** 가능(같은 피처 입력 계약이라 P1 과 한 플래그로 묶는 것이 §S16.1 선례이나, 데이터 대기 시 분리).

### P3. 시차 비동기 거래 — Σ 아시아×미국 블록 3.1× 과소, beta_63d 3× 과소 (M2) — **medium · 직접 재현 · 경험 확정(0.85, 범위 한정)**
`src/portfolio_optimizer.py:73-129 estimate_covariance/_pairwise_covariance`(126d 일별) · `src/features/price.py:167-180`(동시 rolling β vs EW 시장)

**증거(주중 달력, 현지통화)**: ASIA 16종 동시 corr 0.124 vs corr(stock_t, US_{t−1}) 0.283, β0 0.218 vs β_lag1 0.498, Dimson 3.82; 주간(5일 합) corr 0.382 = 3.44× 회복(21d 3.5×), 주간 β 0.73(US 1.02), 서브기간 3.19/2.59/4.45 전 구간; EU 1.22×, US 0.79. FX 변환·주말 행 아티팩트 기각(현지통화·주중 달력에서 더 강함). **Σ**: 97 리밸일 126d Σ 의 ASIA×US 블록 corr 일별 0.044 vs 5일 겹침 0.201, 97/97 날짜 양수 차이, 중앙 3.1×(EU×US 1.27×, US×US 0.94×). **북**: 아시아 액티브 순합 평균 +1.53%(양 72/97, 2026년 +2.2~3.1%, 285A OW 0.9~2.7%, 6857 +1.03%) — 구조적 OW; ex-ante TE 3.25%→3.31%(+2%), 아시아 ex-ante 액티브 분산 점유 10.9%→13.6%, 실현 TE 주간/일별 1.03, 이름 리스크 캡 0.35 교차 전환 0/152.
**피처**: beta_63d 아시아 0.324 vs Dimson 0.997(3× 과소, gain 2.8% 순위 7). idio_vol_63d(gain 17.7% 순위 1)는 아시아에서 **+4~5%만** 과대(16/16 > US) — 아시아의 상위 idio 터실 편입(0.57 vs US 0.31)은 비동기 몫 ~1/5(9.1→8.3종목)이고 나머지는 US 지배 EW 단일시장 대비 진짜 낮은 R²(주간 corr 0.38 vs 0.56)+높은 USD 총변동성 → **리드의 idio_vol 귀속은 대부분 틀림**. vol-quality 틸트 편입 변화 −0.8종목/일(무영향). PCA 타깃 오염은 미검증(21d 지평에서 ≤1/21, 무시 가능 추정).
**T-08(JP T+1 스탬프) 기각**: 교차상관 피크 lag1 16/16(k2≈0), NKY 동일날짜 corr 0.69~0.93, 2024-08-05 폭락(7203 −13.7%·NKY −12.4%)이 같은 행; 2014 연초 5행 고정은 Index NKY 도 동일한 **선행 백필**(01-06 종가) = 휴일 2+주말 2+01-06 = min_flat_run 5 → 추론 01-07. 피해: 패널 시작 전 3행 과다 마스킹 = 0.

**처방** — (A) Σ 채널: lag-1 교차공분산 보정(Scholes-Williams/NW, Bartlett w=0.5, PSD 유지) 또는 5일 겹침 상관 — default-OFF, 알파 비트 불변, do-no-harm 프레임(ΔIR ≥ −0.05·3분할 비악화, ASIA×US 블록 ≥2× 상승·US×US |Δ|<10%·캐릭터 보존). 심사자 주: 정확성 트랙이 아니라 §S13.41/§S16.7 리스크 규율 프레임(DSR +1), IR 이득 기대 0. (B) 피처 채널: beta_63d lag 회귀(β0+β1)·idio 잔차 — E1 표준 프레임, 사전 기대 idio +4~5%·터실 −0.8종목이라 가치 작음.

### P4. 코어 65 피처 중 9개를 모델이 소비하지 못함 — bcast 7 + 완전 중복 2, EWMA 선택층 inert (M4 / W1 T-10) — **medium · 직접 재현 · 경험 확정(0.93)**
`src/features/assembly.py:190 CORE_FEATURE_WHITELIST`(:251/:261/:262 등재) · `conditioning.py:45`·`factor.py:107`·`regime.py`(bcast) · `assembly.py:554/564-565`(중복) · `model_trainer.py:143-149`(드롭 산술)

33개 모델(고유 19) 전부에서 활성 bcast 피처(cal_is_Q1·regime_mkt_ret_21d·fac_F_Quality/Growth/Value_mom_63d·fac_value_growth_63d·fac_yield_slope)의 split 0·gain 0 — 패널에서 날짜별 cs_std==0 비율 100%(쿼리 내 상수)·시계열 가변(n_unique 2~3,264): '데이터가 죽은' 것이 아니라 날짜 그룹 rank_xendcg 가 소비 못 하는 구조(§S13.18 기전; regime.py:10-17 독스트링의 'GBT 상호작용 분기로 작동'은 실측 반증). fin_roe_level_z ≡ best_roe_level_z, fin_pb_level_z ≡ best_px_bps_ratio_level_z(820,750셀 max|diff| 0.0, 코드 동일식). 드롭 집합 31/31 = {fac_F_Quality_mom_63d, fac_F_Value_mom_63d, fac_yield_slope}(EWMA 리플레이 19/19 일치, 죽은 7종 importance 1.75e-5 동률 59위 vs 라이브 최저 0.00239 = 137배) → **라이브 피처 드롭 0건**; 9개 제거 시 56 → n_drop 2, n_keep max(54,60)→56 → 드롭 0(선택층 완전 inert). colsample 슬롯 가설 기각(포함확률 0.800 vs 0.793 불변), 중복 2쌍은 가용확률 0.80→0.96 암묵 편향. **§S15 fix-pack #2(fac_* 캘린더)는 post-fix 33/33 split 0 → 모델 no-op**(결정 로그 7078행 '퇴화 +2 ← 팩터 캘린더' 귀속 근거 없음). 부수: 라이브 모델(2025-11-25 fit, 27트리)에서 tg_upside 도 split 0.

**처방** — (A) whitelist 에서 9개 제거(best_* 사본 유지 — interactions.py:24-26 참조) → RNG 실현 변경으로 바이트 parity 불가, ΔIR 은 시드 잡음(±0.19)으로 취급·성과로 읽지 말 것; `ewma_min_features` 60 은 'EWMA 피처 선택'이 사실상 없는 기능임을 결정 로그에 명기. (C, 무변경) 단위테스트 가드: 화이트리스트 피처의 날짜별 cs_std==0 비율 임계·두 열 max|diff|==0 실패. 심사자 결론: 단독 arm 가치 0(§S12.3(i) 선례) — 다음 정확성 팩 동봉 또는 (C)만.

### P5. 미 휴장일 행 리밸(4/97)에서 US 종목 lag 0일 (W1 T-09) — low · 경험 확정
2018-12-25·2024-02-19·2025-12-25·2026-06-19 전부 리밸일, US 191종 수익률 0, 비US 거래 13/56/14/59종; predictions[h] ≡ pre_execution[h−1]. §S16 O5 캘린더 축과 동일 — 캘린더 정합 팩으로 이연. ΔIR 미측정.

### P6. 음(−)/0 자본 종목의 ROE·P/B 극단값 (M3) — **반박 → 설계 관찰(low)**
압축은 실재: 극단셀(|ROE|>100 ∨ ROE<0∧EPS>0 ∨ P/B>50)이 적격 셀의 9.9%(일평균 ~24종), best_roe_level_z 횡단면 std 0.755 → 극단셀 제외 0.191(−75%), 비극단 240종 IQR 0.198, CL +4.32·MCD/PM 최하위·fin_roe_pb_gap 8종 −3 고정 12년. 그러나 피해 기전은 반박: LightGBM 분위수 비닝이라 best_roe_level_z 분할 1,462건의 gain 가중 90.3%가 |임계|<0.5(압축 본체 내부), 극단 격리 0.4%; 음자본 8종 예측 백분위 0.513 vs 실현 0.500(체계적 오예측 없음); fin_roe_pb_gap 핀 기여는 '최악 품질'이 아니라 **양(+)의 오프셋(17/19 모델)**, 크기 점수 sd 의 0.22(2024+ 모델 ≈0); 액티브 비중 +0.0001(구조적 UW 아님); 대안 변환 IC — 부호 인식 +0.0034(t 0.63), 극단→중앙값 −0.0023(t −1.03), rank/winsor 는 Spearman 불변. 종목 고정효과 비중(0.27/0.61/0.66)은 vol·beta·analyst_rec 도 동급(레벨 피처 일반 성질). **권고 SHELVE(no-flip)**; 다루려면 M4·M5 와 묶은 '피처 위생' 단일 arm.

### P7. capex_intensity_z 극단 고정 (M5) — low
NEE −5.0(클립 고정) · C −4.16 · UBSG −3.30 · SO −3.18 · TSM −2.83 · MU −2.79 · EQIX −2.70 (12년 중앙값), 종목 고정효과 0.93. 설계 관찰.

---

## D — 판정 무결성·운영 결함 (production 가중치 불변)

| ID | 심각도·판정 | 위치 | 내용 | 처방 |
|---|---|---|---|---|
| T-03 | medium · decision_integrity · 경험 확정 | `portfolio_optimizer.py:675`(TOL 0.01) vs `export_operating_data.py:1708-1711`(strict 0.35, \|기여\|) · `validate_portfolio_bundles.py:668` | §S16.7 캡의 보장 상한은 0.36. 재구성 Σ로 실행 북 Euler share 재계산(공식 worst5 대비 오차 ≤0.0016): **strict 0.35 breach 12/97(확실 7·경계 5)**, (0.35,0.36] 12, >0.36 0(공식 0/97 정합). 결정 로그 §S16.7 '0/97·상수 공유' 서술 부정확; 수렴 북이 라이브가 되면 HOLD 가능(라이브 08-18 0.343) | export 가드를 `> cap + NAME_RISK_CAP_TOL` 로 정합(산출물 불변) + §S16.7 절 정정 |
| T-04 | medium · decision_integrity · 경험 확정 | `run_variant.py:332→482→578→588`, `config.py:1496-1501` | 데이터 빈티지·git 지문이 **런 종료 시점**에 stat — §S16.8 1차 런 무효 사고가 정확히 이 경로(1차 metrics fx_mtime 02:15:42Z·size 0 = 로더가 읽었을 수 없음, 시작 ≈01:52Z). 현 outputs 5개 런은 정합 | UniverseData 생성 직전 지문을 정본으로, 종료 시 재stat 을 `data_vintage_at_end` 병기·불일치 경고 |
| T-02 | medium · decision_integrity · **latent** | `scripts/eval_s16_7_arm.py:183`, `eval_s16_8a_arm.py:122`, `eval_s16_3_arm.py:92` | 'fallback 0' 게이트가 SCS 전환률(production 은 SCS 비허용 → 구조적 0.0)만 읽고 bm 폴백(`optimizer_failure_rate`, pkl 전용)에 눈멂; 'ECOS 194'는 infeasible 도 세는 카운트. 4개 arm pkl 전부 failure_rate 0 → 판정 뒤집힘 0 | pkl `optimizer_failure_rate`·`fallback_reason_counts` 를 G2/E2 로, metrics.json 병기, 'ECOS n' 은 optimal 수로 |
| T-05 | low · ops · latent | `portfolio_optimizer.py:741-753`, `backtest.py:1689-1699` | 캡 재해 실패·비수렴 미집계. 오프라인 재해 97일: MVO 캡 반복 {0:85,1:3,2:7,3:1,4:1}·비수렴 0·최대 0.357; 폴백 0. **부수 관측: MVO 타깃의 턴오버 캡 0.15 가 97/97 바인딩**(실행 북은 1/97) — §S16.5-P 갭₁ 33% 의 형태 | 집계 루프에 `name_risk_cap_nonconverged`·반복·max_share 분포 추가 |
| T-07 | medium · dormant · latent(근접 주장 기각) | `portfolio_optimizer.py:315-453`, `backtest.py:1650` | 게이트 강제거래 L1 > 0.15 면 MVO·투영 infeasible → bm 점프. 실측 강제 L1 최대 0.072(2024-03-19), q50/q90/q99 0.0068/0.029/0.044, >0.12 0회; 2020-04-08 0.026 — 헤드룸 2배 이상. 65종 시절 6/94 발동 이력 | `projection_fallback_mode: prev`(기존 스위치) 거버넌스 검토 — W2 G2-03, 교착 분석 후 |
| T-12 | low · decision_integrity · 경험 확정 | `src/utils.py:62`, `harness.py:44-46` | IR 정의 3종 공존: 헤드라인 = 일간 액티브차 기하 복리 **1.7596**(metrics 일치) / 서브기간·E1 3분할·DSR = 산술 **1.7233** / CAGR 차 2.035. 기하−산술 +0.036. P1~P4(산술) 1.523/1.582/2.028/1.859 | 정의 명시 + E1 바(0.36 SE)가 어느 정의에서 유도됐는지 기록 |
| T-11 | low | `scripts/eval_s16_8a_arm.py:162` | e1_summary.json 스키마가 스크립트 출력과 불일치(3분할 델타 코드 부재) | 재생산 가능 스크립트로 정합 |
| T-13~T-18 | low · 미검증 | streamlit/export/validate/model_trainer/config/CLAUDE.md | 레지스트리-번들 가드레일 불일치 · sp500 sub_period_ir NaN(휴장 77행) · HOLD TE 축 무발동 · 첫 슬롯 퇴화 채택·events 오기록 · `__post_init__` 검증 누락 · realized_beta 전표본 vs 문서 252d | 문서/ops 항목 |

**반박·기각**: T-06(캡 루프 래칫 → 북 붕괴 — 하드 제약으로 유계, 저위험 코드 품질) · T-08(JP T+1 스탬프 — 선행 백필 아티팩트) · M3(위 P6). **메인 스스로 무효화**: 첫 TG 프로브의 SNDK/285A/GEV 비율 이상은 원시 'date' 열 미설정(epoch-ns 인덱스) 정렬 오류(§S16 D군 `data_loader.py:588` 동형 함정); 산술 252 연율화 IR 1.723 vs 1.7596 은 정의 차이(T-12 로 확정); 워밍업 all-zero 행은 targets 첫 272일 NaN 이라 학습 미유입.

---

## 성과 개선 후보 (5관점 24건 → 심사 통과 14건, W3 반영 후 재정렬)

채택 근거별 정렬. '사전확률'은 심사자가 결정 로그 전례로 매긴 값. 사전점검은 전부 백테스트 0·인벤토리 비계수이며 **사전등록(측정 전 단독 커밋)·사용자 결정 후에만 실행**한다.

### Tier 0 — 정확성 수정 (채택 기준 = 정확성, 전례 5/5)

| ID | 후보 | 단일 파라미터 | 결정 게이트 | 상태 |
|---|---|---|---|---|
| **G1-01a** | **T-01** 커버리지 갭 임퓨트 NaN 화(코드만) | `per_share_input_contract_enabled` 하위 (a) | P4 갭창 tg_upside CS std ≥0.90 복원 · E0 parity · E1 정확성 4항목 · E2 do-no-harm | **즉시 사전등록 가능** |
| **G1-01b** | **M1** 명목가 분모 — tg_upside·cash_conversion_z·PE/PB/PEG 역산 | `nominal_price_source: PX_LAST_UNADJ` | 데이터 게이트: 무배당 5종 바이트 동일·MO 2014-06-30≈41.94·TG/PX 2.0→1.0~1.25·z 격차 1.56→<0.5 | **데이터 대기** — 사용자 Bloomberg 재인출(별도 시트) 필요, PE×EPS 프록시 무효 |
| G1-02 | 자본 실질성 가드(음자본 ROE/PB NaN) | `book_equity_guard_enabled` | — | **SHELVE**(M3 반박: 예측·비중 피해 없음, 대안 IC t 0.63) |
| G1-03 | 비동기 보정(피처): beta_63d·idio_vol_63d Dimson | `market_model_dimson_lags: 1` | G1 아시아 주간/일별 idio 비 ≥1.15 & G2 지역 갭 ≤0.35σ | 진행 가능하나 **가치 하향**(idio +4~5%, β gain 2.8%) — G5-01 뒤 |

### Tier 1 — 리스크 Σ·거버넌스 (알파 비트 불변, §S13.41/§S16.7 프레임)

| ID | 후보 | 단일 파라미터 | 결정 게이트 | 심사 |
|---|---|---|---|---|
| **G5-02** | 메가캡 변동성 수축 OFF(21종·bm 51%) | `cov_megacap_vol_shrink_enabled: false`(기존) | P0 excess-QLIKE(OFF) ≤0.95×(ON) & 3분할 & 고/저변동 기전 | **p1** · 5분·코드 0 |
| **G5-01** | 비동기 보정(Σ): 상관행렬만 5일 겹침(또는 lag-1 NW) | `cov_corr_overlap_enabled`(K=5) | **P0 는 W3 로 이미 충족(3.1×, 97/97)** → P1 결정: MZ 회귀 β_d CI 하한 >1 & \|β_w−1\|<\|β_d−1\|/2; P3 캐릭터 | p2 · 리스크 규율 프레임(DSR +1), IR 기대 0 |
| G3-04 | 섹터 리스크 몫 캡 0.85 강제(§S16.7 루프 그룹 확장) | `sector_risk_share_cap_enabled` | **P0 코드 작성 전 단독 측정**: 최근 24회 위반율 ≥50%(§S11.5) | p2 · T-03 tol 정합 동반 |
| G5-03 | HAR 단기 항 log σ_trail21 → 모델 B(T2-5) | `option_vol_cov_har_enabled` | §S13.42 P0′ 동일 바(QLIKE ≥5% & MAE ≥5% & 3분할) + L1 ≥0.005 | p3 |
| G2-03 | 투영 infeasible 폴백 bm→prev(T-07) | `projection_fallback_mode: prev`(기존) | 강제 L1 헤드룸(실측 최대 0.072) + 교착 분석 후 거버넌스 | needs_data |
| G5-05/G2-04 | √-law 비용 capacity 게이트(T2-8) | k=1 | 채택 게이트 없음 — 사용자 AUM 입력 시 5분 진단 | governance |

### Tier 2 — IR 바(ΔIR>0.36 & 3분할) 후보 (사전확률 0.15~0.35)

| ID | 후보 | 단일 파라미터 | 결정 게이트 | 심사 |
|---|---|---|---|---|
| G4-01 | 순수 63BD 라벨(보유지평 ~144BD 정합) | `forward_horizon: 63`(embargo 자동) | G-B IC(spec63)−IC(spec20)>0 3/3 · G-D 3슬롯 paired ΔIC ≥0 & NW t≥2 | p1 · 사전확률 ~0.2 |
| G3-01 | top-20 UW 보호(랭커 인증 구간 역베팅 금지) | `protect_top_k_from_uw_enabled`(k=rank_eval_at 20) | P2 반사실 ΔR ≥+0.50%p/yr & 3/3 & 49/49 optimal; P3 캐릭터 | p1 · 성공해도 +0.13~0.27 |
| G2-01≡G3-02 | 실행층 confidence 변조 제거(eta 0.5·band 0.003 고정) | `static_execution_enabled` | P1 순 ΔR ≥+0.50%p/yr & 3/3, turnover ≤1.25× | p2/3 · §S13.11 방향 위험 |
| G2-02 | 게이트 강제거래를 투영에서 eta 로 평활 | `gate_bounds_target_only` | P0 median forced_L1 ≥0.005(실측 q50 0.0068 통과) → P1/P2 | p3 |
| G4-04 | UW 측 μ 평탄화(T2-7) | `uw_mu_flatten_enabled` | μ≤0 내 IC t<2 & μ>0 내 t≥2 & 바닥 UW 기여 비양 | p2 · 약 |
| G4-02 | 시간 감쇠 표본 가중 | `sample_weight_halflife`(age-IC 곡선 도출) | 표류 b<0 NW t≤−2 & 버킷 IC | p3 · SHELVE 최빈 |
| G5-04 | 어닝 이벤트 분산 더미 → 모델 B(T2-6) | `option_vol_cov_earn_var_enabled` | 규칙 정밀도 ≥0.7·재현율 ≥0.6 → P0″ | p5 |

### 종결·기각(재도전 금지 근거)
NW lag-1 교차공분산 '총량' Σ(§S13.46 종결축 — 단 G5-01 블록 보정은 별개로 심사 통과) · 죽은 피처 9 제거 단독 arm(정보 이득 0, §S12.3(i)) · 액티브셰어 캡 0.45→0.50(production 핀+tighten-only 로 무음 no-op·상금 상한 바 아래) · 확장 학습창(§S16.3 E2) · 재훈련 빈티지 앙상블(§S7 A4) · T2-2 선형 top-k ES(eval 지표는 궤적 불변) · T2-3 쿼리 스트라이드 · pca_n_remove 1(REDESIGN L) · subsample_freq · 적응 eta(§S13.45-A) · band 변경(REDESIGN O-d) · 리밸 중간 트리거(§S13.40) · 음자본 가드(M3 반박).

---

## 제안하는 실행 순서

1. **사용자 결정 필요 — 워크북 재인출**: `PX_LAST_UNADJ`(Bloomberg DPDF 배당 조정 OFF) 별도 시트. 이것 없이는 M1 수정 불가. 인출 후 새 빈티지 쌍에서 S0′ 재인증 → G1-01b 사전등록.
2. **코드만으로 되는 것 먼저**: G1-01a(T-01) 사전등록·측정. 산출물 불변 O-트랙 즉시 적용: T-03 가드 tol 정합 + §S16.7 절 정정, T-04 빈티지 지문 시점, T-02 스크립트 폴백 게이트, T-05 캡 집계, T-12 IR 정의 명시, M4/T-10 whitelist 주석·가드 테스트, §S15 #2 no-op 기록.
3. **5분 사전점검 2건(코드 0, 사전등록 커밋 후)**: G5-02 메가캡 수축 OFF QLIKE, G3-04 P0 섹터 위반율.
4. G5-01(Σ 비동기, P1 게이트) → G1-03(피처) 순. Tier 2 는 새 기준선 확정 후 G4-01·G3-01.

---


---

## 완결성 비평 (W3 비평가, 읽기 전용) — 이번 라운드가 검정하지 않은 것

전문 `outputs/s17_prechecks/review_r4_w3_critic.md`. 비평가가 최우선으로 꼽은 누락 3건과 그 검정 설계는 다음 라운드의 첫 항목이다. 전부 **미검정 가설**이며 결함 확정이 아니다.

1. **M3 SHELVE 판정이 production 직접 소비처를 빠뜨렸다** — `src/backtest.py:527 apply_vol_quality_tilt`(production ON)는 `best_roe_level_z` 를 1/99 winsor z 로 가중치에 직접 곱한다. 검정: 97 리밸 상위 vol 터실 내 CL/DELL/HD/SPGI/MA(상단)·MCD/PM/HCA/SBUX/VRSN(하단)의 위치와 `lam·sd·zq` 누적 기여, 부호 인식 ROE 로 재계산 시 비중 델타(§S13.50 오프라인 재현 자산 재사용). 결과에 따라 M3 는 '틸트 채널 한정 결함'으로 재분류될 수 있다.
2. **T-01 의 `_fill_missing` 횡단면 median 채움은 전 시트 공통이다** — BEST_EPS/SALES/CAPEX/FCF(현지통화 절대액 → 타통화 median), **CUR_MKT_CAP(벤치마크 가중치·mega funding 입력)**, EQY_REC_CONS, Factset revision, 정의 불가 NaN 의 PE/PEG/PB/EV_EBITDA 의 상장 후 갭도 같은 경로. 검정: 시트×티커 원시 post-listing NaN 셀 수, 채움값/티커 자기 중앙값 |log 비|>1 셀 수, CUR_MKT_CAP 갭이 bm 가중치에 준 영향(리밸일 교차). T-01 수정 팩의 범위가 TG 시트를 넘을 수 있다.
3. **스핀오프·특별배당 스텝 조정 미검정(M1 보다 심각할 수 있음)** — DELL 297·APP 118 클립 행(T-01 배치 부수 관측)은 M1 배당 드리프트로 설명되지 않는다(DELL 2022 이전 무배당·APP 무배당). 검정: `log(CUR_MKT_CAP/PX_LAST)` 음수 스텝 >3%(특별배당)/>10%(스핀오프: DELL/VMW 2021-11, T/WBD 2022-04, PFE/VTRS, MRK/OGN, IBM/KD, DHR/VLTO, SIE/ENR, NOVN/SDZ, 6758/SFG 2025-10) 스캔 → 같은 날 TG/PX 점프와 `Daily_Returns` 값 확인. 조정돼 있으면 ≈0, 아니면 −30~−50% 가짜 폭락이 모멘텀·vol·Σ·타깃까지 오염.

그 밖의 미검정 항목(요약): M1 빈티지 불안정 실측(08-25 vs 09-01 워크북 비율 비교), C 프록시의 팩터 수익 교란·바이백 오염 분리, 실제 룩어헤드 채널(라벨 21BD 내 배당락 더미) 정량, `_check_target_price_unit_ratio` 의 구조적 탐지 불가; M2 PCA 타깃 오염(아시아 21d specific return 을 US lag0/lag1 에 회귀), 5dov Σ 로 97 리밸 재해 시 아시아 OW 변화(인과), USDJPY 공통요인의 idio 잔류, 현지 휴장 0행의 corr 희석 몫, `estimate_covariance` LW 분기 도달 여부(창 NaN 유무), **EUROPE 43종 블록(1.27×, bm 비중은 아시아의 3~4배) 미정량**; M4 EWMA sqrt 스케일링 inert 실측(재학습 1회 필요), `best_gross_margin_chg_*` 가 죽은 피처인지(op_leverage≈oper_margin_chg 0.977 의 함의), identity-like 피처의 IC 를 between/within 으로 분해; 운영 — feature_attribution/expected_rebalance 가 고배당 12종의 tg_upside 를 '목표가 하방'으로 오표시하는지, 대시보드 "Low-vol" 스타일 노출이 아시아 OW 를 저베타로 오표시하는지, "Quality" 그룹의 best_roe/fin_roe 이중 계상, 과거 tg_* 파생 arm 판정의 오염 피처 기준 표기.

## 부록 — 산출물
- 결정 로그 §S17(정본). 감사 팩·프로브·워크플로우 결과: `outputs/s17_prechecks/` (feature_audit.csv · ticker_median.csv · pinned_tickers.json · zero_gain_by_model.json · high_corr_pairs.json · negative_equity_probe.csv · async_probe.csv · tg_ratio_probe.json · px_adjustment_probe.json · raw_probe.json · *_summary.json · review_r4_w1.json · review_r4_w2.json · review_r4_w3.json).
- 에이전트: W1 16(6 렌즈·1 트리아지·9 검증) · W2 10(5 생성·5 심사) · W3 6(4 리드·1 배치·1 비평). 도구 호출 1,143. 백테스트 0.
