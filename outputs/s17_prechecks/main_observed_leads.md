# 메인 직접 관측 리드 (2026-09-02, 독립 검증 전 — 채택 아님)

아래는 메인 모델이 `outputs/s16_7_name_risk_cap/backtest_result.pkl`(S0′ 인증 런)과 워크북 원시 시트(7장)로
직접 재현한 관찰이다. **독립 검증자가 반박 우선으로 재판정하기 전까지는 "리드"이며 확정 결함이 아니다.**
감사 팩: `<SP>/w3/pack/` (feature_audit.csv, ticker_median.csv, pinned_tickers.json, high_corr_pairs.json,
zero_gain_by_model.json, models_summary.json, negative_equity_probe.csv, async_probe.csv, tg_ratio_probe.json,
px_adjustment_probe.json, raw_probe.json, series_summary.json, predictions_summary.json).

## M1 — PX_LAST가 총수익(배당 재투자) 조정 가격 → `tg_upside`가 미래 배당 누적분만큼 부풀려짐 [Critical 후보 / production / always-on]
- 증거 1: 워크북 PX_LAST 2014-06-30 vs 알려진 명목 종가(분할 조정): MO 19.56 vs 41.94 (0.466) · T 13.18 vs 35.36 (0.373) ·
  VZ 25.80 vs 48.93 (0.527) · PFE 16.53 vs 29.68 (0.557) · XOM 60.72 vs 100.68 (0.603) · KO 0.688 · JPM 0.723 · PG 0.708 ·
  MSFT 0.837 · AAPL 0.876. 배당수익률이 높을수록 비율이 낮다 = 배당 재투자 소급 조정.
- 증거 2: Daily_Returns == PX_LAST.pct_change() 가 4,625일 전부 정확 일치(max diff 0.00000, MO/KO/T/AAPL/TSLA) → 수익률은
  총수익(정상), 가격도 총수익 기준.
- 증거 3: 날짜 정렬 후 median(TG/PX) 연도별 — 고배당 12종(MO,T,VZ,PFE,KO,XOM,TTE,BNP,RIO,SHEL,ENEL,IBM):
  2014 1.91 → 2018 1.65 → 2022 1.46 → 2026 1.09. 무배당 10종(TSLA,AMZN,GOOGL,META,NFLX,NVDA,CRM,ADBE,PLTR,ISRG):
  1.12 → 1.10 → 1.31 → 1.32 (평탄). MO 연도별 2.13→1.00, RIO 2.85→1.02, TSLA ~1.0 평탄.
- 기전: `src/features/sellside.py:368` `upside = tg / local_prices - 1` — TG는 명목(FactSet, 분할만 조정), local_prices는
  배당 조정. t 시점 값에 t 이후 지급 배당이 들어감(룩어헤드) + 인출 시점으로 갈수록 0에 수렴(라이브 시점 분포 이동) +
  리프레시마다 전 이력이 재조정(빈티지 불안정).
- 피해 범위: `tg_upside`(코어 65, 사전 기록상 제거 시 IR −0.48인 핵심 피처), `cash_conversion_z`(§S16.1 P3 수정이
  `shares = market_cap / prices`로 주식수를 유도 — 조정가격이라 고배당주 과거 주식수 과대), 휴면 `fwd_opcf_yield`.
  `_check_target_price_unit_ratio`([0.2,5] 가드)는 비율 1~2라 탐지 불가. 모멘텀·dist_52w_high 등 가격 비율 피처는
  양변 조정이라 정합(총수익 모멘텀 = 표준).
- 정정 경로: 명목 가격 열 필요 — (a) 워크북 재인출(PX_LAST 배당 미조정), (b) 프록시 nominal_px ≈ BEST_PE_RATIO × BEST_EPS
  (둘 다 워크북에 있고 Bloomberg가 명목가로 계산) — 검증·정량화에 즉시 사용 가능.
- 검증자가 할 일: (i) 위 증거 재현, (ii) 프록시 명목가로 tg_upside′ 를 만들어 현행 tg_upside 와의 횡단면 상관·IC(pkl targets) 비교,
  (iii) 고배당 종목의 tg_upside z 시계열이 2014→2026 단조 하락하는지, (iv) 라이브 시점(2026-08) 고배당주의 z가 학습기(2014~2019)
  대비 얼마나 낮아졌는지.

## M2 — 시차 비동기 거래: 아시아 16종의 beta·idio_vol·Σ 왜곡 [High 후보 / production]
- 증거(Daily_Returns 원시, US 191종 EW 대비): ASIA 16종 중앙값 동시 corr 0.125 < 전일 US corr 0.220, β0 0.211 vs β(lag1) 0.376,
  Dimson 비율(β0+β±1)/β0 = 3.12. EUROPE 43종: 0.350 / 0.065 / 1.13. US: 0.571 / −0.045 / 0.82.
  지역 EW: ASIA corr(same-day) 0.179 vs corr(ASIA_t, US_{t-1}) 0.328.
- 패널 실측: beta_63d 티커 중앙값 NUMERIC(KR/JP 15종) −1.16 vs 나머지 −0.02; idio_vol_63d +0.44 vs −0.25 (하위 8종 전부 JP/KR).
- 기전: `src/features/price.py:167-180` beta_63d = 동시 rolling cov/var vs EW 시장(전 250종 평균) → 아시아는 β 과소·잔차 과대;
  idio_vol_63d(모델 gain 1~2위, share 13~47%)가 아시아 종목을 구조적으로 "고 idio vol"로 분류 → vol-quality 틸트 상위 터실 편입 및
  vol 선호와 상호작용. Σ(126d pairwise, `portfolio_optimizer.estimate_covariance`)의 아시아–미국 공분산 ~1/3 과소 →
  아시아 액티브 포지션의 ex-ante TE 과소. PCA specific-return 타깃도 아시아 공통요인 미제거(lagged US 요소가 "specific"에 잔류).
- 검증자가 할 일: (i) 재현, (ii) 아시아 종목의 idio_vol_63d 과대 정도(Dimson 보정 β로 재계산한 잔차 vol 대비), (iii) 리밸일
  Σ에서 아시아×미국 블록 상관 vs 주간 수익률 상관, (iv) 아시아 종목 액티브 비중의 ex-ante vs 실현 TE 기여.

## M3 — 음(−)/0 근처 자본 종목의 ROE·P/B가 극단값으로 소비됨 [High/Medium 후보 / production]
- 증거(BEST_ROE %단위, BEST_PX_BPS_RATIO): CL 중앙 ROE 390%(|ROE|>100 비율 99.98%) · DELL 202% · HD 138% · SPGI 133% · MA 130% ·
  ABBV 117% · AAPL 112% · LMT 100% / 음수: MCD −110%(77% 날짜) · PM −80%(100%, P/B 전부 NaN) · MSCI −60%(P/B 중앙 3,054) ·
  VRSN −51% · HCA −49%(P/B 51) · SBUX −41%(P/B 77) · RDDT −53% · MSI(P/B 67) · MAR · LOW(P/B 80) · ORLY(P/B 74).
- 패널 실측: best_roe_level_z(= `cross_sectional_zscore(raw)`, accounting.py) CL 중앙 +4.32 12년 고정, MCD/HCA/MSCI/VRSN/PM 하위;
  티커 중앙값 std 0.397 → 극단 10종 제외 시 0.174 (IQR 0.168) — 소수 극단이 횡단면 std를 지배. fin_roe_pb_gap(=cs_rank(roe)−cs_rank(pb))
  HCA −3.34 · MCD −3.21 · MSCI −3.16 · SBUX −2.99 · VRSN −2.98 · MSI −2.97 · LYV −2.30 · MAR −2.04 12년 고정.
- 기전: 자사주 매입으로 자본이 음/0 근처인 고ROIC 프랜차이즈가 "최악 품질·최고가"로 고정 라벨 → 랭커가 종목 더미로 학습(§S16 P1 동형).
- 검증자가 할 일: (i) 재현, (ii) 해당 종목들의 모델 예측 평균·활성 비중 부호(구조적 UW인지), (iii) rank/winsor 대안 시 횡단면 분포.

## M4 — 코어 65 피처 중 9개가 정보 0 [Medium / 구조]
- 7개 브로드캐스트 피처(cal_is_Q1, regime_mkt_ret_21d, fac_F_Quality_mom_63d, fac_F_Growth_mom_63d, fac_F_Value_mom_63d,
  fac_value_growth_63d, fac_yield_slope): 33개 모델 전부에서 split 0·gain 0(`zero_gain_by_model.json`) — 날짜 그룹 rank 목적함수에서
  쿼리 내 상수라 그래디언트 합 0(§S13.18 기전과 동일). → §S15 fix-pack #2(fac_* 캘린더 수정)는 모델에 no-op였음.
- 2개 완전 중복: fin_roe_level_z ≡ best_roe_level_z, fin_pb_level_z ≡ best_px_bps_ratio_level_z (상관 1.000, 코드상 동일 식).
- EWMA 드롭 예산 int(65×0.05)=3이 전부 죽은 bcast 피처에 소진(62피처 모델에서도 zero-gain 4 잔존) → 피처 선택층 사실상 inert.
- 근접 중복: oper_margin_chg_63d~op_leverage_63d 0.975, best_eps_chg_252d~earnings_quality_252d 0.934.

## M5 — capex_intensity_z 극단 고정 [Low]
- NEE −5.0(클립 고정) · C −4.16 · UBSG −3.30 · SO −3.18 · TSM −2.83 · MU −2.79 · EQIX −2.70 (12년 중앙값). 헤비테일 비율의
  횡단면 z → 섹터 더미화. 설계 관찰(rank 변환 대안).

## 무효화된 리드 (기록)
- 첫 TG 프로브의 SNDK 0.039 / 285A 0.058 / GEV 0.46 비율은 **메인의 인덱스 정렬 오류**(원시 'date' 열을 set_index 하지 않아
  epoch-ns 인덱스로 행 번호 정렬)였음. 재정렬 후 SNDK 1.15~1.21, 285A 0.99~1.30, GEV 1.03~1.13으로 정상. 검증자 주의: 원시 시트는
  `df.set_index(pd.to_datetime(df['date']))` 후 사용(§S16 D군 data_loader.py:588 동형 함정).
- IR 재계산 1.723(단순 252 연율화) vs metrics 1.7596: compute_metrics가 기하 연율화(compute_performance_metrics)라 정의 차이. 결함 아님.
- 워밍업 252일 all-zero 행: targets가 첫 272일 NaN이라 학습에 미유입. 결함 아님.
