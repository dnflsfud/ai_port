**레포 루트**: `C:/Users/westl/PycharmProjects/pythonProject/venv_vf_new/machine/re_study/c2/ai_port` (아래 경로는 루트 기준). 읽기 전용·pkl 미로드·레포 무변경. 최우선 누락 3건 = **(b5) M3 refuted 판정이 production 직접 소비처 `apply_vol_quality_tilt`를 빠뜨림**, **(b7/c4) T-01의 `_fill_missing` 횡단면 median 채움이 전 시트 공통(정의 불가 NaN 밸류에이션·현지통화 절대액·CUR_MKT_CAP 포함)**, **(a5/c1) 스핀오프·특별배당 스텝 조정(M1 드리프트로 설명 안 되는 DELL 297·APP 118 클립)**.

## (a) 검정하지 않은 대안 가설

- **a1 [M1] 빈티지 불안정 미측정**: 08-25·09-01 두 워크북의 `PX_LAST[2014-06-30]` 종목별 비율을 비교 — 그 주 배당락 종목만 (1−D/P)만큼 달라지고 무배당·`Daily_Returns`는 비트 동일이어야 함(총수익 불변 증명 겸용).
- **a2 [M1] C의 IC 음(−0.026)이 "신호 소거"가 아니라 2015~20 고배당 언더퍼폼(팩터 수익) 반영일 가능성**: 연도별 IC(now) vs IC(hat) 표 + C를 배당수익률 rank·섹터로 잔차화한 뒤 재산출 — 차이가 고배당 언더퍼폼 연도에만 집중되면 레짐 교란, 전 연도 균일하면 구조적 소거.
- **a3 [M1] C 프록시 바이백 오염(AAPL 0.44 log)**: 바이백 미미 서브샘플(SO·DTE·ENEL·IBE·T·VZ 등)에서만 IC now/hat 재산출; 또는 mktcap 양수 스텝(증자)과 음수 스텝(배당) 분리 누적.
- **a4 [M1] 실제 룩어헤드 채널(라벨 21BD 안 배당락)은 미정량**: "향후 21BD 내 배당락" 더미의 21d 타깃 IC와 해당 셀의 tg_upside 잔차 IC — 누적 C가 아니라 이 항이 성과 부풀림의 유일한 기전.
- **a5 [M1] TG "명목" 검증이 분할일뿐 — 스핀오프(DELL/VMW 2021-11, T/WBD 2022-04, PFE/VTRS, MRK/OGN, IBM/KD, DHR/VLTO, SIE/ENR, NOVN/SDZ, 6758/SFG 2025-10)·특별배당(COST 2020/2024, CME 연례, PGR, RIO) 스텝 미검정**: `log(CUR_MKT_CAP/PX_LAST)` 음수 스텝 >3%(특별)/>10%(스핀) 스캔 → 같은 날 TG/PX 점프와 `Daily_Returns` 값 확인(조정돼 있으면 ≈0, 아니면 −30~−50% 가짜 폭락 → 모멘텀·vol·Σ·타깃 전부 오염이라 M1보다 심각).
- **a6 [M2] PCA 타깃 오염(G) 미검정**: 아시아 16종 21d specific-return 타깃을 US EW lag0/lag1(형성창 마지막 날) 수익률에 회귀(t-stat) + `reversal_5d`·`rel_mom_21d`와의 상관이 US 대비 큰지.
- **a7 [M2] "Σ 과소 → 아시아 구조적 OW +1.5%"의 인과 미검정**: `W1_recon.py` 하네스로 97 리밸을 5dov(또는 NW L=1) Σ로 재해 → 아시아 순액티브·285A/6857 OW 변화량.
- **a8 [M2] 아시아 고 idio의 80%를 "낮은 R²+USD 총변동성"에 귀속 — USDJPY 공통요인이 idio에 잔류하는 대안 미분리**: JP 수익률을 [US EW, USDJPY] 2요인 회귀한 잔차 vol vs `idio_vol_63d`; 로컬통화 vs USD idio_vol 비율.
- **a9 [M2] 현지 휴장 0행(JP 연 16~20행)의 corr·β 희석 미분리**: 0행 제외 재계산으로 Dimson 비율 3.82 중 휴장 몫 산출.
- **a10 [M2] `estimate_covariance`(src/portfolio_optimizer.py:88-98)는 창에 NaN이 없으면 LedoitWolf 분기 — 검증자는 pairwise 가정**: 97 리밸 창별 NaN 유무로 분기 확인(COF 2025-05-19 상장 +126d 이후 LW 가능), LW 창의 블록 상관 수축 크기.
- **a11 [M3] "산출 피해 미증명" 결론이 tilt 채널을 빠뜨림**(b5 참조): 97 리밸 상위 vol 터실 내 `zq`에서 CL/DELL/HD/SPGI/MA(winsor 상단)·MCD/PM/HCA/SBUX/VRSN(하단) 위치, `lam·sd·zq` 종목별 누적, 부호 인식 ROE로 재계산 시 델타 변화를 `W1_recon`으로 비중까지 전달.
- **a12 [M3] 음자본 P/B NaN(PM 100%)·손실기업 PE/PEG NaN이 `_fill_missing`(src/data_loader.py:443-451) 횡단면 median으로 "시장 중앙값 밸류에이션"이 됨**: 시트×티커 원시 NaN 비율, 처리 후 값==row median 셀 비율, `rolling_tsz`가 median 경로를 자기 이력으로 쓰는 셀 수.
- **a13 [M4] EWMA sqrt 가중 스케일링 inert 주장 미실측**: 재훈련 1창(2019-02-20 모델)만 가중치 1.0 vs 실제로 재현해 예측 max|diff|(재학습 1회 사전 승인 필요).
- **a14 [M4] op_leverage≈oper_margin_chg(0.977)는 gross-margin 변화≈0을 뜻함 → 코어 `best_gross_margin_chg_63d/252d`가 죽은 피처인지 미검정**: BEST_GROSS_MARGIN 원시 티커별 연간 고유값 수·pct_change==0 비율(feature_audit: 티커 std<0.15가 69/84).
- **a15 [M3/M4] identity-like 피처의 IC 원천 미분해**: 65 코어 각각 raw IC vs 티커-demeaned IC — 사이분산 점유 ≥0.5(capex_intensity 0.83, fin_roe_pe_gap 0.66, realized_vol_126d 0.65, fin_roe_pb_gap 0.61, idio_vol 0.57, analyst_rec 0.57, beta 0.55)에서 IC가 between 성분에서만 나오면 "지속 틸트=종목 더미".

## (b) 확정 결함의 파급 범위 중 미정량 소비처 (grep 결과)

- **b1 [M1] `local_prices`/`data.prices` 소비처 전수**: src/features/sellside.py:368(tg_upside 계열 11개 → 코어 `tg_upside`), :431(`fwd_opcf_yield` 휴면), src/backtest.py:361-386(growth_tilt `rev_tg_share=0` 휴면), src/features/accounting.py:160(shares→`cash_conversion_z`), src/data_loader.py:1422(`_check_target_price_unit_ratio` tail 252 — 조정가가 1로 수렴하므로 탐지 불가), src/data_loader.py:1548(`price_implied_returns` 대조 — Daily_Returns가 PX pct_change 파생이라 동어반복 가드), src/features/price.py:109-127·assembly.py:375-384·conditioning.py:104-106(비율이라 정합), volume_flow.py:87(달러거래량 휴면). **미정량**: scripts/export_operating_data.py:1202 `build_feature_attribution`(SHAP → features.json/feature_attribution.json/expected_rebalance.json)이 tg_upside·fin_pe/pb·cash_conversion을 드라이버로 표시 → 검정: 최신 feature_attribution.json에서 고배당 12종의 tg_upside 기여 부호·크기(라이브 z 전부 음수 = 사용자에게 "목표가 하방"으로 오표시).
- **b2 [M1] 역사 arm 판정 재해석 범위**: `rev_tg_share>0`·tg_* 파생 피처를 켰던 과거 arm(variants/·결정 로그) 목록화 후 그 arm의 ΔIR 판정이 오염 피처 기준임을 표기.
- **b3 [M2] `beta_63d`/`idio_vol_63d` 소비처**: 모델(코어), src/backtest.py:520-544 tilt 터실(ON, −0.8종 검증됨), src/features/interactions.py:23-25(OFF), src/config.py:1286 factor_neutral(OFF)·:1333 residual_sleeve(OFF), **scripts/build_dashboard_data.py:61 "Low-vol" 스타일 노출 = realized_vol+beta+idio** → 아시아 OW가 "저베타·고idio"로 오표시 → 검정: 스타일 노출 시계열에서 아시아 슬리브 기여 분리, Dimson β 대체 시 Low-vol 노출 변화.
- **b4 [M2] Σ 소비처 중 EUROPE 43종 미정량**(블록 1.27×, Dimson 1.22, bm 비중은 아시아의 3~4배): `M2_step3iv`를 EU로 반복 — 순액티브·ex-ante 점유 daily vs 5dov·이름캡(portfolio_optimizer.py:680-753) 교차, risk.json(export_operating_data.py:1692-1718) sector_risk 재계산.
- **b5 [M3] `BEST_ROE` 소비처**: accounting.py:82(best_roe_level_z), :174(roe_pe_z 비코어), :194(quality_gated rank), assembly.py:506(growth composite rank), :554/585/589(fin_roe_*), **src/backtest.py:527 `apply_vol_quality_tilt`(production ON, `best_roe_level_z`를 1/99 winsor z로 가중치 직접 조정 — M3 미검정)**, config.py:1285(OFF), scripts/build_dashboard_data.py:43("Quality" 그룹에 best_roe_level_z와 fin_roe_level_z 둘 다 → 동일 열 2중 계상). `BEST_PX_BPS_RATIO` 소비처: accounting.py:104-105, assembly.py:564-566/585, interactions.py:24/26(OFF).
- **b6 [M4] 중복 열의 산출물 소비**: SHAP 기여가 두 열로 분산돼 feature_attribution의 ROE 드라이버 순위 과소 → 검정: best_roe_level_z+fin_roe_level_z 합산 순위 vs 단독 순위.
- **b7 [T-01] `_fill_missing`(data_loader.py:443-451, :668)은 전 시트 공통**: BEST_EPS/BEST_SALES/BEST_CAPEX/FCF(현지통화 절대액 → 타통화 median), **CUR_MKT_CAP(bm 가중치·mega funding 입력)**, EQY_REC_CONS, Factset revision, BEST_PE/PEG/PB/EV_EBITDA(정의 불가 NaN)의 상장 후 갭 → 검정: 시트×티커 원시 post-listing NaN 셀 수, 채움값/티커 자기 중앙값 |log-비|>1 셀 수, CUR_MKT_CAP 갭이 bm 가중치에 준 영향(리밸일 교차).

## (c) 같은 유형 결함이 있을 법한 다른 시트/피처

- **c1 [M1 유형: 명목÷조정가 + 스텝 조정]** DELL 297·APP 118 클립(T-01)은 M1 드리프트로 설명 불가(DELL 2022 이전 무배당·APP 무배당) → 검정: DELL 2021-11-01(VMW 스핀)·2018-12-28 전후 CUR_MKT_CAP/PX 스텝과 TG/PX; APP 2022 클립 행의 원시 TG/PX(진짜 +300% 업사이드면 결함이 아닌 비강건 z 설계 → `log(TG/PX)` 대안 사전등록).
- **c2 [M3 유형: 분모 0 교차]** `safe_pct_change`(src/features/utils.py:17-20, |base| 분모)를 부호 교차 가능한 EPS/FCF/OPER_MARGIN/GROSS_MARGIN/ROE에 적용하는 코어 피처(best_eps_chg_252d, earnings_quality_252d, op_leverage_63d, oper_margin_chg_63d/252d/accel, fin_roe_chg_63d/252d, fin_eps_chg_63d) — feature_audit |z|≥4.99 포화율 0.64~0.81%로 level_z(0.06~0.08%)의 8~10배 → 검정: 포화 셀 중 |base|가 티커 |시트| 10분위 이하 비율, 대안 (x−x₀)/max(|x|,|x₀|)의 IC·포화율.
- **c3** `cash_conversion_z`=FCF/|NI|(NI≈0 교차: KKR 3.95·LYV 2.86 핀)·`capex_intensity_z`(NEE −5 등 9종 핀) → c2와 동일 rank/log 대안 검정.
- **c4 [T-01+M3 결합]** 손실기업(UBER SNOW PLTR RBLX DASH ABNB NET ZS DDOG CRWD RDDT RR/ LYV AIR)의 BEST_PE/PEG NaN→median 채움 → fin_pe_level_z≈0·fin_roe_pe_gap=rank(roe)−0.5 고정 → 검정: b7 + 해당 종목 예측 백분위 vs 실현.
- **c5 [M2 유형: 다른 비동기]** FX 스탬프 비동기 — Index.xlsx FX(NY 종가) × 도쿄/런던 종가 로컬 수익률 → USD 수익률에 t일 FX의 현지 종가 이후 변동이 혼입 → 검정: JP 종목 USD 수익률 vs USDJPY 익일 수익률 교차상관, Factor_PX_LAST FX 스탬프 시각 확인.
- **c6 [T-09 유형]** 현지 휴장 0행이 realized_vol_21d/126d·idio_vol·Σ 대각(JP 연 16~20행 ≈ 분산 7% 희석)·max/min_ret에 유입 → 검정: 시장별 정확-0 수익률 행 비율, 0행 제외 vol 재계산 비율.
- **c7 [T-03 유형: 허용오차 불일치]** `estimated_te_breached`(export:1704, tol 1e-6)·sector 0.85(export:1717만, 옵티마이저 미강제)·validator HOLD 6종(validate_portfolio_bundles.py:667-669) → 검정: 97 리밸의 TE·섹터 몫이 캡 ±tol 경계에 있는 건수, 체크별 강제 위치(optimizer/export/validator) 표.
- **c8 [T-12 유형: 정의 이중화]** E1 판정이 ΔIR(metrics 기하, scripts/eval_s16_7_arm.py:154)+서브기간 부호(src/harness.py:47 산술) 혼합 → 검정: 최근 10 arm ΔIR을 산술로 재계산해 판정 뒤집힘 여부, "+0.36=1 SE"의 정의 기준(두 정의 부트스트랩 SE) 명시.
- **c9 [M4 유형: 근사 상수]** 티커 std<0.15가 60~87종인 마진/ROE 변화 피처(gross_margin_chg 69/84, oper_margin_chg 72/82, fin_roe_chg 78/87, op_leverage 79) — 시트 갱신이 연 1~4회 스텝이라 1/3 종목이 사실상 상수 → 검정: 원시 시트 티커별 연간 고유값 수·ffill 셀 비율, 스텝 직후 21d 구간만의 IC(스텝 타이밍 신호 vs 상수).
- **c10 [M2+M3 결합: 지역 고정효과]** NUMERIC(JP/KR) 중앙값이 beta −1.16·idio +0.44·tg_upside +0.22(vs −0.16)·analyst_rec +0.33(vs +0.10)·realized_vol +0.3 — 피처 결합이 "지역 더미" 형성 → 검정: 시장 suffix η² 피처별, 지역-demeaned IC.

## (d) 검증자 수치 간 모순

- **d1** M3 메타: `refuted=true`인데 `production_impact=changes_production_numbers`·severity low — 판정·메타 불일치, 게다가 a11 미검정이라 "피해 없음" 결론 미완.
- **d2** M3 "168종 std<0.15" vs 팩 feature_audit 157(적격 마스크 차이) — 기준 통일.
- **d3** M1 caveat(7) cash_conversion_z gain 1.8% vs numbers 1.39%; M1 fin_roe_pb_gap 2.4% vs M3 2.25%; best_roe_level_z M3 1.46% vs M4 1.51% — 33모델 평균/19 unique/median 혼용 → 단일 gain 표(unique 19, median+mean 병기) 지정.
- **d4** W1 T-01의 "DELL/RIO/APP은 M1 배당조정" 귀속 — APP 무배당·DELL 무배당기라 설명 불가(c1).
- **d5** M1 severity critical의 근거가 "tg_upside 제거 시 IR −0.48"(assembly.py:239, iter21·65종·pre-causal) 인용인데 M4(6)에서 라이브 모델 tg_upside split 0 → critical은 데이터 정확성·역사 재인증 근거이지 라이브 성과 근거가 아님을 명시(라벨 재조정).
- **d6** M2 Σ 재구성은 pairwise 가정, 코드는 NaN 없으면 LW 분기(a10) — T-03 재구성이 2026 날짜에서 ±0.0016 일치했으므로 pairwise일 가능성 높지만 명시 확인 필요.
- **d7** M2 lead "idio_vol gain 13~47%" vs 검증자 "7.8~46.8%"; lead async_probe(달력일 0.220) vs 검증자(주중 0.283) — 팩 파일 갱신.
- **d8** 리드 "무효화된 리드: IR 1.723 vs 1.7596 결함 아님" vs W1 T-12 "confirmed(병존이 문제)" — 분류 충돌, c8로 해소.
- **d9** W1 T-03 "strict 0.35 breach 12/97 → 결정 로그 정정" vs 결정 로그 7949행은 이미 "breach 0/97, 최악 0.3564(≤0.36)"로 0.36 기준 명시 → 정정 대상은 로그가 아니라 export(:1710)·validator(:668)의 strict 0.35와 로그 정의의 불일치(HOLD 판정 기준)로 재서술.
- **d10** M2 "asia_beta0 0.218(vs US EW)" vs "asia_beta_simple_bt 0.324(vs 250 EW)" 두 시장 정의가 표에 병기 없이 혼재 — 정의 병기.
- **d11** M4 파급 분모 65(코어) vs 62(활성) 혼용, M1 "10/65 영향" — 활성 62 기준으로 재표기.

참조 스크립트(검증자 산출물, 스크래치): `C:/Users/westl/AppData/Local/Temp/claude/C--Users-westl-PycharmProjects-pythonProject-venv-vf-new-machine-re-study-c2-ai-port/0caf960f-9660-4a4b-a91e-809a2f4ab53d/scratchpad/w3/verify/W1_recon.py`(a7·a11 재해 하네스), `.../w3/pack/feature_audit.csv`·`group_median_by_suffix.json`(c2·c9·c10 근거 수치).