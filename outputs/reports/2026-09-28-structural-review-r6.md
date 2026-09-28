# 구조 감사 6차 (r6) — 2026-09-28

읽기 전용 감사. 코드 수정 0 · 백테스트 0 · production 무변경 · 인벤토리 불변. 정본 요약 = 결정 로그 §S22.

- 범위: `src/` 전체(15,999줄) + `run_variant.py` + `scripts/export_operating_data.py` + `scripts/validate_portfolio_bundles.py` + `run_and_upload*.bat`, 그리고 데이터 원천(`price_v4.py` → `create_ai_signal_data.py`) 중 production 피처에 닿는 경로.
- 수행: 독립 감사 에이전트 4개(A 데이터 계층 / B 피처·타깃 / C 모델·백테스트·설정 / D 포트폴리오·리스크·운영) 병렬 + 메인(M, FY2 사전점검 중 발견).
- 합격기준 확인(메인 재실행): Critical/High 프로브 전부 기대 출력 재현, Medium 프로브(A-03·A-04·A-05·B-02·B-03·C-02·D-01~D-05) 재현, 인용 file:line 표본 일치, 에이전트의 저장소 변경 0.
- 프로브: `outputs/s22_audit/{A,B,C,D,M}/`. 실행은 ai_port 루트에서 `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.`
  (C 프로브는 `run_backtest` 가 `./outputs/progress.md` 를 CWD 에 쓰므로 **스크래치 CWD 에서** `PYTHONPATH="<ai_port>;<C 폴더>"` 로 실행).

## 요약

| 등급 | 건수 | ID |
|---|---:|---|
| Critical | 0 | — |
| High | 6 | M-01, B-01, A-01, A-02, D-01, D-02(+C-02) |
| Medium | 10 | A-03, A-04, A-05, B-02, B-03, D-03, D-04, D-05, C-01(비prod), D-06(비prod) |
| Low | 14 | A-06, A-07, A-08, B-05, B-06, C-03, C-04, C-05, C-06, D-07, D-08, D-09, D-10, M-02 |

철회: B-04(`slope.diff(63)` 롤 계단) — M-01 때문에 현재 패널에는 롤이 없어 성립하지 않음. M-01 을 롤링 1FY/2FY 로 고치면 그때 재점검 대상.

룩어헤드(미래 정보 사용)는 발견되지 않았다(A·B·C 모두 명시 확인). 대신 **같은 날 정보(D-04)**, **라벨 오염(B-01)**, **빈티지 의존 이력(M-01)** 이 성과 수치의 해석을 흔든다.

## High

### M-01 — `Fwd_Sales_Slope_1FY2FY` 는 롤링 FY1→FY2 가 아니라 "수집 시점 고정 회계연도" 이력이고, 회계연도 보고 때마다 과거 전체가 다시 쓰인다
- 위치: `venv_vf_new/price_v4.py:405-432`(BDH 에 `BEST_FPERIOD_OVERRIDE="1FY"/"2FY"`), `:582`(`start_date="20140101"` 전체 재수집), `re_study/create_ai_signal_data.py:694-734`(`_mask_backfilled_prefix` 는 평탄 백필 접두만 제거), 소비 `src/features/fwd_sales_slope.py:61-76`(production 4피처: level·chg_63d·nl_fslope_rev_confirm·nl_fslope_growth_confirm).
- 결함: 상대 기간 `1FY` 가 호출 시점에 한 번 해석되어, 시트 전체가 "현재 FY1(예: FY2026)·FY2(FY2027) 추정치의 과거 이력"이 된다.
- 증거 1 (PIT 브래킷 검정, `M/pit_bf_between_v2.py`): 롤링 PIT 라면 블렌디드 선행(1BF)은 [1FY, 2FY] 안에 있어야 한다. 슬로프 유효 셀 기준 포함 비율 2019 2.8% · 2020 1.5% · 2021 3.6% · 2022 6.4% · 2023 5.5% · 2024 7.2% · **2025 34.5% · 2026 95.9%**. 중앙값 |log(BF/1FY)| 2016 0.47 → 2020 0.35 → 2024 0.10 → 2026 0.02(고정 미래 연도와의 거리 축소 패턴).
- 증거 2 (빈티지 재작성, `M/slope_vintage_zs.py`): 7월 결산 ZS 가 9월 초 연간 실적을 보고한 뒤, 09-03 번들(s17_2) → 09-11 번들(s18_7) 사이에 ZS 슬로프 z 가 2021–2026 **모든 날짜에서** 바뀜(|Δz| 중앙값 2021 0.83 · 2022 0.41 · 2024 0.20). 대조군 AAPL·MSFT 는 0.0003–0.01(재정규화 잔향). 250종 중 이 구간 변화 종목은 ZS 1개(`M/slope_vintage_rewrite.py`).
- 영향: (1) 학습 이력의 피처 의미가 표류 — 2018–2020 은 5–8년 앞 장기 추정 성장률, 2025–26 은 1–2년 성장률. live 값은 근기 성장률이므로 학습-운용 정의 불일치. (2) 백테스트 재현성 — 결산 보고가 있을 때마다 그 종목의 과거 전체가 바뀌어 S0′ 가 무음으로 포크. (3) §S13.25 채택 근거(ΔIR +0.266)는 이 정의 위에서 측정됨.
- 룩어헤드 여부: 값 자체는 각 날짜의 as-of 추정치로 보이며(미래 실적 사용 증거 없음) 미래 정보 누수로 판정하지 않는다. 확정하려면 원천에서 롤링 기간을 명시적으로 받아야 한다.
- 수정 방향(사용자 결정): 원천을 롤링 기간으로 재수집(종목별 결산월로 연도 고정 오버라이드를 날짜별로 이어붙이기, 또는 FactSet `FE_ESTIMATE(SALES, MEAN, ANN_ROLL, +1/+2)`) → 슬로프 재생성 → 정확성 flip 으로 S0′ 재인증. 임시로는 production 에서 슬로프 4피처를 끄는 롤백(`fwd_sales_slope_features_enabled: false`)을 같은 빈티지에서 측정해 비교.

### B-01 — PCA 잔차 타깃이 일간 평균으로 20일 수익률을 중심화해 라벨에 반-모멘텀 항이 섞인다
- 위치: `src/target_engine.py:124`(`pca.fit` 일간 수익률), `:139` `pca.transform(fwd_fit)`(20일 수익률에서 일간 `mean_` 차감), `:145` `inverse_transform`, `:151` `spec = fwd_fit - common` → 라벨 = `(I−P2)·fwd − (I−P2)·μ_daily`. OFF 경로 `:278` 동일.
- 증거: `B/probe_target_zero_fwd.py` — 모든 종목 forward 수익률 0 인데 라벨 범위 [−0.00528, +0.00453], corr(라벨, 과거 252일 평균) = −1.000. `B/probe_target_mean.py`(모멘텀 없는 i.i.d.) — 날짜별 IC 차이 평균 −0.0128(100% 날짜 음), sd(offset)/sd(target) 0.014.
- 영향: production avg_ic 0.0126 과 같은 크기의 가짜 반-모멘텀 신호가 라벨·조기종료(ndcg)·avg_ic·모든 잔차 IC 사전점검에 들어간다. IR 영향 미측정.
- 수정(default-OFF 플래그 + §8 측정): `C = pca.components_[:k]; common = (fwd_fit @ C.T) @ C`, 또는 `horizon * pca.mean_` 로 중심화.

### A-01 — FX 신선도 가드가 원리적으로 발화하지 않고, 더 신선한 Index.xlsx 환율이 낡은 워크북 사본에 덮인다
- 위치: 원천 `re_study/create_ai_signal_data.py:468` `reindex(bdays, method="ffill").ffill()`; `src/data_loader.py:1069` `factor_series.combine_first(external_series)`(워크북 우선), `:1095`/`:1111` 신선도 계산.
- 증거: `A/probe_a01_fx_staleness.py` — 원 관측이면 `stale>7d=['EUR']` 로 실패해야 할 입력이 워크북 경로에서는 `stale_currencies = [] | max_staleness_days = 0`; 09-14 EUR 에 1.10(낡은 값) 사용, 외부 신선 호가 1.13 무시. production metrics 도 6개 통화 3,193일 staleness 0.
- 수정: 신선도는 채우기 전 외부 호가로 계산하고 `external_series.combine_first(factor_series)`, 또는 원천 FX 열의 `.ffill()` 제거.

### A-02 — TG/가격 기저 가드는 배당조정 PX_LAST 로 나누는데 `tg_upside` 는 PX_LAST_UNADJ 로 나눈다
- 위치: `src/data_loader.py:1558`(`local = self.local_prices`), `:1562`(ratio), 소비 `src/features/sellside.py:441`(`_local_price_panel` = 명목가).
- 증거: `A/probe_a02_tg_guard_denominator.py` — 가드 중앙값 1.15(정상 판정, suspect {}, guard_ok True) vs 소비자 기준 0.69, 사건 전 `tg_upside` −0.31(참값 +0.15), |log| 0.511 > 0.25 인데 미탐지. §S18 P1 3번 클래스(RTX·T 형 Abnormal 스핀오프)를 가드가 볼 수 없다.
- 수정: 가드 분모를 소비자와 같은 `_local_price_panel(data, config)` 로.

### D-01 — 스케줄 작업이 미커밋 작업트리 코드로 production 을 계산하고 그대로 커밋·푸시한다
- 위치: `run_and_upload.bat:35`(작업트리로 production 백테스트), `:51` `git add -A`, `:84` push; `scripts/validate_portfolio_bundles.py:645-729` 에 provenance 검사 없음; `export_operating_data.py:113-114` 는 git_dirty 를 기록만 함.
- 증거: `D/p3_scheduled_sweep.py` — c28ecf9(09-09, src 2파일+테스트+variant)·bf6e055(09-15, §S19 수정 13파일; fixes.md:49,73 은 "커밋·업로드 안 함") 둘 다 `git_dirty=True` 로 PRODUCTION 발행·origin 푸시. `validator checks git_dirty: False`.
- 수정: 단계 [2] 전에 `git status --porcelain` 이 outputs/ 밖에서 비어 있지 않으면 중단(또는 HEAD 의 깨끗한 worktree 에서 실행) + `evaluate_production` 에 `git_dirty is False` 검사.

### D-02 (+C-02) — production-ON 채널이 원천 시트 누락 시 경고 한 줄로 무력화되고 게이트가 모른다
- 위치: `src/backtest.py:2116-2123`·`:2143-2147`(iv30_z·관측 마스크 누락 → warning), `:1986-1993`(Earnings_Timeline 누락 → PEAD SKIPPED print), `src/data_loader.py:1300-1308`(ESSENTIAL_SHEETS 에 iv30_z·Earnings_Timeline·Fwd_Sales_Slope_1FY2FY·PX_LAST_UNADJ 없음), `src/features/fwd_sales_slope.py:54-58` 동일 패턴; 검증기는 `option_vol_cov_scaling_applied` 를 읽지 않음.
- 증거: `D/p5_optvol_silent_disable.py` 4줄(essential False · warning-only True · applied False · validator False), `C/probe_run_backtest_stub.py`(PEAD 무효인데 `data_quality keys flagging the skip: []`).
- 수정: 플래그 ON 인데 시트가 없으면 raise(또는 조건부 essential), HOLD 체크 `applied == enabled`.

## Medium
| ID | prod | file:line | 결함 | 증거(재현) |
|---|---|---|---|---|
| A-03 | 드묾 | data_loader.py 가드 중앙값 | 커버리지 이전 CS-median 대체 TG 셀이 가드 중앙값에 섞여 가짜 suspect + 확인 불가 HOLD | `A/probe_a03` guard 16.50 vs 관측 1.10 |
| A-04 | ops | tg_basis_guard.py:79-103, acknowledge_tg_basis.py:35 | 관측 집합에서 빠진 종목의 pending 이 영구 잔존 → 영구 HOLD | `A/probe_a04` acknowledge ValueError |
| A-05 | 드묾 | data_loader.py:363-367 | BusinessDays 가 PX_LAST 끝을 덮는지 미검사 → 최신 행 무음 절단, tail_ffill 0 | `A/probe_a05` 09-14 → 09-10 |
| B-02 | 예 | sellside.py:106-113, :163 | "완만 하락" 마스크가 하락에만·12개월 중 8개월에·길이 상한 없이 발화 → 진짜 하향을 동결 | `B/probe_gradual_mask` 2월 하락 2BD 지연·상향 통과 (prod EPS 12,828·Sales 8,633셀) |
| B-03 | 예 | assembly.py:837-853, utils.py:17-24 | winsor 전 z-score → 이상치 1개가 나머지 분산을 날마다 다르게 압축 | `B/probe_zscore_compression` p90-p10 2.30 → 0.15 |
| D-03 | 드묾 | portfolio_optimizer.py:886-912, validate_portfolio_bundles.py:680-712 | 벤치마크 폴백 북(회전율 0.60 > 캡 0.15)도 PRODUCTION | `D/p2` |
| D-04 | 예 | backtest.py:2177-2180, option_vol_cov.py:78,100-102 | option-vol Σ 스케일이 t 종가(iv30_z[t]·r[t]) 사용 — 나머지 입력은 t−1 | `D/p1` SAME-BAR |
| D-05 | ops | run_and_upload.bat:20-44 | `if errorlevel 1` 은 음수 종료코드(네이티브 크래시)를 성공으로 통과 | `D/p4` −1073741819 → CONTINUES |
| C-01 | 아니오 | backtest.py(simulate_portfolio) | tuning 홀드아웃이 예측만 멈추고 P&L 은 예약 구간까지 적산 | `C/probe_tuning_holdout` |
| D-06 | 아니오 | export_operating_data.py:1668 | TE 감사가 조건화 캡이 아닌 정적 3.5% 와 비교 | 정적 |

## Low
A-06 calendar_type 하드코딩(data_loader.py:853) · A-07 BM 수익률 ffill(utils.py:60, 잠재) · A-08 캐시 분기 계약(죽은 분기) · B-05 라벨 t+1 은 실행 지연 1일로 거래 불가 · B-06 sellside docstring · C-03 `one_way_tc` 미복사(backtest.py:1873, KNOWN-OPEN) · C-04 NaN 에 inf 경고 · C-05 `subsample 0.8` 무효(KNOWN-OPEN S16 P8) · C-06 RL docstring·DR 이중 지연(도달 불가) · D-07 통화 대사 항등식(실패 불가) · D-08 name 캡 허용오차 불일치(KNOWN-OPEN R4 T-03) · D-09 스케줄 실패 이력 없음(09-18 이후 성공 0, 09-28 09:48 런 0xC000013A 중단) · D-10 대시보드 150 칩·라벨 · **M-02** §S20 C5/C6 의 G2 는 C1/C2 와 수학적으로 동일(FY1 을 회귀변수로 넣으면 FY2−FY1 잔차 = FY2 잔차) → FY2 가족 실질 검정 8개(Bonferroni 2.73), §S21 판정 불변.

## 확인된 청정 영역(요약)
설정 93키 전부 PipelineConfig 에 존재·미지 키 거부·소비자 도달, 인과 분할 31/31(엠바고 20), 랭크 라벨 동일 날짜, EWMA 과거만, 오버레이 t → 단일 1행 지연, P&L 진입 비중·종가 리밸, TC 양방향 L1×one_way, TE 단위(√252·일간 Σ), active share 정의, 섹터·종목 한도, ECOS 단일·비최적 해 폴백 계수, Σ PSD, option-vol 모델 라벨 p+21<e, S19 run contract·tg 기저 가드 경로, export·번들 교차 검증.
