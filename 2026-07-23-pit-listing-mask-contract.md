# PIT 상장 마스킹 계약서 (Point-in-Time Listing Mask Contract)

> **불변식**: 유니버스에 포함된 어떤 종목도 **상장일(eligibility start) 이전에는
> ① 포트폴리오 가중치, ② 벤치마크 가중치, ③ 피처·타깃·예측 계산 어디에도
> 포함되지 않는다.** 이 문서는 그 불변식이 코드 어디에서 어떻게 강제되는지,
> 신규 종목 추가 시 무엇을 해야 하는지, 무엇이 금지인지의 정본이다.
>
> 우선순위: `CLAUDE.md`(운영 계약서) > 이 문서 > spec/plan. `CLAUDE.md` §2.1의
> "데이터 정확성 계층 default-ON 예외"와 결정 로그 §S9/§S11.4의 확장 서술을
> 한 곳으로 모은 것이며, 충돌 시 이 문서가 spec/plan을 이긴다.

작성: 2026-07-23 (§S13 유니버스 150→200 준비 중 계약 명문화).
근거 사건: 2026-07-02 구조 리뷰 Critical #1(PLTR·GEV·BE 상장 전 유령값),
§S11.3 실측(ABNB 상장 전 상수 백필 68.0 고정 — **채움률≠실존**),
§S11.4(재무·컨센서스 19시트 백필이 피처 횡단면·임퓨트 median 오염 → 전 시트
이중 마스킹 + S0(150)′ 재인증).

---

## 1. 정본 일자: eligibility start의 해석 순서

모든 유니버스 종목은 `resolve_listing_dates()`가 일자를 해석한다
(`src/data_loader.py:439`). **낮은 번호가 이긴다** (나중에 덮어씀):

| 우선 | 소스 | 용도 |
|---|---|---|
| 1 | `config.listing_dates` (config_override) | **기업행사 오버라이드** — 분사·합병 등 "사건 이후만 학습 인정" (GE/GEV·TT·BN 선례) |
| 2 | `Universe_Meta` 날짜 컬럼 (`Eligibility_Start_Date` > `Listing_Date` > `IPO_Date`, `src/config.py:55`) | 워크북에 명시된 상장일 |
| 3 | `PX_LAST` 선행 상수열 추론 (`_infer_eligibility_from_price`, `src/data_loader.py:411`) | 벤더 백필 자동 탐지 — 선행 동일가 ≥5관측이면 첫 가격 변화일을 상장일로 |

신규 종목은 메타 행만 추가하면 **코드 수정 없이** 같은 규칙을 상속한다.
해석 결과와 소스는 `data.data_quality["listing_mask"]`에 기록된다
(`dates`/`source_counts`/`unresolved_tickers`, `src/data_loader.py:1046-1060`).

**inclusive 규칙** (`mask_pre_listing`, `src/data_loader.py:936`):
- **수익률·타깃·예측 = inclusive** (`index <= 상장일` 마스크) — 상장일 당일
  수익률은 유령 기준가 대비 계산이라 그것도 가짜다.
- **레벨 시트(가격·시총·재무) = exclusive** (`index < 상장일` 마스크) — 상장일
  당일 관측이 첫 실측이다.

## 2. 방어선 7층 (전부 `listing_mask_enabled=True`에 걸려 있음)

| # | 표면 | 강제 지점 | 메커니즘 |
|---|---|---|---|
| 1 | 피처 원료 (1차) | `preprocess_sheets`, `src/data_loader.py:547-556` | **임퓨트 전** 마스킹 — 유령 상수가 횡단면 median에 못 섞임 |
| 2 | 피처 원료 (2차) | `src/data_loader.py:1023-1046` | align/impute가 되채운 셀을 **전 시트 재마스킹**(§S11.4; `Daily_Returns`만 면제 — §4) |
| 3 | 수익률 PIT 뷰 | `src/data_loader.py:1042-1045, 1428-1429` | `raw_returns`·수익률 뷰 inclusive 마스크 — survivorship 점검(`run_selection_bias.py`)·breadth 분모의 근거 |
| 4 | **벤치마크** | `make_capweight_bm_fn`, `src/backtest.py:948-996` + `_listing_eligibility_mask`, `src/backtest.py:904` | 상장 전 이름은 median-fill 없이 **가중치 0** (cap-weighted); EW 폴백도 eligible 이름만 분모 (`make_ew_bm_fn`, `src/backtest.py:920-936`) |
| 5 | 학습 타깃 | `src/backtest.py:1648-1652` | walk-forward 학습 **전** 타깃 inclusive 마스크 — 유령 행이 학습에 불참 |
| 6 | PCA 타깃 기저 | `src/target_engine.py:94, 320` (§S11.7) | 윈도우에 상장 전 NaN이 있는 열은 PCA 기저에서 제외 (eligibility-aware) |
| 7 | **포트폴리오** | 예측 마스킹 `src/backtest.py:1701-1709` (오버레이 **전**) + 리밸런스 `pred_row.dropna()` `src/backtest.py:1057` | 상장 전 이름은 µ가 NaN → 최적화 유니버스에서 탈락 → **가중치 자체가 생성되지 않음** |

체인 요약: 상장 전 셀은 데이터 계층에서 NaN(1–3) → 벤치마크 분모에서 0(4) →
학습·타깃·PCA 불참(5–6) → 예측 NaN으로 최적화 탈락(7). 어느 한 층이 뚫려도
다음 층이 막는 다층 구조이며, **한 층이라도 제거하는 변경은 §S11.4급 재인증
사유**다.

## 3. 설정 스위치 (전부 `src/config.py:42-90`)

- `listing_mask_enabled: bool = True` — **default-ON**. `CLAUDE.md` §2.1
  OFF-default 원칙의 명시적 예외(데이터 정확성 계층): OFF면 다통화·확장
  유니버스에서 결과 자체가 틀린다. **OFF 상태로 산출된 100/150/200 유니버스
  결과는 신뢰·보고 금지.**
- `listing_auto_infer_enabled` / `listing_flat_min_run=5` / `listing_flat_rtol`
  / `listing_flat_atol` — PX_LAST 추론 파라미터.
- `listing_meta_columns` — Universe_Meta 날짜 컬럼 탐색 순서.
- `listing_dates: dict` — 기업행사 오버라이드(최우선). 분사·합병 종목의
  "사건 이후만 인정"은 여기 등록한다.

## 4. 알려진 예외 1건과 그 근거

`Daily_Returns`는 2차 재마스킹 **면제**다
(`LISTING_REMASK_EXEMPT_SHEETS`, `src/data_loader.py:175-178`): 1차 마스크
(inclusive) 후 임퓨트가 상장 전 셀을 **상장된 이름들의 당일 횡단면 median**으로
채우는데, 이는 피처 파이프라인 정렬을 위한 중립값이지 해당 종목의 정보가
아니다. 라벨·PnL 누수는 방어선 5·7(타깃·예측 셀 마스크)이 차단하고, 실존
판정이 필요한 소비자는 `raw_returns`(방어선 3)를 쓴다. **이 예외를 다른
시트로 확대하는 것은 금지** — 확대가 필요하면 §9 절차(결정 로그 보고)로.

## 5. 검증 — 판정 가능한 합격기준

리프레시·유니버스 변경·마스크 관련 코드 수정 후 아래를 실행한다
(WD=`ai_port`, `PYTHONPATH=.`):

```
<PY> -m pytest tests/acceptance/test_listing_mask.py tests/acceptance/test_pit_universe.py tests/test_pit_sp500_ai_v2.py -v
```
(보조 커버리지: `tests/test_data_loader.py`, `tests/test_backtest.py`,
`tests/test_config.py`, `tests/test_universe_fx_conversion.py`)

백테스트/운영 런에서는:
1. 로그에 `[UniverseData] listing re-mask applied: N cell(s)` 존재 확인
   (`src/data_loader.py:1061`).
2. `data.data_quality["listing_mask"]`에서 **`unresolved_tickers == []`** 확인
   — 미해석 종목이 있으면 그 종목은 마스크 밖이다. 즉시 중단·조사.
3. 신규 종목의 `dates` 값이 등록/공지 상장일과 일치하는지 first-valid로 대조
   (상수 백필 때문에 **채움률로 실존을 판정하지 않는다** — §S11.3 실측).
4. 벤치마크: 상장 전 구간 리밸런스 날짜에서 해당 종목 bm 가중치 0 확인
   (§S11.4 인증 때 breadth 분모와 함께 검증된 항목).

## 6. 신규 종목 추가 절차 (§S13 이후 표준)

1. **슬레이트 단계**: IPO가 데이터 시작(2014) 이후인 종목·기업행사(분사·합병)
   종목을 **사전 등록**한다(결정 로그 게이트 3). §S13 등록분:
   - IPO 6건: TTD 2016-09-21 · UBER 2019-05-10 · CRWD 2019-06-12 ·
     DDOG 2019-09-19 · DASH 2020-12-09 · RBLX 2021-03-10.
   - 기업행사 2건: **WDC** 2025-02 SNDK 분사 RemainCo(GE/GEV 선례 기본 적용 —
     분사 이후만 인정) · **COF** 2025-05 Discover 흡수(리프레시 시 규모 단절
     보고 후 마스크 여부 판단).
2. **리프레시 단계**: Universe_Meta에 행 추가(+가능하면 `Listing_Date` 기입).
   자동 추론이 1차 방어지만, 등록 일자와 추론 일자가 다르면 **등록 일자를
   `config.listing_dates` 오버라이드로 고정**하고 차이를 결정 로그에 기록.
3. **인증 단계**: §5의 합격기준 전부 통과 + 백테스트 로그에서 신규 종목 마스크
   적용 확인(§S11.3의 "마스크 13종 적용 확인" 선례) 후에만 새 S0 인증에 진입.

## 7. 금지 사항

- `listing_mask_enabled=False`로 확장 유니버스(100 이후) 백테스트·운영 산출물
  생성/보고 금지 (§3).
- 마스크 층 우회 주입 금지: 마스크 전 패널을 `precomputed_predictions`로
  재주입하면 방어선 7이 무력화된다 — 주입 패널은 반드시 마스크 후 산출물이어야
  한다 (`CLAUDE.md` §4.2 이중 오버레이 금지와 동일 계열).
- 벤치마크 median-fill 경로(레거시, `src/backtest.py:979-984`)는 마스크 OFF
  전용 호환 경로다 — 마스크 ON에서 이 경로로 회귀시키는 변경 금지.
- `Daily_Returns` 면제를 타 시트로 확대 금지 (§4).
- 코드/데이터가 이 문서와 어긋나는 것을 발견하면 **그대로 진행하지 말고**
  결정 로그에 기록 후 보고한다 (`CLAUDE.md` §9).
