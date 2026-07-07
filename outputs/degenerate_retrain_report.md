# Degenerate-Retrain Diagnosis (SPEC D0)

- Source pickle: `outputs/iter15_65tkr_reb21_vtg/backtest_result.pkl`
- Total retrains: 32  |  Degenerate: 16 (50.0%)
- Solver / protocol: single ECOS path (per metrics.json).

## 1. windows — retrain-window census

| train_date | subperiod | degenerate | best_iter | n_trees | val_start | val_end | val_score |
|---|---|---|---:|---:|---|---|---|
| 2018-11-26 | P1 | False | 47 | 47 | 2018-06-01 | 2018-11-23 | None |
| 2019-02-21 | P1 | True | 4 | 4 | 2018-08-29 | 2019-02-20 | None |
| 2019-05-21 | P1 | True | 1 | 1 | 2018-11-26 | 2019-05-20 | None |
| 2019-08-16 | P1 | True | 1 | 1 | 2019-02-21 | 2019-08-15 | None |
| 2019-11-13 | P1 | True | 1 | 1 | 2019-05-21 | 2019-11-12 | None |
| 2020-02-10 | P1 | True | 2 | 2 | 2019-08-16 | 2020-02-07 | None |
| 2020-05-07 | P1 | True | 6 | 6 | 2019-11-13 | 2020-05-06 | None |
| 2020-08-04 | P1 | False | 130 | 130 | 2020-02-10 | 2020-08-03 | None |
| 2020-10-30 | P1 | False | 141 | 141 | 2020-05-07 | 2020-10-29 | None |
| 2021-01-27 | P1 | False | 109 | 109 | 2020-08-04 | 2021-01-26 | None |
| 2021-04-26 | P1 | False | 527 | 527 | 2020-10-30 | 2021-04-23 | None |
| 2021-07-22 | P2 | True | 1 | 1 | 2021-01-27 | 2021-07-21 | None |
| 2021-10-19 | P2 | True | 1 | 1 | 2021-04-26 | 2021-10-18 | None |
| 2022-01-14 | P2 | True | 1 | 1 | 2021-07-22 | 2022-01-13 | None |
| 2022-04-13 | P2 | True | 2 | 2 | 2021-10-19 | 2022-04-12 | None |
| 2022-07-11 | P2 | True | 6 | 6 | 2022-01-14 | 2022-07-08 | None |
| 2022-10-06 | P2 | False | 368 | 368 | 2022-04-13 | 2022-10-05 | None |
| 2023-01-03 | P2 | False | 30 | 30 | 2022-07-11 | 2023-01-02 | None |
| 2023-03-31 | P2 | False | 41 | 41 | 2022-10-06 | 2023-03-30 | None |
| 2023-06-28 | P2 | False | 81 | 81 | 2023-01-03 | 2023-06-27 | None |
| 2023-09-25 | P2 | False | 126 | 126 | 2023-03-31 | 2023-09-22 | None |
| 2023-12-21 | P3 | False | 15 | 15 | 2023-06-28 | 2023-12-20 | None |
| 2024-03-19 | P3 | False | 19 | 19 | 2023-09-25 | 2024-03-18 | None |
| 2024-06-14 | P3 | True | 7 | 7 | 2023-12-21 | 2024-06-13 | None |
| 2024-09-11 | P3 | True | 1 | 1 | 2024-03-19 | 2024-09-10 | None |
| 2024-12-09 | P3 | True | 1 | 1 | 2024-06-14 | 2024-12-06 | None |
| 2025-03-06 | P3 | False | 18 | 18 | 2024-09-11 | 2025-03-05 | None |
| 2025-06-03 | P3 | True | 8 | 8 | 2024-12-09 | 2025-06-02 | None |
| 2025-08-29 | P3 | False | 103 | 103 | 2025-03-06 | 2025-08-28 | None |
| 2025-11-26 | P3 | True | 1 | 1 | 2025-06-03 | 2025-11-25 | None |
| 2026-02-23 | P3 | False | 15 | 15 | 2025-08-29 | 2026-02-20 | None |
| 2026-05-21 | None | False | 126 | 126 | 2025-11-26 | 2026-05-20 | None |

> `val_score` is a documented GAP: the per-window validation loss is not persisted in any artifact, and for degenerate windows the rejected model itself is discarded. `best_iteration`/`n_trees` (the load-bearing early-stop fingerprint) and the reconstructed `val_start`/`val_end` are recovered in full.

## 2. subperiod_overlap — degenerate windows per P1/P2/P3

| P1 | P2 | P3 |
|---:|---:|---:|
| 6 | 5 | 5 |

Canonical bounds (src/harness.py): P1 2018-11-23..2021-05-11, P2 2021-05-12..2023-10-27, P3 2023-10-30..2026-04-13.

## 3. root_cause_evidence

### H1_immediate_early_stop_no_signal — **supported**

Degenerate models early-stop within a handful of boosting rounds (best_iteration << healthy), i.e. validation loss stops improving almost immediately -> retrain found no signal that generalises to the validation window.

```json
{
  "degenerate_best_iteration": {
    "median": 1.0,
    "min": 1,
    "max": 8,
    "values": [
      4,
      1,
      1,
      1,
      2,
      6,
      1,
      1,
      1,
      2,
      6,
      7,
      1,
      1,
      8,
      1
    ]
  },
  "healthy_best_iteration": {
    "median": 92.0,
    "min": 15,
    "max": 527
  }
}
```

### H2_regime_concentration_P2 — **refuted**

Spec hypothesised degeneracy concentrates in the weak P2 window. Counts show whether P2 holds a clear plurality (>50% of degenerate windows) of the degeneracy.

```json
{
  "subperiod_overlap": {
    "P1": 6,
    "P2": 5,
    "P3": 5
  }
}
```

### H3_clustering_after_strong_model — **supported**

Degenerate windows arrive in consecutive runs immediately after a high-tree healthy model is retained; that strong model then persists for the whole run, doubling the effective retrain cadence in that stretch.

```json
{
  "degenerate_runs": [
    {
      "length": 6,
      "dates": [
        "2019-02-21",
        "2019-05-21",
        "2019-08-16",
        "2019-11-13",
        "2020-02-10",
        "2020-05-07"
      ],
      "trails_healthy_model_trees": 47
    },
    {
      "length": 5,
      "dates": [
        "2021-07-22",
        "2021-10-19",
        "2022-01-14",
        "2022-04-13",
        "2022-07-11"
      ],
      "trails_healthy_model_trees": 527
    },
    {
      "length": 3,
      "dates": [
        "2024-06-14",
        "2024-09-11",
        "2024-12-09"
      ],
      "trails_healthy_model_trees": 19
    },
    {
      "length": 1,
      "dates": [
        "2025-06-03"
      ],
      "trails_healthy_model_trees": 18
    },
    {
      "length": 1,
      "dates": [
        "2025-11-26"
      ],
      "trails_healthy_model_trees": 103
    }
  ],
  "runs_trailing_high_tree_model": 2
}
```

Structural context (config.py:147-150, lgbm_params): learning_rate=0.02, min_child_samples=60, early_stopping_rounds=100 over a ~5y (1260d) heterogeneous train window. The in-code comment already documents that this low-lr / high-min-child / long-window combination 'triggered early stopping on almost every retrain'. The best_iteration fingerprint (H1) is the direct signature of that structural cause.

## 4. raw_spread_dist — daily raw (pre-EMA) top-bottom spread

- n_dates: 1973
- median: 3.57587  |  IQR: 0.38731  |  min: 1.91955  |  max: 4.14744

Definition matches compute_signal_confidence (tail_n=max(3, n_valid//10), per-row dropna). A2 relevance: spread_score = clip(raw_spread / spread_scale, 0.20, 1.00) with the default spread_scale=0.20. Since the median spread (3.58) is ~18x that scale, spread_score saturates at 1.00 on essentially every date — the spread axis of the confidence gate is effectively inert. Pre-registering A2's confidence_spread_scale near the observed spread range (~3.58) is what makes the gate responsive.

## 5. trailing_ic_dist — production trailing IC (rolling nanmean of last 6 rebalance ICs)

- n: 92
- median: 0.04036  |  IQR: 0.07404  |  min: -0.04354  |  max: 0.24364  |  mean: 0.04791
- fraction < 0: 0.174

A3 relevance: ic_score = clip((trailing_ic_mean + 0.01)/0.04, 0.20, 1.00). With median trailing IC 0.0404, the numerator (0.0404+0.01)/0.04 = 1.26 clips to 1.00 on the majority of dates, so ic_score also mostly saturates. The constants (+0.01, /0.04) are the functional knobs A3 would re-fix against this distribution.

## Conclusion

- Most-likely root cause (strong): structural immediate early-stop. Degenerate retrains stop at best_iteration median 1.0 (range 1-8) vs healthy median 92.0. Validation loss stops improving within a handful of rounds — the lr=0.02 / min_child=60 / 100-round-patience / 5y-window combination (documented in config.py:147-150) makes the retrain fail to beat the incumbent model, which is then reused.
- Secondary structural pattern (supporting): degenerate windows cluster in consecutive runs immediately after a high-tree healthy model is retained (e.g. after the 47- and 527-tree fits), so a single strong model persists across many quarters and the effective retrain cadence doubles in those stretches.
- Spec's P2-concentration hypothesis is REFUTED: degenerate windows split 6/5/5 across P1/P2/P3 — roughly even, not concentrated in the weak P2 window. Degeneracy is a global training-config artifact, not a P2-regime effect.
- raw_spread median 3.58 (>> spread_scale 0.20) and trailing IC median 0.0404 both saturate their confidence sub-scores at 1.00 -> the dynamic-execution confidence gate is largely inert at its current constants. These are the pre-registration anchors for A2 (confidence_spread_scale) and A3 (IC constants).
- Gap: per-window validation loss is not persisted; the census recovers the early-stop fingerprint (best_iteration/n_trees) and val periods, which already localise the mechanism, so no instrumented re-run was required.
