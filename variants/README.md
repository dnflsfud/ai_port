# Variants — declarative experiment manifests

Each YAML file in this directory describes **one** PipelineConfig override set
plus metadata. Drive any experiment via:

```bash
python run_variant.py --variant variants/<name>.yaml
```

instead of adding another `run_iter{N}.py` script. The previous per-iteration
runner pattern fragmented entry points (8+ scripts at peak) and made config
drift easy — see `../archive/` for the legacy runners this replaces.

## Schema

```yaml
label: iter15_FINAL              # goes into outputs/<label>/
description: "..."               # free text; written into experiment_manifest
out_dir: outputs/iter15_FINAL    # default: outputs/{label}
tuning_mode: production          # {production, tuning, oos_verify}
                                 # - production: no OOS gating
                                 # - tuning: enforce_oos_holdout=True
                                 # - oos_verify: special marker, only one
                                 #   verification run per candidate
overrides:                       # any PipelineConfig field (name: value)
  max_te_annual: 0.045
  satellite_budget: 0.225
  ...
```

Unknown fields under `overrides:` are rejected at load time so typos surface
early instead of silently doing nothing.

## Current variants

- `iter15_FINAL.yaml` — canonical baseline (see `../docs/BASELINE.md`).
  This is the only "blessed" variant on disk right now. Metrics produced
  by this manifest must exactly match `../outputs/iter15_FINAL/metrics.json`.

## Tuning workflow

1. Copy `iter15_FINAL.yaml` to a new file (e.g. `exp_p2_multi_horizon.yaml`).
2. Set `tuning_mode: tuning` and set `train_cutoff_date` in overrides to
   freeze OOS (recommended: `"2024-12-31"`).
3. Run: `python run_variant.py --variant variants/exp_p2_multi_horizon.yaml`.
4. Inspect `outputs/<label>/metrics.json`. Gate against baseline using
   `docs/BASELINE.md` criteria.
5. If the candidate wins tuning metrics, create a SECOND yaml with
   `tuning_mode: oos_verify` and `enforce_oos_holdout: false` and run ONCE
   for the holdout. If it still wins, propose promotion to baseline.

## Rules

- NEVER run a tuning variant with `tuning_mode: production` — that peeks at
  OOS data and inflates selection bias.
- Do NOT commit a yaml that resurrects a rollback-listed lever without
  first reading `../docs/rollback_log.md`.
- Experiment directories follow `outputs/<label>/` convention. Never
  overwrite the top-level `outputs/`.
