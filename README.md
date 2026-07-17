# Pictet Portfolio Logic Adoption

This folder is the working document set for adapting Pictet Quest AI-driven
enhanced-index portfolio-construction logic into `cc2_rl`.

## Artifact Flow

1. **Source PDF**
   - `Pictet_Quest AI-driven strategies_Knowledge_20260531.pdf`
   - Role: source material only. Use it to validate what Pictet actually claims:
     beta 1.0, low tracking error, broad 400-500 stock portfolio shape, index-like
     country/industry/factor exposure, stock/country/industry active weight limits,
     attribution framing, and ESG/Article 8 context.

2. **Design Spec**
   - `2026-06-18-pictet-portfolio-logic-adoption-design.md`
   - Role: target architecture and research protocol. This should explain why each
     candidate exists, what is in or out of scope, and the acceptance gates.

3. **Implementation Plan**
   - `2026-06-18-pictet-portfolio-logic-adoption.md`
   - Role: step-by-step execution checklist. This should be mechanically actionable
     and should not contradict the design spec.

4. **Decision Log**
   - Recommended file: `2026-06-18-pictet-portfolio-logic-adoption-decision-log.md`
   - Role: record measured outcomes after each run, including S0 ECOS baseline,
     attribution parity, overlay ablation, beta sweep, factor exposure binding,
     DSR/selection-bias accounting, and final production flip decisions.
   - Status: not present yet. Create it before the first baseline or ablation result
     is used to justify a production change.

## Intended Reading Order

Read the source PDF only to verify mapping claims, then read the design spec for the
research contract, then execute the implementation plan task by task. The decision
log should be updated after every material measurement so the final production
variant is justified by recorded gates rather than reconstructed memory.

## Current Portfolio Framing

This project is better described as **Pictet-inspired risk discipline and
explainability for the existing cc2_rl portfolio**, not a literal replication of
Pictet Quest. The local baseline now uses a concentrated 100-stock universe,
while the PDF describes a broader enhanced-index portfolio. Portfolio and
benchmark returns are measured in unhedged USD, including listing-currency moves.

Before implementation, the design spec should add a source-vs-local deviation table:

| Dimension | Pictet PDF | Current cc2_rl intent | Action |
|---|---:|---:|---|
| Holdings | 400-500 stocks | 100 stocks | Explicitly mark as intentional non-replication |
| Beta | 1.0 | Measure S0 first, then decide beta-neutral work | Gate P2 before coding |
| Tracking error | Up to 2% | Local target around 3.2%-4.5% guard | Clarify local risk budget |
| Active share | About 50% | Document currently says about 4.75% | Confirm unit/definition |
| Stock active weight | +/-1% | Local active caps differ | Mark as adapted constraint |
| Country/industry exposure | +/-2% | Sector deviation only | Mark missing or out of scope |
| Factor exposure | Similar to index | Proposed soft factor penalty | Validate loadings before ablation |
| ESG / Article 8 | Present in PDF | Out of scope due to missing data | Keep explicitly excluded |

## Required Spec/Plan Cleanup

Fix these before treating the plan as implementation-ready:

1. **Path mismatch**
   - The implementation plan points to
     `docs/superpowers/specs/2026-06-18-pictet-portfolio-logic-adoption-design.md`,
     but the actual design spec is in this root folder.
   - Either move the files into the documented structure or update the plan path.

2. **S0 status ambiguity**
   - ECOS installation and ECOS baseline re-certification are different states.
   - The design spec should say ECOS is installed, while S0 remains pending until
     `run_variant.py --variant variants/iter15_65tkr_reb21_vtg.yaml` is rerun and
     metrics are recorded in the decision log.

3. **Apply vs evaluate wording**
   - Replace "P0-P3 all applied" with "P0-P3 all evaluated; only gate-passing
     candidates are enabled in production."
   - This keeps the text aligned with the beta/factor shelve gates.

4. **Leg C / overlay-free prediction definition**
   - The plan uses `base.raw_predictions` as an overlay-free EMA base in one place,
     while the design spec warns that pre/post overlay and pre/post EMA confusion is
     a blocker.
   - Define one canonical object, such as `pre_overlay_ema_predictions`, and use it
     consistently in alpha attribution and overlay ablation scripts.

5. **Beta implementation mismatch**
   - The design spec describes a 4-tuple `_build_mvo_constraints` change and
     projection objective edit.
   - The implementation plan later switches to an inline `optimize_portfolio`
     objective penalty.
   - Pick one as the authoritative approach. The inline plan is smaller and more
     consistent with existing risk/turnover objective handling, but the spec must be
     updated if that is the intended implementation.

6. **Factor-neutral placeholder risk**
   - `factor_neutral_loadings` still contains columns that must be verified against
     the actual feature panel.
   - Add a required pre-ablation check: all configured columns exist, applied-date
     count is non-zero, and missing/non-finite imputation rates are reported. Without
     this, an inert factor penalty could be misread as "TE already neutralizes style."

7. **Selection-bias accounting**
   - DSR/deflation accounting is currently left as an open item.
   - Make it a production gate in the decision log before enabling any performance-
     affecting candidate, especially overlay removal or beta/factor penalties.

8. **Repository hygiene**
   - `.claude/settings.local.json` contains local absolute paths and permissions.
   - `bash.exe.stackdump` is a local crash artifact.
   - Exclude both from any portable document package or commit unless there is a
     specific debugging reason to keep them.

## Production Rule

Keep all new behavior default-OFF in `PipelineConfig`. Enable candidates only through
the production variant after the decision log records the relevant gate:

- attribution: parity and cost acceptable;
- overlay change: full-period marginal dIR is clear (>1SE) and P1/P2/P3 sign-consistent (NOT OOS — the OOS holdout is inert in the harvest-once re-MVO; see scripts/run_overlay_ablation.py:12);
- beta-neutral: realized beta moves toward 1.0 without fallback or risk-budget damage;
- factor-neutral: penalized active exposures actually bind and drop.
