@echo off
rem S17.1 arm chain A->C->D (decision log S17.1, 2026-09-02). Arm B
rem (s17_2_cov_corr_overlap) was SHELVED at the G1 mechanism precheck and is
rem NOT run (decision log S17.1 precheck result 2).
rem Each arm runs through outputs\run_variant_task.ps1 (ES_SYSTEM_REQUIRED hold,
rem --no-cache, VINTAGE_PRE/POST fingerprint in outputs\<label>_run.status).
rem Sequential on purpose: one backtest at a time (RAM), same vintage pair
rem checked per run by scripts\eval_s17_arm.py G0.
cd /d C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_port
echo [%date% %time%] s17 chain start > outputs\s17_chain.status
for %%L in (s17_1_coverage_gap_fix s17_3_beta_overlap s17_4_dead_feature_prune) do (
  echo [%date% %time%] start %%L >> outputs\s17_chain.status
  powershell -NoProfile -ExecutionPolicy Bypass -File outputs\run_variant_task.ps1 -Label %%L
  echo [%date% %time%] end %%L >> outputs\s17_chain.status
)
echo [%date% %time%] DONE >> outputs\s17_chain.status
