@echo off
rem S18.1 arm chain 1->2->3 (decision log S18.1, 2026-09-07). Base = outputs\s18_s0recert
rem (09-07 12:36 production run frozen: workbook 2026-09-04 05:50:52Z / Index 2026-09-07 02:04:19Z).
rem Each arm runs through outputs\run_variant_task.ps1 (ES_SYSTEM_REQUIRED hold, --no-cache,
rem VINTAGE_PRE/POST fingerprint in outputs\<label>_run.status). Sequential on purpose (RAM);
rem scripts\eval_s18_arm.py G0 checks the vintage pair per arm.
cd /d C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_port
echo [%date% %time%] s18 chain start > outputs\s18_chain.status
for %%L in (s18_1_tilt_negative_equity s18_2_tg_basis_events s18_3_static_execution) do (
  echo [%date% %time%] start %%L >> outputs\s18_chain.status
  powershell -NoProfile -ExecutionPolicy Bypass -File outputs\run_variant_task.ps1 -Label %%L
  echo [%date% %time%] end %%L >> outputs\s18_chain.status
)
echo [%date% %time%] DONE >> outputs\s18_chain.status
