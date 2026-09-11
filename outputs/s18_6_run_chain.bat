@echo off
rem S18.7 chain (2026-09-11): s18_7_s0recert (S0' re-certification on the regenerated 2026-09-11 13:34:45
rem workbook, High-1 cutoff in effect) -> s18_6_business_day_calendar (data audit Medium-2 correctness arm).
rem Vintage pair must stay workbook 2026-09-11 13:34:45 / Index 2026-09-11 11:00:19 (VINTAGE_PRE/POST in
rem outputs\<label>_run.status). Sequential on purpose (RAM). Judge: scripts\eval_s18_arm.py --arm s18_6_business_day_calendar
cd /d C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_port
echo [%date% %time%] s18_6 chain start > outputs\s18_6_chain.status
for %%L in (s18_7_s0recert s18_6_business_day_calendar) do (
  echo [%date% %time%] start %%L >> outputs\s18_6_chain.status
  powershell -NoProfile -ExecutionPolicy Bypass -File outputs\run_variant_task.ps1 -Label %%L
  echo [%date% %time%] end %%L >> outputs\s18_6_chain.status
)
echo [%date% %time%] DONE >> outputs\s18_6_chain.status
