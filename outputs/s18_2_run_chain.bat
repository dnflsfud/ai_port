@echo off
rem S18.2/S18.3 chain (2026-09-08): s18_4_flip2_recert (two-flip S0' re-certification) -> s18_5_static_execution_eta042
rem (S18.3 turnover-neutral re-attempt of arm 3). Vintage pair must stay workbook 2026-09-04 14:50:52 / Index 2026-09-07 11:04:19
rem (VINTAGE_PRE/POST in outputs\<label>_run.status). Sequential on purpose (RAM). Judge: scripts\eval_s18_arm.py --arm s18_5_static_execution_eta042
cd /d C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_port
echo [%date% %time%] s18_2 chain start > outputs\s18_2_chain.status
for %%L in (s18_4_flip2_recert s18_5_static_execution_eta042) do (
  echo [%date% %time%] start %%L >> outputs\s18_2_chain.status
  powershell -NoProfile -ExecutionPolicy Bypass -File outputs\run_variant_task.ps1 -Label %%L
  echo [%date% %time%] end %%L >> outputs\s18_2_chain.status
)
echo [%date% %time%] DONE >> outputs\s18_2_chain.status
