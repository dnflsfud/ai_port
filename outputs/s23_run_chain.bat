@echo off
rem S23.1 chain (2026-09-28): s23_s0recert (S0' re-certification, must reproduce the 12:00 scheduled
rem production IR 1.693143983245076) -> s23_m01_no_slope -> s23_b01_label_uncentered -> s23_d04_optvol_lag.
rem Vintage pair must stay workbook 2026-09-18 14:13:24 / Index 2026-09-28 11:23:35 (VINTAGE_PRE/POST in
rem outputs\<label>_run.status). Sequential on purpose (RAM). Judge: scripts\eval_s23_arm.py --arm <label>
cd /d C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_port
echo [%date% %time%] s23 chain start > outputs\s23_chain.status
for %%L in (s23_s0recert s23_m01_no_slope s23_b01_label_uncentered s23_d04_optvol_lag) do (
  echo [%date% %time%] start %%L >> outputs\s23_chain.status
  powershell -NoProfile -ExecutionPolicy Bypass -File outputs\run_variant_task.ps1 -Label %%L
  echo [%date% %time%] end %%L >> outputs\s23_chain.status
)
echo [%date% %time%] DONE >> outputs\s23_chain.status
