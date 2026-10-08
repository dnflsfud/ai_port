@echo off
rem S25.4 (2026-10-08): four accuracy arms on the 2026-10-08 vintage (base outputs/s25_3_s0recert), sequential.
rem Launch pattern: schtasks + cmd /c (decision log S16 ops note); each arm via run_variant_task.ps1 (--no-cache, VINTAGE_PRE/POST).
rem Judge: scripts\eval_s25_4_arms.py --arm <label> (frame S25.4: G0 + mechanism + E2 + no-harm).
cd /d C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_port
echo [%date% %time%] s25_4 chain start > outputs\s25_4_chain.status
for %%A in (s25_4_b02_no_gradual_mask s25_4_b03_winsor_first s25_4_b05_label_lag s25_4_stale_run_mask) do (
  powershell -NoProfile -ExecutionPolicy Bypass -File outputs\run_variant_task.ps1 -Label %%A
  echo [%date% %time%] %%A done >> outputs\s25_4_chain.status
)
echo [%date% %time%] CHAIN_DONE >> outputs\s25_4_chain.status
