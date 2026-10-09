@echo off
rem S25.6 (2026-10-09): two performance arms on the 2026-10-08 vintage (base outputs/s25_4_combined_recert), sequential.
rem Launch pattern: schtasks + cmd /c; each arm via run_variant_task.ps1 (--no-cache, VINTAGE_PRE/POST).
rem Judge: scripts\eval_s25_6_arms.py --arm <label> (frame: G0 + mechanism + E2 + formal E1; override candidate reported).
cd /d C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_port
echo [%date% %time%] s25_6 chain start > outputs\s25_6_chain.status
for %%A in (s25_6_top_decile_boost s25_6_sector_risk_penalty) do (
  powershell -NoProfile -ExecutionPolicy Bypass -File outputs\run_variant_task.ps1 -Label %%A
  echo [%date% %time%] %%A done >> outputs\s25_6_chain.status
)
echo [%date% %time%] CHAIN_DONE >> outputs\s25_6_chain.status
