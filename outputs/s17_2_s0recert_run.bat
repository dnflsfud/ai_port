@echo off
rem S17.2 S0 recert on the 2026-09-03 PX_LAST_UNADJ vintage (decision log S17.2).
rem Production config copy (variants/s17_2_s0recert.yaml) via outputs/run_variant_task.ps1
rem (ES_SYSTEM_REQUIRED hold, --no-cache, VINTAGE_PRE/POST fingerprint).
cd /d C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_port
echo [%date% %time%] s17_2 s0recert start > outputs\s17_2_s0recert_chain.status
powershell -NoProfile -ExecutionPolicy Bypass -File outputs\run_variant_task.ps1 -Label s17_2_s0recert
echo [%date% %time%] DONE >> outputs\s17_2_s0recert_chain.status
