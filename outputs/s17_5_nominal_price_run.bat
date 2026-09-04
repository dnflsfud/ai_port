@echo off
rem S17.3 G1-01b arm s17_5_nominal_price (decision log S17.3, 2026-09-03): production 1c3acae
rem + nominal_price_source=PX_LAST_UNADJ, run after the data-gate precheck passes.
rem outputs\run_variant_task.ps1: ES_SYSTEM_REQUIRED hold, --no-cache, VINTAGE_PRE/POST fingerprint.
cd /d C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_port
echo [%date% %time%] s17_5 arm start > outputs\s17_5_nominal_price_chain.status
powershell -NoProfile -ExecutionPolicy Bypass -File outputs\run_variant_task.ps1 -Label s17_5_nominal_price
echo [%date% %time%] DONE >> outputs\s17_5_nominal_price_chain.status
