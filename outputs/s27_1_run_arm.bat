@echo off
rem S27.1 (2026-11-03): single risk-discipline arm s27_1_sector_risk_cap on the 2026-10-21 workbook vintage
rem (base = outputs\s27_om_pit_restamp = new S0' OM-PIT; after the S27 flip its overrides equal production's).
rem Launch pattern: schtasks + cmd /c; the arm runs via run_variant_task.ps1 (--no-cache, VINTAGE_PRE/POST).
rem Judge: scripts\eval_s27_1_sector_cap_arm.py.
cd /d C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_port
echo [%date% %time%] s27_1 arm start > outputs\s27_1_chain.status
powershell -NoProfile -ExecutionPolicy Bypass -File outputs\run_variant_task.ps1 -Label s27_1_sector_risk_cap
echo [%date% %time%] s27_1_sector_risk_cap done >> outputs\s27_1_chain.status
echo [%date% %time%] CHAIN_DONE >> outputs\s27_1_chain.status
