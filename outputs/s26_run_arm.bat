@echo off
rem S26 (2026-10-31): single accuracy arm s26_stale_fund_mask on the 2026-10-21 workbook / 2026-10-26 Index vintage
rem (base = outputs\s26_s0_base, frozen copy of the 2026-10-30 production run). Launch pattern: schtasks + cmd /c;
rem the arm runs via run_variant_task.ps1 (--no-cache, VINTAGE_PRE/POST). Judge: scripts\eval_s26_stale_fund_arm.py.
cd /d C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_port
echo [%date% %time%] s26 arm start > outputs\s26_chain.status
powershell -NoProfile -ExecutionPolicy Bypass -File outputs\run_variant_task.ps1 -Label s26_stale_fund_mask
echo [%date% %time%] s26_stale_fund_mask done >> outputs\s26_chain.status
echo [%date% %time%] CHAIN_DONE >> outputs\s26_chain.status
