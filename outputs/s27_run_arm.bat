@echo off
rem S27 (2026-11-03): single look-ahead-class arm s27_om_pit_restamp on the 2026-10-21 workbook vintage
rem (base = outputs\s26_s0_base, frozen copy of the 2026-10-30 production run; the 11-02/11-03 production runs reproduce
rem its IR bit for bit). Launch pattern: schtasks + cmd /c; the arm runs via run_variant_task.ps1 (--no-cache,
rem VINTAGE_PRE/POST). Judge: scripts\eval_s27_om_pit_arm.py.
cd /d C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_port
echo [%date% %time%] s27 arm start > outputs\s27_chain.status
powershell -NoProfile -ExecutionPolicy Bypass -File outputs\run_variant_task.ps1 -Label s27_om_pit_restamp
echo [%date% %time%] s27_om_pit_restamp done >> outputs\s27_chain.status
echo [%date% %time%] CHAIN_DONE >> outputs\s27_chain.status
