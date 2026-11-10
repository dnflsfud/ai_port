@echo off
rem S28.1 (2026-11-10): four runs under one code state and one Index.xlsx window (decision log S28.1).
rem   s28_s0_1106            control, production overrides, 2026-11-06 workbook (G0 parity vs the pre-landing run)
rem   s28_1_kfold_refit_1106 arm, kfold_capacity_refit_enabled, 2026-11-06 workbook
rem   s28_ctl_1021           control, preserved 2026-10-21 workbook
rem   s28_1_kfold_refit_1021 arm, preserved 2026-10-21 workbook
rem Launch pattern: schtasks + cmd /c; each run via run_variant_task.ps1 (--no-cache, VINTAGE_PRE/POST).
rem Judge: scripts\eval_s28_1_kfold_arm.py.
cd /d C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_port
echo [%date% %time%] s28_1 chain start > outputs\s28_1_chain.status
powershell -NoProfile -ExecutionPolicy Bypass -File outputs\run_variant_task.ps1 -Label s28_s0_1106
echo [%date% %time%] s28_s0_1106 done >> outputs\s28_1_chain.status
powershell -NoProfile -ExecutionPolicy Bypass -File outputs\run_variant_task.ps1 -Label s28_1_kfold_refit_1106
echo [%date% %time%] s28_1_kfold_refit_1106 done >> outputs\s28_1_chain.status
powershell -NoProfile -ExecutionPolicy Bypass -File outputs\run_variant_task.ps1 -Label s28_ctl_1021
echo [%date% %time%] s28_ctl_1021 done >> outputs\s28_1_chain.status
powershell -NoProfile -ExecutionPolicy Bypass -File outputs\run_variant_task.ps1 -Label s28_1_kfold_refit_1021
echo [%date% %time%] s28_1_kfold_refit_1021 done >> outputs\s28_1_chain.status
echo [%date% %time%] CHAIN_DONE >> outputs\s28_1_chain.status
