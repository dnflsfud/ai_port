@echo off
rem S17.1 arm D re-run (decision log S17.1, 2026-09-03). The 2026-09-02 chain
rem A->C->D lost arm D to a system reboot at 01:06:43 (boot 01:11:07) after 4
rem retrains; A and C completed EXIT 0. Same vintage pair as the frozen S0'
rem (workbook 2026-09-01 14:05:40 / Index 2026-09-02 11:09:18), re-checked by
rem outputs\run_variant_task.ps1 VINTAGE_PRE/POST and scripts\eval_s17_arm.py G0.
cd /d C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_port
echo [%date% %time%] s17 D rerun start > outputs\s17_d_rerun.status
powershell -NoProfile -ExecutionPolicy Bypass -File outputs\run_variant_task.ps1 -Label s17_4_dead_feature_prune
echo [%date% %time%] DONE >> outputs\s17_d_rerun.status
