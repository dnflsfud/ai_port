@echo off
rem S18.6 (2026-09-09): business-day calendar correctness arm. Prerequisite: the ai_signal_data.xlsx / Index.xlsx
rem vintage pair must be the one the current S0' (s18_4_flip2_recert) was certified on -- confirm mtimes BEFORE launch.
rem If the workbook has been regenerated with the High-1 fetch-day cutoff, re-certify S0' first (s18_4 variant) on the new vintage.
rem Launch pattern: schtasks + start /WAIT /MIN cmd /c (see decision log S16 ops note). Judge: scripts\eval_s18_arm.py --arm s18_6_business_day_calendar
cd /d C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_port
echo [%date% %time%] s18_6 start > outputs\s18_6_run.status
powershell -NoProfile -ExecutionPolicy Bypass -File outputs\run_variant_task.ps1 -Label s18_6_business_day_calendar
echo [%date% %time%] DONE >> outputs\s18_6_run.status
