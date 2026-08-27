@echo off
rem S15.1: monotone margin arm vs S0' 1.5466 (same vintage)
cd /d C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_port
set PYTHONPATH=.
set PYTHONUTF8=1
set PY=C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\Scripts\python.exe
echo [%date% %time%] s15.1 monotone arm start > outputs\s15_1_monotone_runs.status
"%PY%" run_variant.py --variant variants/arm_s15_1_monotone_margin.yaml --no-cache > outputs\s15_1_monotone_run.log 2>&1
echo [%date% %time%] run exit=%errorlevel% >> outputs\s15_1_monotone_runs.status
echo DONE >> outputs\s15_1_monotone_runs.status
