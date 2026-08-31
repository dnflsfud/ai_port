@echo off
rem S16.1: unit/duplication fix pack (s16_unit_fixpack_enabled) vs S0' 1.5356 (same vintage)
rem Same shape as outputs/s15_2_optvol_run.bat, which is the form that actually
rem completed (exit=0, 26 min). The PowerShell wrapper died with 0xC000013A
rem (console control event) mid Phase 5-6 on the first attempt.
cd /d C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_port
set PYTHONPATH=.
set PYTHONUTF8=1
set PY=C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\Scripts\python.exe
echo [%date% %time%] s16.1 unit fixpack run start > outputs\s16_1_runs.status
"%PY%" run_variant.py --variant variants/s16_1_unit_fixpack.yaml --no-cache > outputs\s16_1_run.log 2>&1
echo [%date% %time%] run exit=%errorlevel% >> outputs\s16_1_runs.status
echo DONE >> outputs\s16_1_runs.status
