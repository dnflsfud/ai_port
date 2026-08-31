@echo off
rem S15.2: optvol scale fix (option_vol_scale_fix_enabled) vs S0' 1.5466 (same vintage)
cd /d C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_port
set PYTHONPATH=.
set PYTHONUTF8=1
set PY=C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\Scripts\python.exe
echo [%date% %time%] s15.2 optvol scale fix run start > outputs\s15_2_optvol_runs.status
"%PY%" run_variant.py --variant variants/s15_2_optvol_scale_fix.yaml --no-cache > outputs\s15_2_optvol_run.log 2>&1
echo [%date% %time%] run exit=%errorlevel% >> outputs\s15_2_optvol_runs.status
echo DONE >> outputs\s15_2_optvol_runs.status
