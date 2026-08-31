@echo off
rem S16.3: degenerate fallback fresh_fixed vs the S16.1 run (same vintage, single delta)
rem
rem Console isolation (see outputs/s16_2_run.bat): `start /WAIT` without /B gives
rem python its own console so a control event on this task's console cannot kill
rem the run (0xC000013A killed two earlier attempts mid Phase 5-6).
rem The redirect must live INSIDE the started command, otherwise it binds to
rem `start` itself and the log ends up empty (observed on the S16.2 retry).
cd /d C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_port
set PYTHONPATH=.
set PYTHONUTF8=1
set PY=C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\Scripts\python.exe
echo [%date% %time%] s16.3 fresh fixed run start > outputs\s16_3_runs.status
start "s16_3_run" /WAIT /MIN cmd /c ""%PY%" run_variant.py --variant variants/s16_3_fresh_fixed.yaml --no-cache > outputs\s16_3_run.log 2>&1"
echo [%date% %time%] run exit=%errorlevel% >> outputs\s16_3_runs.status
echo DONE >> outputs\s16_3_runs.status
