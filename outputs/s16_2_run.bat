@echo off
rem S16.2: revision extension cap 21BD vs the S16.1 run (same vintage, single delta)
rem
rem Attempt 1 (13:16, plain "%PY%" ... in this console) died in Phase 5-6 with
rem 0xC000013A = STATUS_CONTROL_C_EXIT, the same mode that killed the first
rem S16.1 attempt and that the decision log records for S13.27. `start /WAIT`
rem without /B gives python its OWN console, so a control event delivered to
rem this task's console cannot propagate into the run.
cd /d C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_port
set PYTHONPATH=.
set PYTHONUTF8=1
set PY=C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\Scripts\python.exe
echo [%date% %time%] s16.2 revision extension cap run start > outputs\s16_2_runs.status
start "s16_2_run" /WAIT /MIN "%PY%" run_variant.py --variant variants/s16_2_revision_extension_cap.yaml --no-cache > outputs\s16_2_run.log 2>&1
echo [%date% %time%] run exit=%errorlevel% >> outputs\s16_2_runs.status
echo DONE >> outputs\s16_2_runs.status
