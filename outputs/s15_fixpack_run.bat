@echo off
rem S15: fix-pack correctness measurement run (production + s15_fixpack_enabled)
cd /d C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_port
set PYTHONPATH=.
set PYTHONUTF8=1
set PY=C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\Scripts\python.exe
echo [%date% %time%] s15 fixpack run start > outputs\s15_fixpack_runs.status
"%PY%" run_variant.py --variant variants/s15_fixpack.yaml --no-cache > outputs\s15_fixpack_run.log 2>&1
echo [%date% %time%] run exit=%errorlevel% >> outputs\s15_fixpack_runs.status
echo DONE >> outputs\s15_fixpack_runs.status
