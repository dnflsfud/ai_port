@echo off
rem S14.2: S0(250) production recert + frozen-overlay paired-replay base (sequential)
cd /d C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_port
set PYTHONPATH=.
set PYTHONUTF8=1
set PY=C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\Scripts\python.exe
echo [%date% %time%] run1 S0(250) production start > outputs\s14_2_runs.status
"%PY%" run_variant.py --variant variants/codex_causal_rank_65.yaml --no-cache > outputs\s14_2_s0_250_run.log 2>&1
echo [%date% %time%] run1 exit=%errorlevel% >> outputs\s14_2_runs.status
echo [%date% %time%] run2 overlay-free base start >> outputs\s14_2_runs.status
"%PY%" run_variant.py --variant variants/s14_overlay_base_250.yaml --no-cache > outputs\s14_2_base_run.log 2>&1
echo [%date% %time%] run2 exit=%errorlevel% >> outputs\s14_2_runs.status
echo DONE >> outputs\s14_2_runs.status
