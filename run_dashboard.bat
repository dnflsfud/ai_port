@echo off
REM ============================================================================
REM  run_dashboard.bat
REM  Raw workbook (ai_signal_data.xlsx) -> full Pictet pipeline -> Streamlit
REM
REM  Pipeline (each step must succeed before the next runs):
REM    [1] run_pictet_adoption.py    S0 + attribution + overlay + factor + DSR
REM                                  -> outputs/adoption_summary.json (+ stages)
REM    [2] scripts/data_quality_report.py  -> outputs/data_quality_report.json
REM    [3] scripts/export_operating_data.py-> outputs/operating/*.json + returns.csv
REM    [4] streamlit run streamlit_app.py  -> dashboard (blocks until Ctrl+C)
REM
REM  The raw workbook path is read from src/config.py (data_path); nothing is
REM  hardcoded here. The first full recompute is heavy (~20-40 min); later runs
REM  reuse cached backtests and are faster.
REM
REM  To only RE-OPEN the dashboard on already-computed outputs, skip this and run:
REM    python -m streamlit run streamlit_app.py
REM ============================================================================
setlocal

set "PY=C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\Scripts\python.exe"

REM Project root = folder of this .bat (strip trailing backslash)
set "PROJ=%~dp0"
if "%PROJ:~-1%"=="\" set "PROJ=%PROJ:~0,-1%"

cd /d "%PROJ%"
set "PYTHONPATH=%PROJ%"

echo ============================================================
echo  Pictet portfolio dashboard build
echo  Project : %PROJ%
echo  Python  : %PY%
echo ============================================================

if not exist "%PY%" (
    echo [ERROR] venv python not found: %PY%
    goto :fail
)

echo.
echo [0/4] Checking raw data workbook (path from src/config.py)...
"%PY%" -c "import os,sys; from src.config import PipelineConfig as C; p=C().data_path; print('  data_path =', p); sys.exit(0 if os.path.exists(p) else 3)"
if errorlevel 1 (
    echo [ERROR] Raw workbook missing or src import failed - see path above. Fix src/config.py data_path.
    goto :fail
)

echo.
echo [1/4] run_pictet_adoption.py  (S0 + attribution + overlay + factor + DSR)
echo       Recomputing from the raw workbook - first run can take 20-40 min...
"%PY%" run_pictet_adoption.py
if errorlevel 1 (
    echo [ERROR] run_pictet_adoption.py failed.
    goto :fail
)

echo.
echo [2/4] scripts/data_quality_report.py
"%PY%" scripts\data_quality_report.py
if errorlevel 1 (
    echo [ERROR] data_quality_report.py failed.
    goto :fail
)

echo.
echo [3/4] scripts/export_operating_data.py
"%PY%" scripts\export_operating_data.py
if errorlevel 1 (
    echo [ERROR] export_operating_data.py failed.
    goto :fail
)

echo.
echo [4/4] Launching Streamlit dashboard  (Ctrl+C in this window to stop)...
"%PY%" -m streamlit run streamlit_app.py
if errorlevel 1 (
    echo [ERROR] Streamlit failed to launch.
    goto :fail
)

echo.
echo Done.
endlocal
exit /b 0

:fail
echo.
echo Build aborted - see the error above.
pause
endlocal
exit /b 1
