@echo off
setlocal
REM ============================================================
REM run_and_upload_scheduled.bat - non-interactive scheduler
REM entry point. Skips the Streamlit dashboard launch and logs
REM the full run to logs\scheduled_run_<timestamp>.log, copies
REM it to logs\scheduled_run_last.log (unchanged path for
REM readers) and appends one status line per run to
REM logs\scheduled_run_history.log (S22 D-09: failed runs used
REM to overwrite each other, leaving no failure history).
REM ============================================================
set "AI_PORT_NO_DASHBOARD=1"
if not exist "%~dp0logs" mkdir "%~dp0logs"
set "RUN_TS="
for /f "delims=" %%T in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "RUN_TS=%%T"
if not defined RUN_TS set "RUN_TS=unknown_%RANDOM%"
set "RUN_LOG=%~dp0logs\scheduled_run_%RUN_TS%.log"
call "%~dp0run_and_upload.bat" "scheduled: weekday run" > "%RUN_LOG%" 2>&1
set "RC=%ERRORLEVEL%"
copy /y "%RUN_LOG%" "%~dp0logs\scheduled_run_last.log" >nul
>> "%~dp0logs\scheduled_run_history.log" echo %RUN_TS% exit=%RC% log=scheduled_run_%RUN_TS%.log
exit /b %RC%
