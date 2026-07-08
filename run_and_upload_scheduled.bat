@echo off
setlocal
REM ============================================================
REM run_and_upload_scheduled.bat - non-interactive scheduler
REM entry point. Skips the Streamlit dashboard launch and logs
REM the full run to logs\scheduled_run_last.log (overwritten).
REM ============================================================
set "AI_PORT_NO_DASHBOARD=1"
if not exist "%~dp0logs" mkdir "%~dp0logs"
call "%~dp0run_and_upload.bat" "scheduled: weekday run" > "%~dp0logs\scheduled_run_last.log" 2>&1
exit /b %ERRORLEVEL%
