@echo off
setlocal
set "PY=C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\Scripts\python.exe"
REM D-probe 4: run_and_upload.bat pattern `"%PY%" step.py` + `if errorlevel 1 (abort)`
REM against a native crash exit status (0xC0000005 access violation) and a normal failure.
"%PY%" -c "import ctypes; ctypes.windll.kernel32.ExitProcess(0xC0000005)"
echo native-crash exit ERRORLEVEL=%ERRORLEVEL%
if errorlevel 1 (echo   bat pattern: ABORT) else (echo   bat pattern: CONTINUES as success)
"%PY%" -c "import sys; sys.exit(1)"
echo python sys.exit(1) ERRORLEVEL=%ERRORLEVEL%
if errorlevel 1 (echo   bat pattern: ABORT) else (echo   bat pattern: CONTINUES as success)
endlocal
