@echo off
setlocal enabledelayedexpansion
REM ============================================================
REM run_and_upload.bat - env check, tests, S0 backtest, then
REM commit and upload ai_port (standalone repo) to GitHub.
REM Usage: run_and_upload.bat [commit message]
REM Target repo: https://github.com/dnflsfud/ai_port (private)
REM ============================================================

cd /d "%~dp0"
set "PY=C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\Scripts\python.exe"
set "PYTHONPATH=."
set "GH_REPO=dnflsfud/ai_port"
set "GH_URL=https://github.com/dnflsfud/ai_port.git"

echo [1/5] Environment check...
if not exist "%PY%" (echo ERROR: python not found: %PY% & exit /b 1)
"%PY%" -c "import cvxpy; assert 'ECOS' in cvxpy.installed_solvers(), 'ECOS missing'"
if errorlevel 1 (echo ERROR: cvxpy/ECOS check failed & exit /b 1)

echo [2/5] Running test suite...
"%PY%" -m pytest tests/ -q
if errorlevel 1 (echo ERROR: tests failed - aborting before backtest/upload & exit /b 1)

echo [3/5] Running S0 production backtest - full pipeline, about 4 min...
"%PY%" run_variant.py --variant variants\iter15_65tkr_reb21_vtg.yaml --no-cache
if errorlevel 1 (echo ERROR: backtest failed - aborting before upload & exit /b 1)

echo [4/5] Git commit - ai_port standalone repo...
if not exist ".git" (
  git init -b main
  if errorlevel 1 (echo ERROR: git init failed & exit /b 1)
)
git add -A
git diff --cached --quiet
if errorlevel 1 (
  set "MSG=%~1"
  if "!MSG!"=="" set "MSG=run: tests + S0 reproduce %DATE% %TIME%"
  git commit -m "!MSG!"
  if errorlevel 1 (echo ERROR: git commit failed & exit /b 1)
) else (
  echo No changes to commit - skipping commit.
)

echo [5/6] Upload to GitHub...
git remote get-url origin >nul 2>&1
if errorlevel 1 (
  where gh >nul 2>&1
  if not errorlevel 1 (
    gh auth status >nul 2>&1
    if not errorlevel 1 (
      echo Creating private repo %GH_REPO% via gh...
      gh repo create %GH_REPO% --private --source . --remote origin
    )
  )
)
git remote get-url origin >nul 2>&1
if errorlevel 1 (
  echo Adding remote origin %GH_URL% - repo must already exist on GitHub
  git remote add origin %GH_URL%
)
git push -u origin main
if errorlevel 1 (
  echo.
  echo ERROR: push failed. Most likely causes:
  echo   1. Not authenticated: run "gh auth login" once, then re-run this bat.
  echo   2. Repo does not exist yet: run "gh repo create %GH_REPO% --private"
  echo      or create it at https://github.com/new and re-run this bat.
  set "PUSH_FAIL=1"
)

echo [6/6] Launching Streamlit dashboard...
start "ai_port dashboard" cmd /c ""%PY%" -m streamlit run streamlit_app.py"
if defined PUSH_FAIL exit /b 1
echo DONE: run + upload + dashboard complete.
endlocal
