@echo off
setlocal enabledelayedexpansion
REM ============================================================
REM run_and_upload.bat - env check, tests, S0 + Causal Rank backtests, then
REM commit and upload ai_port (standalone repo) to GitHub.
REM Usage: run_and_upload.bat [commit message]
REM Target repo: https://github.com/dnflsfud/ai_port (private)
REM ============================================================

cd /d "%~dp0"
set "PY=C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\Scripts\python.exe"
set "PYTHONPATH=."
set "GH_REPO=dnflsfud/ai_port"
set "GH_URL=https://github.com/dnflsfud/ai_port.git"

echo [1/10] Environment check...
if not exist "%PY%" (echo ERROR: python not found: %PY% & exit /b 1)
"%PY%" -c "import cvxpy; assert 'ECOS' in cvxpy.installed_solvers(), 'ECOS missing'"
if errorlevel 1 (echo ERROR: cvxpy/ECOS check failed & exit /b 1)

echo [2/10] Running test suite...
"%PY%" -m pytest tests/ -q
if errorlevel 1 (echo ERROR: tests failed - aborting before backtest/upload & exit /b 1)

echo [3/10] Running S0 production backtest - full pipeline, about 4 min...
"%PY%" run_variant.py --variant variants\iter15_65tkr_reb21_vtg.yaml --no-cache
if errorlevel 1 (echo ERROR: backtest failed - aborting before upload & exit /b 1)

echo [4/10] Refreshing S0 operating dashboard data...
"%PY%" scripts\export_operating_data.py
if errorlevel 1 (echo ERROR: S0 operating data export failed - aborting before upload & exit /b 1)

echo [5/10] Running Causal Rank 65 challenger - full pipeline...
"%PY%" run_variant.py --variant variants\codex_causal_rank_65.yaml --no-cache
if errorlevel 1 (echo ERROR: Causal Rank backtest failed - aborting before upload & exit /b 1)

echo [6/10] Refreshing Causal Rank operating dashboard data...
"%PY%" scripts\export_operating_data.py --variant variants\codex_causal_rank_65.yaml --operating-dir outputs\operating_codex_causal_rank_65
if errorlevel 1 (echo ERROR: Causal Rank operating export failed - aborting before upload & exit /b 1)

echo [7/10] Validating both portfolio bundles and publishing registry...
"%PY%" scripts\validate_portfolio_bundles.py --bundle outputs\operating --bundle outputs\operating_codex_causal_rank_65
if errorlevel 1 (echo ERROR: portfolio bundle validation failed - aborting before upload & exit /b 1)

echo [8/10] Git commit - ai_port standalone repo...
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

echo [9/10] Upload to GitHub...
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
git fetch origin main >nul 2>&1
if not errorlevel 1 (
  echo Syncing with remote - git pull --rebase origin main...
  git pull --rebase origin main
  if errorlevel 1 (
    echo ERROR: rebase conflict - resolve manually, push skipped
    git rebase --abort
    set "PUSH_FAIL=1"
  )
)
if not defined PUSH_FAIL (
  git push -u origin main
  if errorlevel 1 (
    echo.
    echo ERROR: push failed. Most likely causes:
    echo   1. Not authenticated: run "gh auth login" once, then re-run this bat.
    echo   2. Repo does not exist yet: run "gh repo create %GH_REPO% --private"
    echo      or create it at https://github.com/new and re-run this bat.
    set "PUSH_FAIL=1"
  )
)

if defined AI_PORT_NO_DASHBOARD (
  echo [10/10] Dashboard launch skipped ^(AI_PORT_NO_DASHBOARD^).
) else (
  echo [10/10] Launching Streamlit dashboard...
  start "ai_port dashboard" cmd /c ""%PY%" -m streamlit run streamlit_app.py"
)
if defined PUSH_FAIL exit /b 1
echo DONE: run + upload + dashboard complete.
endlocal
