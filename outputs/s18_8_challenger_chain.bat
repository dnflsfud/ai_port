@echo off
rem S18.8 challenger calendar alignment chain (2026-09-14, user-approved). Re-run the Legacy S0 challenger
rem (iter15_65tkr_reb21_vtg, now business_day_calendar_enabled: true) on the current vintage pair
rem (workbook 2026-09-11 13:34:45 / Index 2026-09-14 11:00:57 -- VINTAGE_PRE/POST in
rem outputs\iter15_65tkr_reb21_vtg_run.status), refresh its operating bundle (outputs\operating), then
rem rebuild the two-bundle registry that aborted the 2026-09-14 scheduled run
rem ("portfolio last_rebalance_date mismatch: ['2026-08-18', '2026-09-04']"). The production bundle is NOT
rem re-run: its 2026-09-14 12:02 run already reproduces S0' 1.7512 bit-exactly. Judge: decision log S18.8 C1-C4.
rem Launch pattern: schtasks ai_port_s18_8_chain + cmd /c (see decision log S16 ops note).
cd /d C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_port
set "PY=C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\Scripts\python.exe"
set "PYTHONPATH=."
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
echo [%date% %time%] s18_8 chain start > outputs\s18_8_chain.status
echo [%date% %time%] start backtest iter15_65tkr_reb21_vtg >> outputs\s18_8_chain.status
powershell -NoProfile -ExecutionPolicy Bypass -File outputs\run_variant_task.ps1 -Label iter15_65tkr_reb21_vtg
echo [%date% %time%] end backtest >> outputs\s18_8_chain.status
echo [%date% %time%] start export outputs\operating >> outputs\s18_8_chain.status
"%PY%" scripts\export_operating_data.py > outputs\s18_8_export.log 2>&1
echo [%date% %time%] export EXIT %errorlevel% >> outputs\s18_8_chain.status
echo [%date% %time%] start validate >> outputs\s18_8_chain.status
"%PY%" scripts\validate_portfolio_bundles.py --bundle outputs\operating --bundle outputs\operating_codex_causal_rank_65 > outputs\s18_8_validate.log 2>&1
echo [%date% %time%] validate EXIT %errorlevel% >> outputs\s18_8_chain.status
echo [%date% %time%] DONE >> outputs\s18_8_chain.status
