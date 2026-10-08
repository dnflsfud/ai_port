@echo off
rem S25.3 (2026-10-08): generator-pair chain on the 2026-10-02 pull -- old-generator workbook run, then S0' re-certification
rem on the new-generator canonical workbook. Launch pattern: schtasks + cmd /c (decision log S16 ops note); each arm via
rem run_variant_task.ps1 (--no-cache, VINTAGE_PRE/POST). Judge: scripts\eval_s25_3_pair.py (written after the runs; frame S25.3).
cd /d C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_port
echo [%date% %time%] s25_3 chain start > outputs\s25_3_chain.status
powershell -NoProfile -ExecutionPolicy Bypass -File outputs\run_variant_task.ps1 -Label s25_3_oldgen_1002
echo [%date% %time%] oldgen done >> outputs\s25_3_chain.status
powershell -NoProfile -ExecutionPolicy Bypass -File outputs\run_variant_task.ps1 -Label s25_3_s0recert
echo [%date% %time%] s0recert done >> outputs\s25_3_chain.status
echo [%date% %time%] CHAIN_DONE >> outputs\s25_3_chain.status
