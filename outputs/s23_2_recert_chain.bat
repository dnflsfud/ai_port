@echo off
rem S23.2 combined re-certification (2026-09-29): production after the M-01 + B-01 flips on the frozen
rem S23.1 vintage (workbook 2026-09-18 14:13:24 / Index 2026-09-28 11:23:35). Judge: scripts\eval_s23_arm.py --arm s23_combined_recert
cd /d C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_port
echo [%date% %time%] s23_2 recert start > outputs\s23_2_chain.status
powershell -NoProfile -ExecutionPolicy Bypass -File outputs\run_variant_task.ps1 -Label s23_combined_recert
echo [%date% %time%] DONE >> outputs\s23_2_chain.status
