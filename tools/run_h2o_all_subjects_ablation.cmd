@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_h2o_all_subjects_ablation.ps1" %*
exit /b %errorlevel%
