@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_h2o_megapose_ablation.ps1" %*
exit /b %errorlevel%
