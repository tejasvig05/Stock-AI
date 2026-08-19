@echo off
REM run_pipeline.bat
REM Wrapper script for Windows Task Scheduler. Using a .bat file instead of
REM pointing Task Scheduler directly at python.exe is more reliable --
REM it guarantees the working directory is set correctly and gives a
REM single, simple target to configure in Task Scheduler.

cd /d C:\Users\91769\stock_ai_project
C:\Users\91769\anaconda3\envs\stockai\python.exe src\run_daily_pipeline.py

REM Exit code from the python script passes through automatically,
REM so Task Scheduler can detect success/failure if needed.
