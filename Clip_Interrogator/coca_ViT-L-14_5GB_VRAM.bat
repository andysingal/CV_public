@echo off
REM Change the path below to the directory containing your virtual environment
set VENV_PATH=venv

REM Activate the virtual environment
call %VENV_PATH%\Scripts\activate.bat

REM Execute the Python script
python coca_ViT-L-14_5GB_VRAM.py

pause