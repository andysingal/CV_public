@echo off

REM Create virtual environment folder
echo composing venv
IF NOT EXIST venv (
    python -m venv venv
) ELSE (
    echo venv folder already exists, skipping making new venv...
)
call .\venv\Scripts\activate.bat

echo installing necessary libraries
echo models will be downloaded when first time run into cache folder
REM Install packages from requirements.txt

pip install open_clip_torch

pip3 install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install transformers

pip install bitsandbytes --prefer-binary --extra-index-url=https://jllllll.github.io/bitsandbytes-windows-webui

pip install accelerate

pip install clip-interrogator==0.6.0

pip install gradio

pip install scipy

REM Show completion message
echo Virtual environment made and ready to use

REM Pause to keep the command prompt open
pause