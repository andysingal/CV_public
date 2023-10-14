#!/bin/bash

# Create a virtual environment folder
echo "Composing venv"
if [ ! -d "venv" ]; then
    python -m venv venv
else
    echo "venv folder already exists, skipping making a new venv..."
fi

# Activate the virtual environment
source venv/bin/activate

# Install necessary libraries
echo "Installing necessary libraries"
echo "Models will be downloaded when first run into cache folder"

# Install packages from requirements.txt (if you have one)
# You can replace this with your actual requirements.txt file
# pip install -r requirements.txt

pip install open_clip_torch
pip3 install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install bitsandbytes --prefer-binary --extra-index-url=https://jllllll.github.io/bitsandbytes-windows-webui
pip install accelerate
pip install clip-interrogator==0.6.0
pip install gradio
pip install scipy

# Show completion message
echo "Virtual environment made and ready to use"

# Deactivate the virtual environment
deactivate

# Pause to keep the terminal open (optional)
read -p "Press Enter to exit..."

