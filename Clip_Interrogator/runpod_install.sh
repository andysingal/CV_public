

# Create virtual environment folder
echo "composing venv"

if [ ! -d "venv" ]; then
    python -m venv venv
else
    echo "venv folder already exists, skipping making new venv..."
fi

# Activate virtual environment
source venv/bin/activate

echo "installing necessary libraries"
echo "models will be downloaded when first time run into cache folder"

# Install packages
pip install open_clip_torch

pip3 install torch==2.0.1 torchvision torchaudio -f https://download.pytorch.org/whl/cu118

pip install transformers

pip install bitsandbytes

pip install accelerate

pip install clip-interrogator==0.6.0

pip install gradio

# Show completion message
echo "Virtual environment made and ready to use"
