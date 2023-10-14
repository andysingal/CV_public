#!/bin/bash

# Change the path below to the directory containing your virtual environment
VENV_PATH="venv"

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# Execute the Python script
python half_precision_17_GB_VRAM.py

# Pause (optional)
read -p "Press Enter to exit..."
