#!/bin/bash
#SBATCH --job-name=62
#SBATCH --output=PT_62_%j.out
#SBATCH --error=PT_62_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=main

# Load necessary modules
module load python/3.10.13
module load cuda/11.8.0--gcc-11.4.0-l7ay6m6
module load cudnn/8.6.0.163-11.8--gcc-11.4.0-jc33sc5

# Define the virtual environment directory
VENV_DIR="$HOME/CDD-CESM/class/myenv"
# Define the directory on the server
SCRIPT_DIR="$HOME/CDD-CESM/class"

# Function to setup virtual environment
setup_venv() {
    echo "Setting up virtual environment..."
    python3 -m venv $VENV_DIR
    source $VENV_DIR/bin/activate
    pip install --upgrade pip setuptools wheel
    pip install pybind11
    pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
    python -c "import torch; print('GPU available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
    pip install "numpy<2.0" pandas scikit-learn matplotlib
    pip install opencv-python-headless
    pip install fvcore pyyaml cloudpickle omegaconf
    pip install timm
    pip install pycocotools
    pip install torch-lr-finder
    pip install albumentations
    pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'
    pip install pytorch-lightning
    pip install tensorboard
    pip install pillow  # Make sure we have the latest Pillow
    echo "Virtual environment setup complete."
}

# Function to run the train script
run_train_script() {
    echo "Running train script..."
    source $VENV_DIR/bin/activate
    export PYTHONPATH=$PYTHONPATH:$VENV_DIR/lib/python3.10/site-packages
    export CUDA_VISIBLE_DEVICES=0  # Use the first GPU
    if [ -f "$SCRIPT_DIR/train_62.py" ]; then
        python $SCRIPT_DIR/train_62.py
    else
        echo "Error: $SCRIPT_DIR/train_torch.py not found."
    fi
}

# Main script execution
setup_venv
run_train_script
