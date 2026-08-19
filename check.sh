#!/bin/bash
#SBATCH --job-name=tensor_gpu_check         # Name of the job
#SBATCH --gres=gpu:1                        # Request 1 GPU
#SBATCH --ntasks=1                          # Run a single task
#SBATCH --cpus-per-task=1                   # Request 1 CPU core
#SBATCH --mem=4G                            # Request 4GB of RAM
#SBATCH --time=00:05:00                      # Max runtime (5 minutes)
#SBATCH --output=tensor_gpu_check_%j.log     # Standard output and error log

# --- 1. Load Environment (Uncomment the one you use) ---
module load cuda/12.6

export CUDA_HOME=$CUDA_DIR
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

source ~/miniforge3/bin/activate gat

# Check the active Nvidia driver capability
nvidia-smi

# Check if a CUDA module is loaded on your HPC cluster
module list

# Extract your true PyTorch version string (e.g., 2.3.0 or 2.1.2)
export TORCH=$(python -c "import torch; print(torch.__version__.split('+')[0])")
export CUDA="cu126"

# Verify what it resolved to
echo "Targeting PyTorch version: $TORCH with CUDA: $CUDA"
# pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.13.0+cu126.html

python -c "import torch_geometric; print('PyG Version:', torch_geometric.__version__)"