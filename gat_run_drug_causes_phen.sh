#!/bin/bash

#SBATCH --job-name=GAT_RUN_DRUG_CAUSES_PHEN
#SBATCH --output=GAT_RUN_DRUG_CAUSES_PHEN.log
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00

# --- CRITICAL CLUSTER DRIVERS ---
module load cuda/12.6

# --- ACTIVATE ENVIRONMENT ---
source ~/miniforge3/bin/activate gat

# # --- CRITICAL HPC STABILITY FIXES ---
# # FIX 1 & 2: Direct joblib/loky temporary files to local storage ($TMPDIR)
export JOBLIB_TEMP_FOLDER=$TMPDIR
export LOKY_TEMP_FOLDER=$TMPDIR

# # FIX 3: Force Python multiprocessing start method
export PYTHON_START_METHOD='forkserver'

# # FIX 4 (AGGRESSIVE): Explicitly limit all parallel operations to a single thread/core.
# # This is the final step to bypass the hard system limit on IPC resources (semaphores/locks).
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1

# # CRITICAL LOKY BYPASS: Explicitly tell the joblib/loky backend to use only 1 CPU.
# export LOKY_MAX_CPU_COUNT=1

# # GPU Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TF_CPP_MIN_LOG_LEVEL=1

srun python code/modelling.py --input_directory data/T2D_and_Alzheimer \
--ancestry_val unspecified \
--ancestry_test unspecified \
--prediction_type drug_causes_side_effect \
--output_directory output/output_drug_causes_phen \
--focus_test_relation False \
--negative_sampling_mode source_aware \

# --ignore_relations "drug,treats,clinical_outcome" \