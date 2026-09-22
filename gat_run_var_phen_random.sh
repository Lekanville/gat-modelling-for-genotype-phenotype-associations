#!/bin/bash

#SBATCH --job-name=GAT_RUN_VAR_PHEN_RANDOM
#SBATCH --output=GAT_RUN_VAR_PHEN_RANDOM.log
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

srun python code/modelling.py --input_directory data \
--ancestry_val unspecified \
--ancestry_test unspecified \
--prediction_type variant_phenotype \
--output_directory output/output_var_phen_random \
--focus_test_relation False \
--negative_sampling_mode source_aware \
--ignore_relations 'gene,interacts_with,gene|variant,maps_to,gene|drug,causes,clinical_outcome|drug,treats,clinical_outcome|
ancestry,prevalent_in,clinical_outcome|clinical_outcome,lin_similarity_with,clinical_outcome|
clinical_outcome,genetically_correlated,clinical_outcome|drug,targets,gene|gene,belongs_to,pathway|
gene,associated_with,clinical_outcome|gene,expressed_in,tissue|tissue,enriched_for,clinical_outcome|
variant,observed_in,ancestry'

# --ignore_relations "drug,treats,clinical_outcome" \