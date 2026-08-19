# GAT for genotype-phenotype and drug side-effect modelling

This project builds a heterogeneous graph attention model over genetic, phenotypic, ancestry, drug, and side-effect relations, with a domain-adversarial ancestry branch and multi-task edge prediction.

The current implementation focuses on:
- variant-to-phenotype association prediction
- drug-to-side-effect prediction
- ancestry-aware evaluation for out-of-distribution testing

---

## Prerequisites

- Conda or Mamba
- CUDA-capable GPU recommended
- Linux-based HPC environment if using the provided Slurm scripts

---

## Installation

Clone the repository and set up the environment using the provided configuration file:

```bash
# Clone the repository
git clone [git@github.com:Lekanville/gat-modelling-for-genotype-phenotype-associations.git](https://github.com/Lekanville/gat-modelling-for-genotype-phenotype-associations.git)
cd gat-modelling-for-genotype-phenotype-associations
```

## Environment setup

Create and activate the environment defined in [environment.yml](environment.yml):

```bash
conda env create -f environment.yml
conda activate gat_env
```

---

## Repository structure

```text
GAT/
├── code/
│   ├── modelling.py
│   └── tools/
│       ├── create_edges.py
│       ├── create_nodes.py
│       ├── hgt_gat.py
│       ├── inference_utils.py
│       ├── load_edge_data.py
│       ├── load_node_data.py
│       ├── train_utils.py
│       └── viz.py
│  
├── data/
│   ├── edges/
│   └── nodes/
├── output/
├── .gitignore
├── environment.yml
├── gat_run.sh
├── gat_run_anc.sh
├── check.sh
├── README.md
└── test.sh
```

---

## Data and graph construction

The pipeline reads node tables and edge tables from the `data/` directory and builds a heterogeneous graph with node types such as:
- variant
- gene
- phenotype
- ancestry
- tissue
- pathway
- drug
- side_effect

Important modelling decisions:
- repeated `variant-phenotype` rows are collapsed to one canonical edge per `(variant, phenotype)` pair using the strongest observed magnitude
- this prevents ancestry-specific GWAS duplicates from inflating the graph and leaking across train/validation/test splits
- ancestry-aware hold-out is applied conservatively: tiny ancestry groups are kept in training unless they are large enough to support a reliable validation/test split

---

## Training modes

### 1. Random split mode
open gat_run.sh

Use this when no ancestry hold-out is requested:

```bash
python code/modelling.py \
  --input_directory data \
  --ancestry_test unspecified \
  --ancestry_val unspecified \
  --output_directory output/output_random
```

### 2. Ancestry hold-out mode

Use this for leave-one-ancestry-out evaluation:

```bash
python code/modelling.py \
  --input_directory data \
  --ancestry_test EAS \
  --ancestry_val GME \
  --output_directory output_anc
```
Possible ancestry codes are EAS, OCE, SAS, EUR, AMR, GME, and AFR

---

## Slurm execution

The repository includes example job submission files:

```bash
sbatch gat_run.sh
```

The ancestry-specific script is intended for ancestry-aware out-of-distribution evaluation and uses the same model training pipeline with ancestry hold-out masks for variant-phenotype edges.

---

## Model summary

The script trains a heterogeneous graph neural network with:
- HGT encoder
- relation-specific link scorers
- multi-task training for variant-phenotype and drug-side-effect relations
- domain adversarial ancestry branch using phenotype ancestry distributions
- early stopping based on validation loss

It reports training/validation/test metrics such as:
- AUROC
- average precision (AP)
- task-specific losses

---

## Notes

- The project is currently focused on drug side-effect prediction as the primary supervised task.
- Variant-to-phenotype prediction is included but is generally more sparse and more sensitive to data quality and split design.
- Output folders such as `output/` and `output_anc/` are generated during runs and are intended to contain logs, visualizations, and checkpoints.

---

## Reproducibility

To recreate the environment locally:

```bash
conda env create -f environment.yml
conda activate gat_env
```

Then run the appropriate script or Python command depending on whether you want random splits or ancestry hold-out evaluation.
