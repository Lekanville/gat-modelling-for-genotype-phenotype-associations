# GAT for genotype-phenotype and drug side-effect modelling

This project builds a heterogeneous graph attention model over genetic, phenotypic, ancestry, drug, and side-effect relations, with a domain-adversarial ancestry branch and multi-task edge prediction.

The current implementation focuses on:
- variant-to-phenotype association prediction
- drug-to-side-effect prediction
- ancestry-aware out-of-distribution evaluation
- source-aware heterogeneous link ranking with per-source candidate pools
- fair negative sampling diagnostics and score/ROC visualisation for supervised edge prediction

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
├── gat_run_drug_phen.sh
├── gat_run_var_phen_anc.sh
├── gat_run_var_phen_random.sh
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
- for drug-side-effect prediction, edges are split by source drug so all edges from the same drug remain in a single fold; the script also reports the number of unique drugs in the train, validation, and test splits to make the partitioning transparent

---

## Training modes

### 1. Variant-Phenotype Prediction using random split mode

Use this when performing variant-to-phenotype predictions with no specific ancestry hold-out.

```bash
python code/modelling.py \
  --input_directory data \
  --ancestry_test unspecified \
  --ancestry_val unspecified \
  --prediction_type variant_phenotype \
  --output_directory output/output_random \
  --focus_test_relation False \
  --negative_sampling_mode source_aware
```
Example scripts are provided in `gat_run_var_phen_random.sh`.

### 2. Variant-Phenotype Prediction with ancestry hold-out

Use this when performing ancestry-aware out-of-distribution evaluation.

```bash
python code/modelling.py \
  --input_directory data/T2D_and_Alzheimer \
  --ancestry_test AMR \
  --ancestry_val EAS \
  --prediction_type variant_phenotype \
  --output_directory output/output_var_phen_anc_focus \
  --focus_test_relation False \
  --negative_sampling_mode source_aware
```
Possible ancestry codes are typically EAS, OCE, SAS, EUR, AMR, GME, and AFR depending on the dataset.
Example script: `gat_run_var_phen_anc.sh`.

### 3. Drug-causes-side-effect prediction

Use this mode for drug-to-outcome prediction.

```bash
python code/modelling.py \
  --input_directory data \
  --ancestry_test unspecified \
  --ancestry_val unspecified \
  --prediction_type drug_causes_side_effect \
  --output_directory output/output_drug_causes_phen \
  --focus_test_relation False \
  --negative_sampling_mode source_aware
```
Example script: `gat_run_drug_causes_phen.sh`.

### Supported prediction types

The script supports the following modes:

```bash
--prediction_type multi_task
--prediction_type variant_phenotype
--prediction_type drug_treats_phenotype
--prediction_type drug_causes_side_effect
```

The relevant supervised relation for the drug-causes task is the `('drug', 'causes', 'clinical_outcome')` edge family.

---

## Command-line options

The main CLI arguments are:

```bash
--input_directory      Directory containing node and edge data
--ancestry_test        Test ancestry label for ancestry-aware evaluation
--ancestry_val         Validation ancestry label for ancestry-aware evaluation
--prediction_type      Model objective: multi_task, variant_phenotype, drug_treats_phenotype, or drug_causes_side_effect
--output_directory     Directory for logs, checkpoints, and diagnostics
--ignore_relations     Optional relation tuples to remove before training, e.g.
                       "drug,treats,clinical_outcome"
                       or "drug,treats,clinical_outcome|gene,associated_with,clinical_outcome"
--focus_test_relation  When set to true, restrict evaluation to one validation and test edge per source
--negative_sampling_mode Choose the negative pool strategy:
                       source_aware  (default, recommended)
                       global        (full global destination pool for comparison)
```

### `--ignore_relations`

This removes edge types from the heterogeneous graph before training. It is useful for ablation studies when checking whether a particular relation is creating leakage or dominating the signal.

### `--focus_test_relation`

This is a leakage-control / diagnostic setting. When enabled, validation and test scoring is restricted to the single relation implied by the target prediction task instead of evaluating all supervised relations jointly.

### `--negative_sampling_mode`

This controls how negative samples are drawn:

- `source_aware`: for each source node, build a candidate destination pool, remove its observed positives, and sample non-linked destinations from that pool. This is the recommended and default configuration.
- `global`: sample negatives from the full destination set for the relation. This is useful for ablation comparisons, but it often makes the ranking task easier and can inflate AUROC/AP.

---

## Split logic and evaluation protocol

The model uses source-aware train/validation/test masks for the supervised relations.

For the ancestry-aware variant-phenotype setup:
- `ancestry_aware_split()` creates train/validation/test masks based on ancestry groups
- validation and test sets are explicitly held out from training
- the same split logic is used consistently for all downstream scoring and diagnostics

For source-based drug splits:
- each source drug is assigned to a fold as a unit
- all edges from the same drug remain in a single split
- the script prints the number of unique sources in each split for transparency

This is intended to prevent obvious leakage through repeated source-level edge patterns.

---

## Evaluation setup and why negative sampling is source-aware

The project treats the supervised tasks as link-ranking problems rather than one global binary classification over all possible edges.

For each supervised relation and source node:
- positives are the held-out true links for that source
- negatives are non-linked destination candidates for that same source
- the model ranks the candidate set
- AUROC and average precision are computed from that ranking

This is important because a global negative pool can be extremely large and artificially easy. In contrast, a source-aware candidate pool is a more faithful approximation to the real prediction task.

The code logs compact summary statistics for the candidate pool, such as:
- number of sources in the split
- average candidate pool size
- min/max candidate pool size
- sampled negatives per source
- mean negatives per source
- whether the mode is `source_aware` or `global`

---

## Diagnostics generated during runs

The training pipeline saves and prints diagnostics to help understand whether the model is learning genuine structure or exploiting the graph topology.

These include:
- positive-vs-negative score histograms for validation/test splits
- ROC curves for supervised relations
- AUROC and AP metrics for train/validation/test
- confusion-matrix summaries
- source-aware split counts for train/validation/test
- route-attention summaries over HGT paths
- graph audit plots for node and edge counts

The histogram and ROC plots are especially useful for understanding whether the positive links are separated from sampled negatives in a realistic ranking setup.

---

## Slurm execution

The repository includes example job submission files such as:

```bash
sbatch gat_run_var_phen_anc.sh
sbatch gat_run_drug_causes_phen.sh
```

These scripts activate the environment, set the required CUDA variables, and call the Python training pipeline with the appropriate prediction type and negative-sampling configuration.

---

## Model summary

The script trains a heterogeneous graph neural network with:
- HGT encoder
- relation-specific link scorers
- multi-task or single-task training depending on `--prediction_type`
- source-aware or global negative sampling depending on `--negative_sampling_mode`
- split-aware train/validation/test evaluation
- early stopping based on validation loss
- domain adversarial ancestry branch for variant-phenotype tasks

It reports metrics such as:
- AUROC
- average precision (AP)
- losses for each supervised relation
- confusion-matrix summaries
- unique source counts per train/validation/test split
- route-attention summaries over the HGT message-passing paths

---

## Notes

- The project is currently focused on supervised heterogeneous link prediction tasks with source-aware candidate pools and fairness-aware negative sampling.
- Variant-to-phenotype prediction remains useful for ancestry-aware generalization studies but is typically more sensitive to graph sparsity and split design.
- Drug-causes-side-effect prediction is a central supervised task and is heavily monitored through split counts, candidate-pool summaries, and ranking diagnostics.
- Output folders such as `output/` and `output_anc/` are generated during runs and contain logs, visualizations, score plots, and checkpoint files.

---

## Reproducibility

To recreate the environment locally:

```bash
conda env create -f environment.yml
conda activate gat_env
```

Then run the appropriate script or Python command depending on whether you want random splits or ancestry hold-out evaluation.
