import torch
import random
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from tools.load_node_data import node_data
from tools.load_edge_data import edge_data
from tools.create_nodes import nodes_init
from tools.create_edges import edges_init
from tools.viz import (audit_and_visualize_heterodata, plot_training_history)
from tools.train_utils import (
    deduplicate_variant_phenotype_edges,
    deduplicate_drug_side_effect_edges,
    relation_batch,
    edge_loss_and_metrics,
)
from tools.hgt_gat import split_mask, split_by_source_node, ancestry_aware_split, HetModel
from tools.inference_utils import (
    topk_pheno_for_variant,
    topk_pheno_for_drug,
    topk_side_effect_for_drug,
    polypharmacy_risk,
)


parser = argparse.ArgumentParser(description= "A script to filter data")
parser.add_argument('-i', '--input_directory', type=str, required=True, help= 'The input datasets folder')
parser.add_argument('-j', '--ancestry_test', type=str, required=True, help= 'The ancestry to hold out for testing')
parser.add_argument('-k', '--ancestry_val', type=str, required=True, help= 'The ancestry to hold out for validation')
parser.add_argument('-o', '--output_directory', type=str, required=True, help= 'The output directory to save results')
parser.add_argument(
    '--prediction_type',
    type=str,
    default='multi_task',
    choices=['multi_task', 'variant_phenotype', 'drug_side_effect'],
    help='Select whether to train/evaluate variant-phenotype, drug-side-effect, or both tasks.'
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
random.seed(42) 


def gat_modelling(INPUT, ANCESTRY_TEST, ANCESTRY_VAL, OUTPUT, PREDICTION_TYPE):

    Path(OUTPUT).mkdir(parents=True, exist_ok=True)

    # The nodes data
    (df_variants, df_tissues, df_genes, df_ancestry, df_phenotype, df_meddra, df_pathway, df_drugs) = node_data(INPUT)

    # The edge data 
    (df_ancestry_phenotype, df_gene_gene, df_variant_gene, df_drug_phenotype_treats, df_drug_phenotype_causes, df_drug_gene, df_gene_pathway, 
        df_gene_phenotype, df_gene_tissue, df_phenotype_phenotype_lin, df_phenotypes_phenotypes_ldsc, df_tissue_phenotype,
            df_variant_ancestry, df_variant_phenotype) = edge_data(INPUT)

    df_variant_phenotype = deduplicate_variant_phenotype_edges(df_variant_phenotype)
    df_drug_phenotype_causes = deduplicate_drug_side_effect_edges(df_drug_phenotype_causes)

    edge_dfs = (df_ancestry_phenotype, df_gene_gene, df_variant_gene, df_drug_phenotype_treats, df_drug_phenotype_causes, df_drug_gene, df_gene_pathway, 
        df_gene_phenotype, df_gene_tissue, df_phenotype_phenotype_lin, df_phenotypes_phenotypes_ldsc, df_tissue_phenotype,
            df_variant_ancestry, df_variant_phenotype) 


    # Initialize Features & ID Mappings 
    data = HeteroData()

    #Bbuilding the nides
    (variant_to_idx, tissue_to_idx, gene_to_idx, ancestry_to_idx, 
        pheno_to_idx, meddra_to_idx, pathway_to_idx, drug_to_idx, data) = nodes_init(data, df_variants, df_tissues, df_genes, 
                                                                   df_ancestry, df_phenotype, df_meddra, df_pathway, df_drugs)
    
    mappings = (variant_to_idx, tissue_to_idx, gene_to_idx, ancestry_to_idx, 
        pheno_to_idx, meddra_to_idx, pathway_to_idx, drug_to_idx)
    
    # Building the edges
    data = edges_init(data, edge_dfs, mappings)
    print("All 8 Node Feature Matrices successfully loaded!")
    audit_and_visualize_heterodata(data, OUTPUT)

    # Build phenotype ancestry soft-label distributions from variant-phenotype edges
    # Each phenotype gets a probability vector over ancestry domains (sum=1).
    (_, _, _, ancestry_to_idx, pheno_to_idx, _, _, _) = mappings
    num_pheno = len(pheno_to_idx)
    num_ancestry = len(ancestry_to_idx)

    # Count occurrences of ancestries per phenotype using per-variant edges
    pheno_anc_counts = defaultdict(lambda: [0] * num_ancestry)
    if 'target_ancestry' in df_variant_phenotype.columns:
        for _, row in df_variant_phenotype.iterrows():
            pheno = row.get('target_phenotype') or row.get('target_hpo') or row.get('trait')
            anc = row.get('target_ancestry')
            if pd.isna(pheno) or pd.isna(anc):
                continue
            anc_code = str(anc).replace('ANC_', '')
            if pheno in pheno_to_idx and anc_code in ancestry_to_idx:
                ph_idx = pheno_to_idx[pheno]
                anc_idx = ancestry_to_idx[anc_code]
                pheno_anc_counts[ph_idx][anc_idx] += 1

    # Convert counts to probability distributions; use uniform prior when no observations
    pheno_dist = np.zeros((num_pheno, num_ancestry), dtype=np.float32)
    for p_idx in range(num_pheno):
        counts = np.array(pheno_anc_counts.get(p_idx, [0] * num_ancestry), dtype=np.float32)
        s = counts.sum()
        if s == 0:
            pheno_dist[p_idx] = np.ones(num_ancestry, dtype=np.float32) / float(num_ancestry)
        else:
            pheno_dist[p_idx] = counts / s

    # Attach soft-label distributions to HeteroData for domain adversary training
    data['phenotype'].ancestry_dist = torch.tensor(pheno_dist, dtype=torch.float32)

    rel_variant = ('variant', 'associated_with', 'phenotype')
    rel_drug = ('drug', 'causes', 'side_effect')
    relation_alias = {
        rel_variant: 'variant_phenotype',
        rel_drug: 'drug_side_effect',
    }
    relation_name = {
        rel_variant: 'variant-phenotype',
        rel_drug: 'drug-side-effect',
    }

    if PREDICTION_TYPE == 'variant_phenotype':
        supervised_relations = [rel_variant]
    elif PREDICTION_TYPE == 'drug_side_effect':
        supervised_relations = [rel_drug]
    else:
        supervised_relations = [rel_variant, rel_drug]

    print(f"Prediction type: {PREDICTION_TYPE}")
    print(f"Supervised relations: {[relation_name[r] for r in supervised_relations]}")
    drug_split_counts = {}

    if (ANCESTRY_TEST == 'unspecified') and (ANCESTRY_VAL == 'unspecified'):
        # Random splitting if no ancestry is specified
        print("No ancestry specified for test or validation. Using random splits.")
        for rel in supervised_relations:
            if rel in data.edge_types:
                if rel == rel_drug:
                    tr, va, te = split_by_source_node(data[rel].edge_index)
                    drug_split_counts = {
                        'train_unique_drugs': int(torch.unique(data[rel].edge_index[0, tr]).numel()),
                        'val_unique_drugs': int(torch.unique(data[rel].edge_index[0, va]).numel()),
                        'test_unique_drugs': int(torch.unique(data[rel].edge_index[0, te]).numel()),
                    }
                    print(
                        "Drug-side-effect source-stratified masks: "
                        f"train={int(tr.sum())}, val={int(va.sum())}, test={int(te.sum())}"
                    )
                    print(
                        "Drug-side-effect split drug counts: "
                        f"train={drug_split_counts['train_unique_drugs']}, "
                        f"val={drug_split_counts['val_unique_drugs']}, "
                        f"test={drug_split_counts['test_unique_drugs']}"
                    )
                else:
                    e = data[rel].edge_index.size(1)
                    tr, va, te = split_mask(e)
                data[rel].train_mask = tr
                data[rel].val_mask = va
                data[rel].test_mask = te

    else:
        # Ancestry-specific splitting for variant-phenotype; random split for drug-side_effect
        print(f"Ancestry specified for test: {ANCESTRY_TEST}, validation: {ANCESTRY_VAL}. Using ancestry-aware splits for variant-phenotype, random splits for drug-side_effect.")
        for rel in supervised_relations:
            if rel in data.edge_types:
                if rel == rel_variant:
                    # Ancestry-aware split for variant-phenotype
                    tr, va, te = ancestry_aware_split(df_variant_phenotype, target_test_ancestry=ANCESTRY_TEST, target_val_ancestry=ANCESTRY_VAL)
                    data[rel].train_mask = tr
                    data[rel].val_mask = va
                    data[rel].test_mask = te
                    print(
                        "Variant-phenotype final masks: "
                        f"train={int(tr.sum())}, val={int(va.sum())}, test={int(te.sum())}"
                    )
                else:
                    tr, va, te = split_by_source_node(data[rel].edge_index)
                    drug_split_counts = {
                        'train_unique_drugs': int(torch.unique(data[rel].edge_index[0, tr]).numel()),
                        'val_unique_drugs': int(torch.unique(data[rel].edge_index[0, va]).numel()),
                        'test_unique_drugs': int(torch.unique(data[rel].edge_index[0, te]).numel()),
                    }
                    print(
                        "Drug-side-effect source-stratified masks: "
                        f"train={int(tr.sum())}, val={int(va.sum())}, test={int(te.sum())}"
                    )
                    print(
                        "Drug-side-effect split drug counts: "
                        f"train={drug_split_counts['train_unique_drugs']}, "
                        f"val={drug_split_counts['val_unique_drugs']}, "
                        f"test={drug_split_counts['test_unique_drugs']}"
                    )
                    data[rel].train_mask = tr
                    data[rel].val_mask = va
                    data[rel].test_mask = te

    EMBED_DIM = 128 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    assert set(data.metadata()[0]) == set(data.x_dict.keys()), \
                f"Mismatch! Metadata expects {data.metadata()[0]}, but x_dict has {list(data.x_dict.keys())}"
    model = HetModel(data, hidden=EMBED_DIM).to(device) # HGT Encoder + Multi-relation Scorers #
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    @torch.no_grad()
    def evaluate(mask_key='val_mask'):
        model.eval()
        z = model(data)
        out = {}
        for rel in supervised_relations:
            pos, neg = relation_batch(data, rel, mask_key=mask_key, num_neg=4096)
            loss_rel, met = edge_loss_and_metrics(model, z, rel, pos, neg)
            out['/'.join(rel)] = {'loss': loss_rel.item(), **met}
        return out

    ###############################################################################
    # Train (multi-task + domain adversary) with Early Stopping & Checkpointing #
    ###############################################################################

    EPOCHS = 100
    lambda_grl_schedule = lambda e: min(1.0, e/20.0)  # ramp up GRL

    patience = 10
    min_delta = 1e-4
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = f"{OUTPUT}/best_het_model.pt"
    history = {
        'prediction_type': PREDICTION_TYPE,
        'epoch': [],
        'train_total': [],
        'val_total': [],
        'task_aliases': [],
        'task_display_names': {},
        'epoch_time': [],
        'lr': [],
    }
    for rel in supervised_relations:
        alias = relation_alias[rel]
        history['task_aliases'].append(alias)
        history['task_display_names'][alias] = relation_name[rel]
        history[f'{alias}_auroc'] = []
        history[f'{alias}_ap'] = []
        history[f'{alias}_train_loss'] = []
        history[f'{alias}_val_loss'] = []

    for epoch in range(1, EPOCHS+1):
        epoch_start = time.perf_counter()
        model.train()
        opt.zero_grad()
        z = model(data)

        # Multi-task edge losses
        losses = []
        losses_by_rel = {}
        metrics = {}
        for rel in supervised_relations:
            pos, neg = relation_batch(data, rel, mask_key='train_mask', num_neg=2048)
            loss_rel, met = edge_loss_and_metrics(model, z, rel, pos, neg)
            losses.append(loss_rel)
            losses_by_rel[rel] = loss_rel
            metrics['/'.join(rel)] = met

        use_domain_adv = rel_variant in supervised_relations
        if use_domain_adv:
            # Domain adversarial regularization applies only when phenotype task is active.
            lam = lambda_grl_schedule(epoch)
            phen_z = z['phenotype']
            dom_logits = model.domain_adv(phen_z, lam)
            dom_target = data['phenotype'].ancestry_dist
            dom_log_probs = F.log_softmax(dom_logits, dim=1)
            dom_loss = F.kl_div(dom_log_probs, dom_target, reduction='batchmean')
        else:
            dom_loss = torch.tensor(0.0, device=device)

        total_loss = sum(losses)
        if use_domain_adv:
            total_loss = total_loss + 0.2 * dom_loss
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()

        ###############################################################################
        # New
        model.eval()
        with torch.no_grad():
            val_out = evaluate('val_mask')
            val_loss_total = sum(v['loss'] for v in val_out.values())

        epoch_time = time.perf_counter() - epoch_start
        history['epoch'].append(epoch)
        history['train_total'].append(total_loss.item())
        history['val_total'].append(val_loss_total)
        for rel in supervised_relations:
            alias = relation_alias[rel]
            rel_key = '/'.join(rel)
            history[f'{alias}_train_loss'].append(losses_by_rel[rel].item())
            history[f'{alias}_val_loss'].append(val_out[rel_key]['loss'])
            history[f'{alias}_auroc'].append(metrics[rel_key]['auroc'])
            history[f'{alias}_ap'].append(metrics[rel_key]['ap'])
        history['epoch_time'].append(epoch_time)
        history['lr'].append(opt.param_groups[0]['lr'])
        ###############################################################################

        if epoch % 5 == 0:
            msg = (
                f"Epoch {epoch:02d} | total_loss={total_loss.item():.3f} "
                f"| val_loss={val_loss_total:.3f} "
            )
            if use_domain_adv:
                msg += f"| dom_loss={dom_loss.item():.3f} "
            for rel in supervised_relations:
                rel_key = '/'.join(rel)
                msg += (
                    f"| {relation_name[rel]} train_loss={losses_by_rel[rel].item():.3f} "
                    f"AUROC={metrics[rel_key]['auroc']:.3f} AP={metrics[rel_key]['ap']:.3f} "
                )
            print(msg)

        if val_loss_total < best_val_loss - min_delta:
            best_val_loss = val_loss_total
            patience_counter = 0
            torch.save(model.state_dict(), str(best_model_path))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best validation loss: {best_val_loss:.3f}")
                break

    # Load the best performing model weights back into memory for final testing & inference
    model.load_state_dict(torch.load(str(best_model_path), map_location=device))

    # Plot training history
    plot_training_history(history, OUTPUT)

    ############################################
    # Validation on held-out edges          #
    ############################################

    val = evaluate('val_mask')
    test = evaluate('test_mask')
    print(f"VAL ({PREDICTION_TYPE}):", val)
    print(f"TEST ({PREDICTION_TYPE}):", test)


    ############################################
    # Inference utilities                   #
    ############################################
    if PREDICTION_TYPE in ['multi_task', 'variant_phenotype']:
        print("Top phenotypes for variant 0:", topk_pheno_for_variant(model, data, device, 0, k=5))
        print("Top phenotypes for drug 0:", topk_pheno_for_drug(model, data, device, 0, k=5))
    if PREDICTION_TYPE in ['multi_task', 'drug_side_effect']:
        print("Top side effects for drug 0:", topk_side_effect_for_drug(model, data, device, 0, k=10))
        print("Polypharmacy risk (drug 0 + drug 1):", polypharmacy_risk(model, data, 0, 1, k=5))


if __name__ == "__main__":
    args = parser.parse_args()
    gat_modelling(args.input_directory, args.ancestry_test, args.ancestry_val, args.output_directory, args.prediction_type)