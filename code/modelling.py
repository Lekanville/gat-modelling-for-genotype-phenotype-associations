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
from tools.viz import (audit_and_visualize_heterodata, plot_training_history, plot_score_histogram, plot_roc_curve)
from tools.train_utils import (
    deduplicate_variant_phenotype_edges,
    deduplicate_drug_side_effect_edges,
    relation_batch,
    edge_loss_and_metrics,
)
from tools.scaling_utils import (
    compute_variant_phenotype_magnitude,
    scale_tissue_phenotype_edges,
    scale_drug_causes_llr,
)
from tools.hgt_gat import (
    ancestry_aware_split,
    build_source_split,
    source_split_masks_from_mapping,
    summarize_split,
    HetModel,
)
from tools.inference_utils import (
    topk_pheno_for_variant,
    topk_pheno_for_drug,
    topk_side_effect_for_drug,
    polypharmacy_risk,
)
from tools.process_utils import parse_bool, apply_relation_filters, restrict_one_edge_per_source

parser = argparse.ArgumentParser(description= "A script to filter data")
parser.add_argument('-i', '--input_directory', type=str, required=True, help= 'The input datasets folder')
parser.add_argument('-j', '--ancestry_test', type=str, required=True, help= 'The ancestry to hold out for testing')
parser.add_argument('-k', '--ancestry_val', type=str, required=True, help= 'The ancestry to hold out for validation')
parser.add_argument('-o', '--output_directory', type=str, required=True, help= 'The output directory to save results')
parser.add_argument(
    '--prediction_type',
    type=str,
    default='multi_task',
    choices=[
        'multi_task',
        'variant_phenotype',
        'drug_treats_phenotype',
        'drug_causes_side_effect',
    ],
    help='Select whether to train/evaluate variant-phenotype, drug repurposing, drug-side-effect, or the combined multi-task setup.'
)
parser.add_argument(
    '--ignore_relations',
    type=str,
    default='',
    help='Comma-separated relation tuples to remove from the heterogeneous graph before training, e.g. "drug,treats,clinical_outcome" or "drug,treats,clinical_outcome;gene,associated_with,clinical_outcome".'
)
parser.add_argument(
    '--focus_test_relation',
    type=str,
    default='false',
    help='When true, restrict validation/test scoring to the relation implied by the selected prediction_type. This is a leakage-control setting for the held-out split, not a graph-wide relation filter.'
)
parser.add_argument(
    '--negative_sampling_mode',
    type=str,
    default='source_aware',
    choices=['source_aware', 'global'],
    help='Choose whether negatives are sampled from a source-specific candidate pool or from the full destination set for the relation.'
)


def gat_modelling(INPUT, ANCESTRY_TEST, ANCESTRY_VAL, OUTPUT, PREDICTION_TYPE, IGNORE_RELATIONS='', FOCUS_TEST_RELATION=False, NEGATIVE_SAMPLING_MODE='source_aware'):

    FOCUS_TEST_RELATION = parse_bool(FOCUS_TEST_RELATION)
    NEGATIVE_SAMPLING_MODE = str(NEGATIVE_SAMPLING_MODE).strip().lower()
    if NEGATIVE_SAMPLING_MODE not in {'source_aware', 'global'}:
        NEGATIVE_SAMPLING_MODE = 'source_aware'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    Path(OUTPUT).mkdir(parents=True, exist_ok=True)

    # The nodes data
    (df_variants, df_tissues, df_genes, df_ancestry, df_clinical_outcomes, df_pathway, df_drugs) = node_data(INPUT)
    

    # The edge data 
    (df_ancestry_phenotype, df_gene_gene, df_variant_gene, df_drug_phenotype_treats, df_drug_phenotype_causes, df_drug_gene, df_gene_pathway, 
        df_gene_phenotype, df_gene_tissue, df_phenotype_phenotype_lin, df_phenotypes_phenotypes_ldsc, df_tissue_phenotype,
            df_variant_ancestry, df_variant_phenotype) = edge_data(INPUT)

    df_variant_phenotype = deduplicate_variant_phenotype_edges(df_variant_phenotype)
    df_drug_phenotype_causes = deduplicate_drug_side_effect_edges(df_drug_phenotype_causes)

    if (ANCESTRY_TEST == 'unspecified') and (ANCESTRY_VAL == 'unspecified'):
        variant_split_labels, variant_split_map = build_source_split(df_variant_phenotype, 'rsid')
        variant_tr_mask = torch.tensor(variant_split_labels == 'train', dtype=torch.bool)
        variant_va_mask = torch.tensor(variant_split_labels == 'val', dtype=torch.bool)
        variant_te_mask = torch.tensor(variant_split_labels == 'test', dtype=torch.bool)
        use_ancestry_split = False
    else:
        variant_tr_mask, variant_va_mask, variant_te_mask = ancestry_aware_split(
            df_variant_phenotype,
            target_test_ancestry=ANCESTRY_TEST,
            target_val_ancestry=ANCESTRY_VAL,
        )
        variant_split_map = {}
        use_ancestry_split = True

    if ('beta_unified' in df_variant_phenotype.columns):
        split_labels = np.empty(len(df_variant_phenotype), dtype=object)
        split_labels[variant_tr_mask.numpy()] = 'train'
        split_labels[variant_va_mask.numpy()] = 'val'
        split_labels[variant_te_mask.numpy()] = 'test'

        df_variant_phenotype = df_variant_phenotype.copy()
        df_variant_phenotype['split'] = split_labels
        df_variant_phenotype = compute_variant_phenotype_magnitude(
            df_variant_phenotype,
            split_col='split',
            train_label='train',
        )

    # if 'weight' in df_tissue_phenotype.columns:
    #     tissue_tr_mask, tissue_va_mask, tissue_te_mask = split_mask(len(df_tissue_phenotype))
    #     tissue_split_labels = np.empty(len(df_tissue_phenotype), dtype=object)
    #     tissue_split_labels[tissue_tr_mask.numpy()] = 'train'
    #     tissue_split_labels[tissue_va_mask.numpy()] = 'val'
    #     tissue_split_labels[tissue_te_mask.numpy()] = 'test'

    #     df_tissue_phenotype = df_tissue_phenotype.copy()
    #     df_tissue_phenotype['split'] = tissue_split_labels
    #     df_tissue_phenotype = scale_tissue_phenotype_edges(
    #         df_tissue_phenotype,
    #         split_col='split',
    #         train_label='train',
    #     )

    drug_split_map = {}
    drug_source_df = pd.concat(
        [
            df_drug_phenotype_treats[['source']].rename(columns={'source': 'chembl_id'}),
            df_drug_phenotype_causes[['chembl_id']],
        ],
        ignore_index=True,
    )

    if not drug_source_df.empty:
        _, drug_split_map = build_source_split(drug_source_df, 'chembl_id')

        if 'source' in df_drug_phenotype_treats.columns:
            df_drug_phenotype_treats = df_drug_phenotype_treats.copy()
            df_drug_phenotype_treats['split'] = df_drug_phenotype_treats['source'].astype(str).map(
                {str(k): v for k, v in drug_split_map.items()}
            ).fillna('test')

        if 'llr' in df_drug_phenotype_causes.columns:
            df_drug_phenotype_causes = df_drug_phenotype_causes.copy()
            df_drug_phenotype_causes['split'] = df_drug_phenotype_causes['chembl_id'].astype(str).map(
                {str(k): v for k, v in drug_split_map.items()}
            ).fillna('test')
            df_drug_phenotype_causes = scale_drug_causes_llr(
                df_drug_phenotype_causes,
                split_col='split',
                train_label='train',
            )

    edge_dfs = (df_ancestry_phenotype, df_gene_gene, df_variant_gene, df_drug_phenotype_treats, df_drug_phenotype_causes, df_drug_gene, df_gene_pathway, 
                    df_gene_phenotype, df_gene_tissue, df_phenotype_phenotype_lin, df_phenotypes_phenotypes_ldsc, df_tissue_phenotype,
                        df_variant_ancestry, df_variant_phenotype) 


    # Initialize Features & ID Mappings 
    data = HeteroData()

    # Building the nodes
    (variant_to_idx, tissue_to_idx, gene_to_idx, ancestry_to_idx, 
        clinical_outcomes_to_idx, pathway_to_idx, drug_to_idx, data) = nodes_init(data, df_variants, df_tissues, df_genes, 
                                                                   df_ancestry, df_clinical_outcomes, df_pathway, df_drugs)

    # (variant_to_idx, ancestry_to_idx, clinical_outcomes_to_idx, data) = nodes_init(data, df_variants, df_tissues, df_genes, 
    #                                                                    df_ancestry, df_clinical_outcomes, df_pathway, df_drugs)
    
    mappings = (variant_to_idx, tissue_to_idx, gene_to_idx, ancestry_to_idx, 
         clinical_outcomes_to_idx, pathway_to_idx, drug_to_idx)

    # mappings = (variant_to_idx, ancestry_to_idx, clinical_outcomes_to_idx)
    
    # Building the edges
    data = edges_init(data, edge_dfs, mappings)
    original_edge_counts = {tuple(edge): data[edge].edge_index.shape[1] for edge in list(data.edge_types)}
    data = apply_relation_filters(data, IGNORE_RELATIONS)
    print("All 8 Node Feature Matrices successfully loaded!")
    audit_and_visualize_heterodata(data, OUTPUT, ignored_relations=IGNORE_RELATIONS, original_edge_counts=original_edge_counts)

    # Build ancestry targets from the phenotype subset used in the variant-phenotype task.
    # The full clinical_outcome tensor still needs a full-length vector for indexing, but
    # the ancestral prior should be defined only on phenotype nodes that actually appear
    # in the variant-phenotype edges.
    
    (_, _, _, ancestry_to_idx, clinical_outcomes_to_idx, _, _) = mappings
    # (_, ancestry_to_idx, clinical_outcomes_to_idx) = mappings
    num_ancestry = len(ancestry_to_idx)
    phenotype_idx = set()

    if 'target_ancestry' in df_variant_phenotype.columns:
        for _, row in df_variant_phenotype.iterrows():
            pheno = row.get('cui') or row.get('target_hpo') or row.get('trait')
            anc = row.get('target_ancestry')
            if pd.isna(pheno) or pd.isna(anc):
                continue
            if str(pheno) in clinical_outcomes_to_idx:
                phenotype_idx.add(clinical_outcomes_to_idx[str(pheno)])

    phenotype_idx = sorted(phenotype_idx)
    phenotype_set = set(phenotype_idx)
    pheno_anc_counts = defaultdict(lambda: [0] * num_ancestry)
    if 'target_ancestry' in df_variant_phenotype.columns:
        for _, row in df_variant_phenotype.iterrows():
            pheno = row.get('cui') or row.get('target_hpo') or row.get('trait')
            anc = row.get('target_ancestry')
            if pd.isna(pheno) or pd.isna(anc):
                continue
            anc_code = str(anc).replace('ANC_', '')
            if str(pheno) in clinical_outcomes_to_idx and anc_code in ancestry_to_idx:
                ph_idx = clinical_outcomes_to_idx[str(pheno)]
                anc_idx = ancestry_to_idx[anc_code]
                if ph_idx in phenotype_set:
                    pheno_anc_counts[ph_idx][anc_idx] += 1

    phenotype_dist = np.full((len(phenotype_idx), num_ancestry), 1.0 / float(num_ancestry), dtype=np.float32)
    for row_idx, p_idx in enumerate(phenotype_idx):
        counts = np.array(pheno_anc_counts.get(p_idx, [0] * num_ancestry), dtype=np.float32)
        s = counts.sum()
        if s > 0:
            phenotype_dist[row_idx] = counts / s

    full_pheno_dist = np.full((len(clinical_outcomes_to_idx), num_ancestry), 1.0 / float(num_ancestry), dtype=np.float32)
    for row_idx, p_idx in enumerate(phenotype_idx):
        full_pheno_dist[p_idx] = phenotype_dist[row_idx]

    # Attach soft-label distributions to HeteroData for domain adversary training.
    # Only phenotype nodes in the variant-phenotype task receive ancestry targets.
    data['clinical_outcome'].ancestry_dist = torch.tensor(full_pheno_dist, dtype=torch.float32)
    phenotype_mask = torch.tensor(
        [idx in phenotype_set for idx in range(len(clinical_outcomes_to_idx))],
        dtype=torch.bool,
        device=device,
    )

    rel_variant = ('variant', 'associated_with', 'clinical_outcome')
    rel_drug_treats = ('drug', 'treats', 'clinical_outcome')
    rel_drug_causes = ('drug', 'causes', 'clinical_outcome')

    relation_alias = {
        rel_variant: 'variant_phenotype',
        rel_drug_treats: 'drug_treats_phenotype',
        rel_drug_causes: 'drug_causes_side_effect',
    }
    relation_name = {
        rel_variant: 'variant-phenotype',
        rel_drug_treats: 'drug-treats-phenotype',
        rel_drug_causes: 'drug-causes-side-effects',
    }
    prediction_relation_map = {
        'variant_phenotype': rel_variant,
        'drug_treats_phenotype': rel_drug_treats,
        'drug_causes_side_effect': rel_drug_causes,
    }

    if PREDICTION_TYPE == 'variant_phenotype':
        supervised_relations = [rel_variant]
    elif PREDICTION_TYPE == 'drug_treats_phenotype':
        supervised_relations = [rel_drug_treats]
    elif PREDICTION_TYPE == 'drug_causes_side_effect':
        supervised_relations = [rel_drug_causes]
    else:
        supervised_relations = [rel_variant, rel_drug_treats, rel_drug_causes]

    if FOCUS_TEST_RELATION:
        target_relation = prediction_relation_map.get(PREDICTION_TYPE, supervised_relations[0])
        test_relations = [target_relation]
    else:
        test_relations = supervised_relations

    print(f"Prediction type: {PREDICTION_TYPE}")
    print(f"Supervised relations: {[relation_name[r] for r in supervised_relations]}")
    if FOCUS_TEST_RELATION:
        print(f"Test-focus relation: {relation_name[test_relations[0]]}")
    drug_split_counts = {}

    if (ANCESTRY_TEST == 'unspecified') and (ANCESTRY_VAL == 'unspecified'):
        # Use the same canonical source split for both scaling and the model masks.
        if PREDICTION_TYPE == 'variant_phenotype':
            print("No ancestry specified for test or validation. Using source-aware random splits.")
        for rel in supervised_relations:
            if rel in data.edge_types:
                if rel == rel_variant:
                    source_name_by_node = {int(v): k for k, v in variant_to_idx.items()}
                    tr, va, te = source_split_masks_from_mapping(data[rel].edge_index, source_name_by_node, variant_split_map)
                    summarize_split('Variant-phenotype', data[rel].edge_index, tr, va, te, source_name='variant')
                else:
                    source_name_by_node = {int(v): k for k, v in drug_to_idx.items()}
                    tr, va, te = source_split_masks_from_mapping(data[rel].edge_index, source_name_by_node, drug_split_map)
                    label_name = 'Drug-treats-phenotype' if rel == rel_drug_treats else 'Drug-causes-side-effect'
                    summarize_split(label_name, data[rel].edge_index, tr, va, te, source_name='drug')
                data[rel].train_mask = tr
                data[rel].val_mask = va
                data[rel].test_mask = te

    else:
        if use_ancestry_split:
            # Ancestry-specific splitting for variant-phenotype; random split for drug-side_effect
            print(f"Ancestry specified for test: {ANCESTRY_TEST}, validation: {ANCESTRY_VAL}. Using ancestry-aware splits for variant-phenotype, random splits for drug-causes-side-effects.")
        else:
            print("No ancestry specified for test or validation. Using random splits.")

        for rel in supervised_relations:
            if rel in data.edge_types:
                if rel == rel_variant:
                    tr, va, te = variant_tr_mask, variant_va_mask, variant_te_mask
                    data[rel].train_mask = tr
                    data[rel].val_mask = va
                    data[rel].test_mask = te
                    summarize_split('Variant-phenotype', data[rel].edge_index, tr, va, te, source_name='variant')
                else:
                    source_name_by_node = {int(v): k for k, v in drug_to_idx.items()}
                    tr, va, te = source_split_masks_from_mapping(data[rel].edge_index, source_name_by_node, drug_split_map)
                    label_name = 'Drug-treats-phenotype' if rel == rel_drug_treats else 'Drug-causes-side-effects'
                    summarize_split(label_name, data[rel].edge_index, tr, va, te, source_name='drug')
                    data[rel].train_mask = tr
                    data[rel].val_mask = va
                    data[rel].test_mask = te

    if FOCUS_TEST_RELATION:
        for rel in supervised_relations:
            if rel in data.edge_types:
                raw_val = data[rel].val_mask.clone()
                raw_te = data[rel].test_mask.clone()
                data[rel].val_mask = restrict_one_edge_per_source(data[rel].edge_index, raw_val)
                data[rel].test_mask = restrict_one_edge_per_source(data[rel].edge_index, raw_te)
                print(
                    f"{relation_name[rel]} final masks after one-edge-per-source val/test focus: "
                    f"train={int(data[rel].train_mask.sum())}, val={int(data[rel].val_mask.sum())}, test={int(data[rel].test_mask.sum())}"
                )

    EMBED_DIM = 128 
    data = data.to(device)
    assert set(data.metadata()[0]) == set(data.x_dict.keys()), \
                f"Mismatch! Metadata expects {data.metadata()[0]}, but x_dict has {list(data.x_dict.keys())}"
    model = HetModel(data, hidden=EMBED_DIM).to(device) # HGT Encoder + Multi-relation Scorers #
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    @torch.no_grad()
    def evaluate(mask_key='val_mask', relations=None, threshold_overrides=None):
        model.eval()
        z = model(data)
        relations = supervised_relations if relations is None else relations
        out = {}
        threshold_overrides = threshold_overrides or {}
        for rel in relations:
            rel_key = '/'.join(rel)
            pos, neg = relation_batch(data, rel, mask_key=mask_key, num_neg=2048, sampling_mode=NEGATIVE_SAMPLING_MODE)
            threshold = threshold_overrides.get(rel_key)
            loss_rel, met = edge_loss_and_metrics(model, z, rel, pos, neg, threshold=threshold)
            out[rel_key] = {'loss': loss_rel.item(), **met}
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
            pos, neg, neg_stats = relation_batch(
                data,
                rel,
                mask_key='train_mask',
                num_neg=512,
                return_stats=True,
                sampling_mode=NEGATIVE_SAMPLING_MODE,
            )
            loss_rel, met = edge_loss_and_metrics(model, z, rel, pos, neg)
            met['negative_sampling_stats'] = neg_stats
            losses.append(loss_rel)
            losses_by_rel[rel] = loss_rel
            metrics['/'.join(rel)] = met

        use_domain_adv = rel_variant in supervised_relations
        if use_domain_adv:
            # Domain adversarial regularization applies only to phenotype nodes in the
            # variant-phenotype supervision task, not to all clinical outcomes.
            lam = lambda_grl_schedule(epoch)
            if phenotype_mask.any():
                phen_z = z['clinical_outcome'][phenotype_mask]
                dom_target = data['clinical_outcome'].ancestry_dist[phenotype_mask]
                dom_logits = model.domain_adv(phen_z, lam)
                dom_log_probs = F.log_softmax(dom_logits, dim=1)
                dom_loss = F.kl_div(dom_log_probs, dom_target, reduction='batchmean')
            else:
                dom_loss = torch.tensor(0.0, device=device)
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
            val_out = evaluate('val_mask', relations=test_relations if FOCUS_TEST_RELATION else supervised_relations)
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
                msg += f"| {relation_name[rel]} train_loss={losses_by_rel[rel].item():.3f}"
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

    val = evaluate('val_mask', relations=test_relations if FOCUS_TEST_RELATION else supervised_relations)
    val_best_thresholds = {rel_key: results['best_threshold'] for rel_key, results in val.items()}
    test = evaluate('test_mask', relations=test_relations if FOCUS_TEST_RELATION else supervised_relations, threshold_overrides=val_best_thresholds)
    for rel_key, results in val.items():
        if 'positive_scores' in results and 'negative_scores' in results:
            plot_score_histogram(
                results['positive_scores'],
                results['negative_scores'],
                str(Path(OUTPUT) / f'{rel_key.replace("/", "_")}_val_score_hist.png'),
                title=f'{rel_key} validation score histogram',
            )
            plot_roc_curve(
                results['positive_scores'],
                results['negative_scores'],
                str(Path(OUTPUT) / f'{rel_key.replace("/", "_")}_val_roc.png'),
                title=f'{rel_key} validation ROC curve',
            )
    for rel_key, results in test.items():
        if 'positive_scores' in results and 'negative_scores' in results:
            plot_score_histogram(
                results['positive_scores'],
                results['negative_scores'],
                str(Path(OUTPUT) / f'{rel_key.replace("/", "_")}_test_score_hist.png'),
                title=f'{rel_key} test score histogram',
            )
            plot_roc_curve(
                results['positive_scores'],
                results['negative_scores'],
                str(Path(OUTPUT) / f'{rel_key.replace("/", "_")}_test_roc.png'),
                title=f'{rel_key} test ROC curve',
            )
    print(f"VAL ({PREDICTION_TYPE}):")
    for rel_key, results in val.items():
        print(
            f"{rel_key}: loss={results['loss']:.6f}, auroc={results['auroc']:.6f}, ap={results['ap']:.6f}, "
            f"best_threshold={results['best_threshold']:.4f}, precision={results['precision']:.4f}, recall={results['recall']:.4f}, f1={results['f1']:.4f}, "
            f"n_pos={results['n_pos']}, n_neg={results['n_neg']}"
        )
    print(f"TEST ({PREDICTION_TYPE}):")
    for rel_key, results in test.items():
        print(
            f"{rel_key}: loss={results['loss']:.6f}, auroc={results['auroc']:.6f}, ap={results['ap']:.6f}, "
            f"best_threshold={results['best_threshold']:.4f}, precision={results['precision']:.4f}, recall={results['recall']:.4f}, f1={results['f1']:.4f}, "
            f"n_pos={results['n_pos']}, n_neg={results['n_neg']}, "
            f"positive_scores={results['positive_scores']}, negative_scores={results['negative_scores']}"
        )
    print("\n" + model.encoder.route_attention_report(
        title='Final HGT attended paths (all routes after training):',
        limit=None,
    ))

    ############################################
    # Inference utilities                   #
    ############################################
    if PREDICTION_TYPE in ['multi_task', 'variant_phenotype']:
        print("Top phenotypes for variant 0:", topk_pheno_for_variant(model, data, device, 0, k=5))
    if PREDICTION_TYPE in ['multi_task', 'drug_treats_phenotype']:
        print("Top phenotypes for drug 0:", topk_pheno_for_drug(model, data, device, 0, k=5))
    if PREDICTION_TYPE in ['multi_task', 'drug_causes_side_effect']:
        print("Top side effects for drug 0:", topk_side_effect_for_drug(model, data, device, 0, k=10))
        print("Polypharmacy risk (drug 0 + drug 1):", polypharmacy_risk(model, data, 0, 1, k=5))


if __name__ == "__main__":
    args = parser.parse_args()
    gat_modelling(
        args.input_directory,
        args.ancestry_test,
        args.ancestry_val,
        args.output_directory,
        args.prediction_type,
        args.ignore_relations,
        args.focus_test_relation,
    )