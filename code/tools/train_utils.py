import torch
import pandas as pd
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import negative_sampling
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision


############################################
# Training utilities                    #
############################################

# Initialize metrics globally once
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOPK_K_VALUES = (50, 100, 500, 1000, 2500)
bce = nn.BCEWithLogitsLoss()
auroc = BinaryAUROC().to(device)
ap = BinaryAveragePrecision().to(device)


def compute_topk_ranking_metrics(pos_scores, neg_scores, ks=TOPK_K_VALUES):
    """Compute Precision@K, Recall@K and Lift@K for a binary ranking task."""
    pos_scores = pos_scores.float().reshape(-1)
    neg_scores = neg_scores.float().reshape(-1)
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        out = {}
        for k in ks:
            out[f'precision@{k}'] = 0.0
            out[f'recall@{k}'] = 0.0
            out[f'lift@{k}'] = 0.0
        return out

    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([
        torch.ones_like(pos_scores, dtype=torch.long),
        torch.zeros_like(neg_scores, dtype=torch.long),
    ])
    total_pos = int(labels.sum().item())
    total_items = int(scores.numel())
    baseline_rate = total_pos / max(total_items, 1)

    ranking = torch.argsort(scores, descending=True)
    sorted_labels = labels[ranking]
    out = {}
    for k in ks:
        k_eff = min(int(k), int(sorted_labels.numel()))
        if k_eff <= 0:
            out[f'precision@{k}'] = 0.0
            out[f'recall@{k}'] = 0.0
            out[f'lift@{k}'] = 0.0
            continue
        top_labels = sorted_labels[:k_eff]
        tp = int(top_labels.sum().item())
        precision = tp / k_eff
        recall = tp / max(total_pos, 1)
        lift = (precision / max(baseline_rate, 1e-12))
        out[f'precision@{k}'] = float(precision)
        out[f'recall@{k}'] = float(recall)
        out[f'lift@{k}'] = float(lift)
    return out


def compute_topk_coverage_summary(pos_edges, neg_edges, pos_scores, neg_scores, ks=TOPK_K_VALUES):
    """Coverage summary for drug-level ranking. For each source in the evaluation set,
    report whether at least one true positive appears in the top-k candidates, and count
    the unique positive outcomes recovered across the whole evaluation pool."""
    pos_edges = pos_edges.to(torch.long)
    neg_edges = neg_edges.to(torch.long)
    pos_scores = pos_scores.float().reshape(-1)
    neg_scores = neg_scores.float().reshape(-1)

    if pos_edges.numel() == 0:
        return {
            'n_test_drugs': 0,
            'n_unique_outcomes': 0,
            'unique_outcomes_recovered_topk': {f'k={k}': 0 for k in ks},
            'percent_drugs_with_true_positive_in_topk': {f'k={k}': 0.0 for k in ks},
        }

    unique_sources = sorted(set(pos_edges[0].tolist()) | set(neg_edges[0].tolist()))
    total_drugs = len(unique_sources)
    unique_outcomes = set(pos_edges[1].tolist())

    summary = {
        'n_test_drugs': int(total_drugs),
        'n_unique_outcomes': int(len(unique_outcomes)),
    }

    for k in ks:
        recovered_outcomes = set()
        drugs_with_hit = 0
        for src in unique_sources:
            src_pos_mask = pos_edges[0] == src
            src_neg_mask = neg_edges[0] == src
            if src_pos_mask.any():
                src_pos_edges = pos_edges[:, src_pos_mask]
                src_pos_scores = pos_scores[src_pos_mask]
            else:
                src_pos_edges = torch.empty((2, 0), dtype=torch.long, device=pos_edges.device)
                src_pos_scores = torch.empty((0,), dtype=torch.float32, device=pos_scores.device)
            if src_neg_mask.any():
                src_neg_edges = neg_edges[:, src_neg_mask]
                src_neg_scores = neg_scores[src_neg_mask]
            else:
                src_neg_edges = torch.empty((2, 0), dtype=torch.long, device=neg_edges.device)
                src_neg_scores = torch.empty((0,), dtype=torch.float32, device=neg_scores.device)

            combined_edges = torch.cat([src_pos_edges, src_neg_edges], dim=1) if src_pos_edges.numel() + src_neg_edges.numel() > 0 else torch.empty((2, 0), dtype=torch.long, device=pos_edges.device)
            combined_scores = torch.cat([src_pos_scores, src_neg_scores], dim=0) if src_pos_scores.numel() + src_neg_scores.numel() > 0 else torch.empty((0,), dtype=torch.float32, device=pos_scores.device)
            combined_labels = torch.cat([
                torch.ones(src_pos_scores.numel(), dtype=torch.long, device=pos_scores.device),
                torch.zeros(src_neg_scores.numel(), dtype=torch.long, device=neg_scores.device),
            ]) if combined_scores.numel() > 0 else torch.empty((0,), dtype=torch.long, device=pos_scores.device)

            if combined_scores.numel() == 0:
                continue

            top_idx = torch.argsort(combined_scores, descending=True)[:min(int(k), combined_scores.numel())]
            top_labels = combined_labels[top_idx]
            top_outcomes = combined_edges[1, top_idx]
            if int(top_labels.sum().item()) > 0:
                drugs_with_hit += 1
                recovered_outcomes.update(top_outcomes[top_labels == 1].tolist())

        summary[f'precision@{k}'] = 0.0
        summary[f'recall@{k}'] = 0.0
        summary[f'lift@{k}'] = 0.0
        summary[f'unique_outcomes_recovered_top{k}'] = int(len(recovered_outcomes))
        summary[f'percent_drugs_with_true_positive_in_top{k}'] = float(drugs_with_hit / max(total_drugs, 1) * 100.0)
    return summary

def deduplicate_variant_phenotype_edges(df_variant_phenotype):
    """Keep the strongest edge per variant-phenotype pair to avoid ancestry-row duplication."""
    if df_variant_phenotype is None or df_variant_phenotype.empty:
        return df_variant_phenotype

    df = df_variant_phenotype.copy()
    required = {'rsid', 'target_phenotype'}
    missing = required - set(df.columns)
    if missing:
        return df

    df = df.dropna(subset=['rsid', 'target_phenotype']).copy()
    if 'magnitude' in df.columns:
        df['magnitude'] = pd.to_numeric(df['magnitude'], errors='coerce').fillna(0.0)

    df['pair_key'] = df['rsid'].astype(str) + '|' + df['target_phenotype'].astype(str)
    if 'magnitude' in df.columns:
        df = df.sort_values(['pair_key', 'magnitude'], ascending=[True, False], na_position='last')
    n_before = len(df)
    df = df.drop_duplicates(subset='pair_key', keep='first').drop(columns=['pair_key']).reset_index(drop=True)
    n_after = len(df)
    if n_after < n_before:
        print(f"Deduplicated variant-phenotype edges: {n_before - n_after} redundant rows removed.")
    return df


def deduplicate_drug_side_effect_edges(df_drug_side_effect):
    """Keep one strongest drug-side-effect edge per (drug, side_effect) pair."""
    if df_drug_side_effect is None or df_drug_side_effect.empty:
        return df_drug_side_effect

    df = df_drug_side_effect.copy()
    required = {'chembl_id', 'meddraCode'}
    missing = required - set(df.columns)
    if missing:
        return df

    df = df.dropna(subset=['chembl_id', 'meddraCode']).copy()
    if 'llr_norm' in df.columns:
        df['llr_norm'] = pd.to_numeric(df['llr_norm'], errors='coerce').fillna(0.0)

    df['pair_key'] = df['chembl_id'].astype(str) + '|' + df['meddraCode'].astype(str)
    if 'llr_norm' in df.columns:
        df = df.sort_values(['pair_key', 'llr_norm'], ascending=[True, False], na_position='last')
    n_before = len(df)
    df = df.drop_duplicates(subset='pair_key', keep='first').drop(columns=['pair_key']).reset_index(drop=True)
    n_after = len(df)
    if n_after < n_before:
        print(f"Deduplicated drug-side-effect edges: {n_before - n_after} redundant rows removed.")
    return df

def relation_batch(data, rel, mask_key='train_mask', num_neg=None, return_stats=False, sampling_mode='source_aware'):
    """Return positive edges and negatives for a relation.

    By default, negatives are sampled per source node from that source's destination pool,
    excluding its observed positives. This avoids the pathological case where a global
    destination set makes random negatives trivially easy to rank. If sampling_mode is
    set to 'global', negatives are sampled from the full destination space.

    If return_stats=True, also return a compact summary of the candidate pool sizes and
    the sampling regime instead of logging one line per source.
    """
    sampling_mode = str(sampling_mode).strip().lower()
    if sampling_mode not in {'source_aware', 'global'}:
        sampling_mode = 'source_aware'

    ei = data[rel].edge_index
    mask = data[rel][mask_key]
    pos = ei[:, mask]
    empty_neg = torch.empty((2, 0), dtype=torch.long, device=ei.device if ei.numel() > 0 else torch.device('cpu'))
    if pos.numel() == 0:
        stats = {
            'n_sources': 0,
            'candidate_pool_size_avg': 0.0,
            'candidate_pool_size_min': 0,
            'candidate_pool_size_max': 0,
            'sampled_negatives': 0,
            'mean_negatives_per_source': 0.0,
            'sampling_mode': sampling_mode,
            'desired_total_negatives': int(num_neg or 0),
        }
        return (pos, empty_neg, stats) if return_stats else (pos, empty_neg)

    if pos.dim() != 2 or pos.size(1) == 0:
        stats = {
            'n_sources': 0,
            'candidate_pool_size_avg': 0.0,
            'candidate_pool_size_min': 0,
            'candidate_pool_size_max': 0,
            'sampled_negatives': 0,
            'mean_negatives_per_source': 0.0,
            'sampling_mode': sampling_mode,
            'desired_total_negatives': int(num_neg or 0),
        }
        return (pos, empty_neg, stats) if return_stats else (pos, empty_neg)

    src_type, _, dst_type = rel
    num_src_nodes = data[src_type].num_nodes
    num_dst_nodes = data[dst_type].num_nodes
    device = pos.device
    dst_pool = torch.arange(num_dst_nodes, device=device)
    desired_total = int(num_neg or pos.size(1))

    if sampling_mode == 'global':
        num_neg_samples = min(desired_total, int(num_src_nodes * num_dst_nodes))
        neg = negative_sampling(
            edge_index=pos,
            num_nodes=(num_src_nodes, num_dst_nodes),
            num_neg_samples=num_neg_samples,
            force_undirected=False,
        )
        stats = {
            'n_sources': int(pos[0].unique().numel()),
            'candidate_pool_size_avg': float(num_dst_nodes),
            'candidate_pool_size_min': int(num_dst_nodes),
            'candidate_pool_size_max': int(num_dst_nodes),
            'sampled_negatives': int(neg.size(1)),
            'mean_negatives_per_source': float(neg.size(1) / max(1, pos[0].unique().numel())),
            'sampling_mode': 'global',
            'desired_total_negatives': desired_total,
        }
        return (pos, neg, stats) if return_stats else (pos, neg)

    neg_src_list = []
    neg_dst_list = []
    candidate_sizes = []
    sampled_counts = []

    for src in pos[0].unique():
        src_mask = pos[0] == src
        pos_dst = pos[1, src_mask]
        allowed = torch.ones(num_dst_nodes, dtype=torch.bool, device=device)
        if pos_dst.numel() > 0:
            allowed[pos_dst] = False

        candidate_dsts = dst_pool[allowed]
        candidate_count = int(candidate_dsts.numel())
        candidate_sizes.append(candidate_count)
        if candidate_count == 0:
            sampled_counts.append(0)
            continue

        n_needed = min(desired_total, candidate_count)
        sample_idx = torch.randperm(candidate_count, device=device)[:n_needed]
        neg_src_list.append(torch.full((n_needed,), src, device=device, dtype=torch.long))
        neg_dst_list.append(candidate_dsts[sample_idx])
        sampled_counts.append(int(n_needed))

    if not neg_src_list:
        neg = torch.empty((2, 0), dtype=torch.long, device=device)
    else:
        neg = torch.stack([
            torch.cat(neg_src_list),
            torch.cat(neg_dst_list),
        ], dim=0)

    stats = {
        'n_sources': int(len(candidate_sizes)),
        'candidate_pool_size_avg': float(sum(candidate_sizes) / len(candidate_sizes)) if candidate_sizes else 0.0,
        'candidate_pool_size_min': int(min(candidate_sizes)) if candidate_sizes else 0,
        'candidate_pool_size_max': int(max(candidate_sizes)) if candidate_sizes else 0,
        'sampled_negatives': int(sum(sampled_counts)),
        'mean_negatives_per_source': float(sum(sampled_counts) / len(sampled_counts)) if sampled_counts else 0.0,
        'sampling_mode': 'source_aware',
        'desired_total_negatives': desired_total,
    }
    return (pos, neg, stats) if return_stats else (pos, neg)


def compute_confusion_matrix(logits, labels, threshold=0.5):
    """Return a simple confusion-matrix summary for a binary edge prediction task."""
    preds = (logits >= threshold).to(torch.int64)
    labels = labels.to(torch.int64)
    tn = torch.sum((preds == 0) & (labels == 0)).item()
    fp = torch.sum((preds == 1) & (labels == 0)).item()
    fn = torch.sum((preds == 0) & (labels == 1)).item()
    tp = torch.sum((preds == 1) & (labels == 1)).item()
    return {
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'threshold': float(threshold),
    }


def select_best_threshold(pos_scores, neg_scores, n_candidates=200):
    """Choose the threshold that maximizes F1 on the observed positive/negative scores."""
    pos_scores = pos_scores.float().reshape(-1)
    neg_scores = neg_scores.float().reshape(-1)
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return 0.5, 0.0, 0.0, 0.0

    all_scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([
        torch.ones_like(pos_scores, dtype=torch.long),
        torch.zeros_like(neg_scores, dtype=torch.long),
    ])

    if all_scores.numel() > n_candidates:
        threshold_values = torch.linspace(all_scores.min().item(), all_scores.max().item(), steps=n_candidates)
    else:
        threshold_values = torch.unique(all_scores)

    best_threshold = 0.5
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0

    for threshold in threshold_values:
        preds = (all_scores >= threshold).long()
        tp = ((preds == 1) & (labels == 1)).sum().float().item()
        fp = ((preds == 1) & (labels == 0)).sum().float().item()
        fn = ((preds == 0) & (labels == 1)).sum().float().item()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)

        if f1 > best_f1:
            best_threshold = float(threshold)
            best_precision = precision
            best_recall = recall
            best_f1 = f1

    return best_threshold, best_precision, best_recall, best_f1


def edge_loss_and_metrics(model, z, rel, pos, neg, threshold=None):
    pos_logit = model.link_logits(z, pos, rel)
    neg_logit = model.link_logits(z, neg, rel)

    # Labels
    y = torch.cat([torch.ones_like(pos_logit), torch.zeros_like(neg_logit)])
    logits = torch.cat([pos_logit, neg_logit])

    pos_count = float(pos_logit.numel())
    neg_count = float(neg_logit.numel())
    if pos_count > 0 and neg_count > 0:
        pos_weight = max(neg_count / max(pos_count, 1.0), 1.0)
        weight_tensor = torch.full_like(logits, pos_weight, device=logits.device)
        loss = F.binary_cross_entropy_with_logits(logits, y.float(), pos_weight=weight_tensor)
    else:
        loss = bce(logits, y)

    with torch.no_grad():
        auroc.reset()
        ap.reset()
        auroc_val = auroc(logits, y.int()).item()
        ap_val = ap(logits, y.int()).item()

        if threshold is None:
            selected_threshold, precision, recall, f1 = select_best_threshold(pos_logit, neg_logit)
        else:
            selected_threshold = float(threshold)
            preds = (logits >= selected_threshold).long()
            tp = ((preds == 1) & (y == 1)).sum().float().item()
            fp = ((preds == 1) & (y == 0)).sum().float().item()
            fn = ((preds == 0) & (y == 1)).sum().float().item()
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)

        conf_matrix = compute_confusion_matrix(logits, y, threshold=selected_threshold)
        topk_metrics = compute_topk_ranking_metrics(pos_logit, neg_logit, ks=TOPK_K_VALUES)
        coverage_summary = compute_topk_coverage_summary(pos, neg, pos_logit, neg_logit, ks=TOPK_K_VALUES)

    result = {
        'auroc': auroc_val,
        'ap': ap_val,
        'n_pos': int(pos.size(1)),
        'n_neg': int(neg.size(1)),
        'confusion_matrix': conf_matrix,
        'positive_scores': pos_logit.detach().cpu(),
        'negative_scores': neg_logit.detach().cpu(),
        'best_threshold': float(selected_threshold),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'n_test_drugs': int(coverage_summary.get('n_test_drugs', 0)),
        'n_unique_outcomes': int(coverage_summary.get('n_unique_outcomes', 0)),
    }
    result.update(topk_metrics)
    for k in TOPK_K_VALUES:
        result[f'unique_outcomes_recovered_top{k}'] = int(coverage_summary.get(f'unique_outcomes_recovered_top{k}', 0))
        result[f'percent_drugs_with_true_positive_in_top{k}'] = float(coverage_summary.get(f'percent_drugs_with_true_positive_in_top{k}', 0.0))
    return loss, result
