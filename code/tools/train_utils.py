import torch
import pandas as pd
from torch import nn
from torch_geometric.utils import negative_sampling
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision


############################################
# Training utilities                    #
############################################

# Initialize metrics globally once
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bce = nn.BCEWithLogitsLoss()
auroc = BinaryAUROC().to(device)
ap = BinaryAveragePrecision().to(device)

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


def edge_loss_and_metrics(model, z, rel, pos, neg):
    pos_logit = model.link_logits(z, pos, rel)
    neg_logit = model.link_logits(z, neg, rel)

    # Labels
    y = torch.cat([torch.ones_like(pos_logit), torch.zeros_like(neg_logit)])
    logits = torch.cat([pos_logit, neg_logit])

    loss = bce(logits, y)

    with torch.no_grad():
        auroc.reset()
        ap.reset()
        # Compute metrics without resetting inside the batch loop
        auroc_val = auroc(logits, y.int()).item()
        ap_val = ap(logits, y.int()).item()
        conf_matrix = compute_confusion_matrix(logits, y, threshold=0.5)

    return loss, {
        'auroc': auroc_val,
        'ap': ap_val,
        'n_pos': int(pos.size(1)),
        'n_neg': int(neg.size(1)),
        'confusion_matrix': conf_matrix,
        'positive_scores': pos_logit.detach().cpu(),
        'negative_scores': neg_logit.detach().cpu(),
    }
