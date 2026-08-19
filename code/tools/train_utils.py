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

def relation_batch(data, rel, mask_key='train_mask', num_neg=None):
    """Return positive edges and sampled negatives for a relation."""
    ei = data[rel].edge_index
    mask = data[rel][mask_key]
    pos = ei[:, mask]
    num_neg = num_neg or pos.size(1)
    
    src_type, _, dst_type = rel
    num_src_nodes = data[src_type].num_nodes
    num_dst_nodes = data[dst_type].num_nodes
    
    # Corrected negative sampling call
    neg = negative_sampling(
        edge_index=pos,
        num_nodes=(num_src_nodes, num_dst_nodes),
        num_neg_samples=num_neg,
        # method='sparse'
    )
    return pos, neg

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

    return loss, {'auroc': auroc_val, 'ap': ap_val}
