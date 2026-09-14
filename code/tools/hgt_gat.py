import torch
import random
import numpy as np
import pandas as pd
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import negative_sampling
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

torch.manual_seed(42)
random.seed(42) 

############################################
# Split Mask #
############################################
def split_mask(num_edges, train=0.7, val=0.15):
    perm = torch.randperm(num_edges)
    n_tr = int(train * num_edges)
    n_va = int(val * num_edges)
    m_tr = torch.zeros(num_edges, dtype=torch.bool)
    m_tr[perm[:n_tr]] = True
    m_va = torch.zeros(num_edges, dtype=torch.bool)
    m_va[perm[n_tr:n_tr+n_va]] = True
    m_te = ~(m_tr | m_va)
    return m_tr, m_va, m_te

def ancestry_aware_split(df_variant_phenotype, target_test_ancestry, target_val_ancestry):
    """
    Splits edges such that a specific ancestry is completely held out 
    for the test set, forcing the model to generalize.
    """
    # Assuming your dataframe contains an 'ancestry' column
    target_test_ancestry = f"ANC_{target_test_ancestry}"
    target_val_ancestry = f"ANC_{target_val_ancestry}"
    test_mask = df_variant_phenotype['target_ancestry'] == target_test_ancestry # Test ancestry
    val_mask = df_variant_phenotype['target_ancestry'] == target_val_ancestry  # Validation ancestry
    train_mask = ~(test_mask | val_mask)
    
    return (
        torch.tensor(train_mask.values, dtype=torch.bool),
        torch.tensor(val_mask.values, dtype=torch.bool),
        torch.tensor(test_mask.values, dtype=torch.bool)
    )


def build_source_split(df, source_col, train=0.7, val=0.15, random_state=42):
    """Create one canonical source-based split and return row labels plus source->split mapping."""
    if df is None or df.empty or source_col not in df.columns:
        empty = np.empty(0, dtype=object)
        return empty, {}

    unique_sources = pd.Series(df[source_col].dropna().astype(str).unique())
    if unique_sources.empty:
        labels = np.full(len(df), 'test', dtype=object)
        return labels, {}

    perm = unique_sources.sample(frac=1, random_state=random_state).to_numpy()
    n_tr = int(len(perm) * train)
    n_va = int(len(perm) * val)

    train_sources = set(perm[:n_tr])
    val_sources = set(perm[n_tr:n_tr + n_va])
    test_sources = set(perm[n_tr + n_va:])

    source_to_split = {}
    for s in train_sources:
        source_to_split[str(s)] = 'train'
    for s in val_sources:
        source_to_split[str(s)] = 'val'
    for s in test_sources:
        source_to_split[str(s)] = 'test'

    labels = np.full(len(df), 'test', dtype=object)
    src_series = df[source_col].astype(str)
    labels[src_series.isin(train_sources)] = 'train'
    labels[src_series.isin(val_sources)] = 'val'
    return labels, source_to_split


def source_split_masks_from_mapping(edge_index, source_name_by_node, source_to_split):
    """Turn one canonical source->split map into train/val/test edge masks."""
    if edge_index.size(1) == 0:
        empty = torch.zeros(0, dtype=torch.bool)
        return empty, empty, empty

    edge_count = edge_index.size(1)
    m_tr = torch.zeros(edge_count, dtype=torch.bool)
    m_va = torch.zeros(edge_count, dtype=torch.bool)
    m_te = torch.zeros(edge_count, dtype=torch.bool)

    for i, src_idx in enumerate(edge_index[0].tolist()):
        src_name = source_name_by_node.get(int(src_idx))
        label = source_to_split.get(str(src_name), 'test')
        if label == 'train':
            m_tr[i] = True
        elif label == 'val':
            m_va[i] = True
        else:
            m_te[i] = True

    return m_tr, m_va, m_te


def summarize_split(label, edge_index, tr, va, te, source_name=None):
    """Print a consistent train/val/test summary for a source-based split."""
    counts = {
        'train_edges': int(tr.sum()),
        'val_edges': int(va.sum()),
        'test_edges': int(te.sum()),
    }
    if source_name is not None:
        unique_counts = {
            'train_unique_sources': int(torch.unique(edge_index[0, tr]).numel()),
            'val_unique_sources': int(torch.unique(edge_index[0, va]).numel()),
            'test_unique_sources': int(torch.unique(edge_index[0, te]).numel()),
        }
        print(
            f"{label} source-stratified masks: "
            f"train={counts['train_edges']}, val={counts['val_edges']}, test={counts['test_edges']}"
        )
        print(
            f"{label} split source counts: "
            f"train={unique_counts['train_unique_sources']}, "
            f"val={unique_counts['val_unique_sources']}, "
            f"test={unique_counts['test_unique_sources']}"
        )
    else:
        print(
            f"{label} final masks: "
            f"train={counts['train_edges']}, val={counts['val_edges']}, test={counts['test_edges']}"
        )

############################################
# HGT Encoder + Multi-relation Scorers #
############################################
class HGTEncoder(nn.Module):
    def __init__(self, metadata, hidden, heads=2, layers=2):
        super().__init__()
        self.metadata = metadata
        self.lin_dict = nn.ModuleDict({
            nt: Linear(-1, hidden) for nt in metadata[0]
        })
        self.layers = nn.ModuleList([
            HGTConv(in_channels=hidden, out_channels=hidden,
                    metadata=metadata, heads=heads) for _ in range(layers)
        ])
        self.norms = nn.ModuleDict({nt: nn.LayerNorm(hidden) for nt in metadata[0]})

    def forward(self, x_dict, edge_index_dict):
        x_dict = {nt: self.lin_dict[nt](x) for nt, x in x_dict.items() if nt in self.lin_dict}
        for conv in self.layers:
            print("x_dict keys:", x_dict.keys())
            print("Expected node types:", self.metadata[0])
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {nt: self.norms[nt](F.gelu(x)) for nt, x in x_dict.items()}
        return x_dict

class RelScorer(nn.Module):
    """Simple DistMult-like scorer per relation."""
    def __init__(self, hidden, relations):
        super().__init__()
        self.rel_params = nn.ParameterDict({
            '__'.join(rel): nn.Parameter(torch.randn(hidden)) for rel in relations
        })

    def score(self, h_src, h_dst, rel_key):
        r = self.rel_params[rel_key]  # (H,)
        return (h_src * r * h_dst).sum(-1)

    def forward(self, h_src, h_dst, rel_key):
        return self.score(h_src, h_dst, rel_key)

class DomainAdversary(nn.Module):
    """Ancestry-invariance for phenotype embeddings via GRL."""
    def __init__(self, hidden, n_domains=3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden//2), nn.ReLU(),
            nn.Linear(hidden//2, n_domains)
        )
    def forward(self, z_phenotype, lambda_grl):
        # Gradient reversal via hook
        rev = GradReverse.apply(z_phenotype, lambda_grl)
        return self.classifier(rev)

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

class HetModel(nn.Module):
    def __init__(self, data, hidden):
        super().__init__()
        self.metadata = data.metadata()
        self.encoder = HGTEncoder(self.metadata, hidden=hidden, heads=2, layers=2)
        self.scorer  = RelScorer(hidden, relations=self.metadata[1])
        self.domain_adv = DomainAdversary(hidden, n_domains=3)  

    def forward(self, data):
        z = self.encoder(data.x_dict, data.edge_index_dict)
        return z

    def link_logits(self, z, edge_index, rel):
        src_t, _, dst_t = rel
        key = '__'.join(rel)
        h_src = z[src_t][edge_index[0]]
        h_dst = z[dst_t][edge_index[1]]
        return self.scorer(h_src, h_dst, key)