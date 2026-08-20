import torch
import random
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


def split_by_source_node(edge_index, train=0.7, val=0.15):
    """Split edges so each source node appears in exactly one split."""
    if edge_index.size(1) == 0:
        empty = torch.zeros(0, dtype=torch.bool)
        return empty, empty, empty

    src_nodes = edge_index[0]
    unique_src = torch.unique(src_nodes)
    perm = unique_src[torch.randperm(unique_src.numel())]

    n_src = perm.numel()
    n_tr = int(train * n_src)
    n_va = int(val * n_src)

    src_tr = perm[:n_tr]
    src_va = perm[n_tr:n_tr + n_va]
    src_te = perm[n_tr + n_va:]

    m_tr = torch.isin(src_nodes, src_tr)
    m_va = torch.isin(src_nodes, src_va)
    m_te = torch.isin(src_nodes, src_te)
    return m_tr, m_va, m_te

def ancestry_aware_split(df_variant_phenotype, target_test_ancestry, target_val_ancestry, min_group_size=5):
    """
    Split variant-phenotype associations by the canonical variant-phenotype pair,
    not by repeated ancestry rows. This avoids leakage when the same biological
    edge appears in multiple ancestry-specific GWAS studies.

    For tiny ancestry groups (e.g. only a few rows), we do not use them as a
    hold-out set because they likely reflect a small subset of the same European
    associations and would create unstable validation/test targets.
    """
    if df_variant_phenotype is None or df_variant_phenotype.empty:
        raise ValueError("df_variant_phenotype is empty; cannot create ancestry-aware split.")

    df = df_variant_phenotype.copy()
    required = {'rsid', 'target_phenotype', 'target_ancestry'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for ancestry split: {sorted(missing)}")

    df = df.dropna(subset=['rsid', 'target_phenotype', 'target_ancestry']).copy()
    if 'magnitude' in df.columns:
        df['magnitude'] = pd.to_numeric(df['magnitude'], errors='coerce').fillna(0.0)

    df['target_ancestry'] = df['target_ancestry'].astype(str).str.replace('ANC_', '', regex=False)
    df['pair_key'] = df['rsid'].astype(str) + '|' + df['target_phenotype'].astype(str)

    # Keep only the strongest evidence for a given variant-phenotype pair.
    if 'magnitude' in df.columns:
        df = df.sort_values(['pair_key', 'magnitude'], ascending=[True, False], na_position='last')
        dedup_df = df.drop_duplicates(subset='pair_key', keep='first').reset_index(drop=True)
    else:
        dedup_df = df.drop_duplicates(subset='pair_key', keep='first').reset_index(drop=True)

    target_test_ancestry = str(target_test_ancestry).replace('ANC_', '')
    target_val_ancestry = str(target_val_ancestry).replace('ANC_', '')

    group_counts = dedup_df['target_ancestry'].value_counts()
    print(f"Unique variant-phenotype pairs by ancestry after deduplication: {group_counts.to_dict()}")
    print(f"Requested ancestry hold-out: test={target_test_ancestry}, val={target_val_ancestry}, min_group_size={min_group_size}")

    use_test = target_test_ancestry in group_counts and group_counts[target_test_ancestry] >= min_group_size
    use_val = target_val_ancestry in group_counts and group_counts[target_val_ancestry] >= min_group_size

    if not use_test and not use_val:
        # Conservative fallback: tiny ancestry groups are not usable as held-out splits.
        n = len(dedup_df)
        train_mask = torch.ones(n, dtype=torch.bool)
        val_mask = torch.zeros(n, dtype=torch.bool)
        test_mask = torch.zeros(n, dtype=torch.bool)
        return train_mask, val_mask, test_mask

    if not use_test:
        print(f"Warning: ancestry '{target_test_ancestry}' has too few unique variant-phenotype pairs; keeping it in training.")
        test_mask = torch.zeros(len(dedup_df), dtype=torch.bool)
    else:
        test_mask = dedup_df['target_ancestry'].eq(target_test_ancestry).to_numpy()
        test_mask = torch.tensor(test_mask, dtype=torch.bool)

    if not use_val:
        print(f"Warning: ancestry '{target_val_ancestry}' has too few unique variant-phenotype pairs; keeping it in training.")
        val_mask = torch.zeros(len(dedup_df), dtype=torch.bool)
    else:
        val_mask = dedup_df['target_ancestry'].eq(target_val_ancestry).to_numpy()
        val_mask = torch.tensor(val_mask, dtype=torch.bool)

    train_mask = ~(test_mask | val_mask)
    print(f"Final ancestry split counts: train={int(train_mask.sum())}, val={int(val_mask.sum())}, test={int(test_mask.sum())}")
    return train_mask, val_mask, test_mask

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

    # def forward(self, x_dict, edge_index_dict):
    #     x_dict = {nt: self.lin_dict[nt](x) for nt, x in x_dict.items()}
    #     for conv in self.layers:
    #         x_dict = conv(x_dict, edge_index_dict)
    #         x_dict = {nt: self.norms[nt](F.gelu(x)) for nt, x in x_dict.items()}
    #     return x_dict

    # def forward(self, x_dict, edge_index_dict):
    #     x_dict = {nt: self.lin_dict[nt](x) for nt, x in x_dict.items() if nt in self.lin_dict}
    #     for conv in self.layers:
    #         prev_x_dict = x_dict
    #         out_dict = conv(x_dict, edge_index_dict)

    #         x_dict = {}
    #         for nt in self.lin_dict.keys():
    #             if nt in out_dict and out_dict[nt] is not None:
    #                 x = out_dict[nt]
    #             else:
    #                 x = prev_x_dict[nt]
    #             x_dict[nt] = x

    #         x_dict = {nt: self.norms[nt](F.gelu(x)) for nt, x in x_dict.items()}
    #     return x_dict

    def forward(self, x_dict, edge_index_dict):
        x_dict = {nt: self.lin_dict[nt](x) for nt, x in x_dict.items()}

        for conv in self.layers:
            prev_x_dict = x_dict
            out_dict = conv(x_dict, edge_index_dict)

            x_dict = {}
            for nt in self.metadata[0]:
                if nt in out_dict and out_dict[nt] is not None:
                    # Apply Activation + Norm + Residual on valid target nodes
                    x_dict[nt] = self.norms[nt](F.gelu(out_dict[nt]) + prev_x_dict[nt])
                else:
                    # Carry forward untouched representation without re-applying GELU/Norm
                    x_dict[nt] = prev_x_dict[nt]

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
        # Number of ancestry domains equals number of ancestry nodes in the graph
        n_domains = data['ancestry'].num_nodes if 'ancestry' in data.metadata()[0] else 1
        self.domain_adv = DomainAdversary(hidden, n_domains=n_domains)

    def forward(self, data):
        z = self.encoder(data.x_dict, data.edge_index_dict)
        return z

    def link_logits(self, z, edge_index, rel):
        src_t, _, dst_t = rel
        key = '__'.join(rel)
        h_src = z[src_t][edge_index[0]]
        h_dst = z[dst_t][edge_index[1]]
        return self.scorer(h_src, h_dst, key)