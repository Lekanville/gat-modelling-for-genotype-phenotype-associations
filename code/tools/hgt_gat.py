import math
import torch
import random
import numpy as np
import pandas as pd
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.conv.hgt_conv import construct_bipartite_edge_index
from torch_geometric.utils import negative_sampling, softmax
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
class HGTAttentionRecorder(HGTConv):
    """Capture per-relation attention statistics from the HGT message-passing path.

    PyG's built-in HGTConv does not expose attention tensors publicly. We therefore
    reproduce its message-passing path and store the message attention for each
    relation block so that route importance can be inspected by source/destination
    relation and by layer.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_attention = {}
        self._last_alpha = None
        self._relation_ranges = {}

    def _compute_relation_ranges(self, edge_index_dict):
        ranges = {}
        offset = 0
        for rel in edge_index_dict.keys():
            num_edges = edge_index_dict[rel].size(1)
            ranges[rel] = (offset, offset + num_edges)
            offset += num_edges
        self._relation_ranges = ranges

    def forward(self, x_dict, edge_index_dict):
        self.last_attention = {}
        self._last_alpha = None
        self._compute_relation_ranges(edge_index_dict)

        out_channels = self.out_channels
        heads = self.heads
        dim = out_channels // heads

        k_dict, q_dict, v_dict, out_dict = {}, {}, {}, {}
        kqv_dict = self.kqv_lin(x_dict)
        for key, val in kqv_dict.items():
            k, q, v = torch.tensor_split(val, 3, dim=1)
            k_dict[key] = k.view(-1, heads, dim)
            q_dict[key] = q.view(-1, heads, dim)
            v_dict[key] = v.view(-1, heads, dim)

        q, dst_offset = self._cat(q_dict)
        k, v, src_offset = self._construct_src_node_feat(
            k_dict, v_dict, edge_index_dict)

        edge_index, edge_attr = construct_bipartite_edge_index(
            edge_index_dict,
            src_offset,
            dst_offset,
            edge_attr_dict=self.p_rel,
            num_nodes=k.size(0),
        )

        out = self.propagate(edge_index, k=k, q=q, v=v, edge_attr=edge_attr)

        for node_type, start_offset in dst_offset.items():
            end_offset = start_offset + q_dict[node_type].size(0)
            if node_type in self.dst_node_types:
                out_dict[node_type] = out[start_offset:end_offset]

        a_dict = self.out_lin({
            key: F.gelu(value) if value is not None else value
            for key, value in out_dict.items()
        })

        for node_type, out_node in out_dict.items():
            out_node = a_dict[node_type]
            if out_node.size(-1) == x_dict[node_type].size(-1):
                alpha = self.skip[node_type].sigmoid()
                out_node = alpha * out_node + (1 - alpha) * x_dict[node_type]
            out_dict[node_type] = out_node

        if self._last_alpha is not None:
            alpha = self._last_alpha
            for rel, (start, end) in self._relation_ranges.items():
                if end > start:
                    rel_alpha = alpha[start:end]
                    if rel_alpha.numel() > 0:
                        self.last_attention[str(rel)] = rel_alpha.mean(dim=0)

        return out_dict

    def message(self, k_j: torch.Tensor, q_i: torch.Tensor, v_j: torch.Tensor,
                edge_attr: torch.Tensor, index: torch.Tensor, ptr: torch.Tensor,
                size_i: int):
        alpha = (q_i * k_j).sum(dim=-1) * edge_attr
        alpha = alpha / math.sqrt(q_i.size(-1))
        alpha = softmax(alpha, index, ptr, size_i)
        self._last_alpha = alpha.detach().clone()
        out = v_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)

    def route_attention_summary(self):
        summary = {}
        for rel_key, attn in self.last_attention.items():
            if torch.is_tensor(attn):
                summary[str(rel_key)] = float(attn.mean().item())
        return summary


class HGTEncoder(nn.Module):
    def __init__(self, metadata, hidden, heads=2, layers=2):
        super().__init__()
        self.metadata = metadata
        self.lin_dict = nn.ModuleDict({
            nt: Linear(-1, hidden) for nt in metadata[0]
        })
        self.layers = nn.ModuleList([
            HGTAttentionRecorder(in_channels=hidden, out_channels=hidden,
                                metadata=metadata, heads=heads) for _ in range(layers)
        ])
        self.norms = nn.ModuleDict({nt: nn.LayerNorm(hidden) for nt in metadata[0]})

    def forward(self, x_dict, edge_index_dict):
        missing = [nt for nt in self.metadata[0] if nt not in x_dict]
        if missing:
            raise KeyError(f"HGTEncoder missing node types: {missing}. Available keys: {sorted(x_dict.keys())}")

        projected = {nt: self.lin_dict[nt](x) for nt, x in x_dict.items() if nt in self.lin_dict}
        for conv in self.layers:
            prev = projected.copy()
            projected = conv(projected, edge_index_dict)
            for nt in self.metadata[0]:
                if nt not in projected:
                    projected[nt] = prev[nt]
            projected = {
                nt: self.norms[nt](F.gelu(projected[nt]))
                for nt in self.metadata[0]
                if nt in projected
            }
        return projected

    def route_attention_summary(self):
        """Return a route-level summary aggregated across HGT layers.

        Each relation type (e.g. ('drug', 'causes', 'clinical_outcome')) is reported once,
        using the mean attention across layers. The layer ids are retained for traceability
        when a route differs materially between layers.
        """
        grouped = {}
        for layer_idx, layer in enumerate(self.layers):
            for rel_key, attn_mean in layer.route_attention_summary().items():
                if rel_key not in grouped:
                    grouped[rel_key] = {'values': [], 'layers': []}
                grouped[rel_key]['values'].append(attn_mean)
                grouped[rel_key]['layers'].append(layer_idx)

        summary = []
        for rel_key, info in grouped.items():
            values = info['values']
            summary.append({
                'relation': rel_key,
                'mean_attention': float(sum(values) / len(values)),
                'layers': info['layers'],
                'layer_count': len(values),
            })

        return sorted(summary, key=lambda d: d['mean_attention'], reverse=True)

    def route_attention_report(self, title='HGT attended paths (sorted by mean attention):', limit=None):
        """Return a clean manager-facing text summary of all attended routes."""
        summary = self.route_attention_summary()
        if not summary:
            return f"{title}\n  No HGT attention values were captured in this runtime."

        rows = summary if limit is None else summary[:limit]
        lines = [title]
        for idx, item in enumerate(rows, start=1):
            rel = item['relation']
            if isinstance(rel, tuple):
                rel = ' -> '.join(str(r) for r in rel)
            layers = item.get('layers', [])
            layer_text = f"layers={layers}" if layers else "layers=[]"
            lines.append(
                f"  {idx:02d}. path={rel} | {layer_text} | mean_attention={item['mean_attention']:.6f}"
            )
        return "\n".join(lines)

    def print_route_attention_summary(self, title='HGT attended paths (sorted by mean attention):', limit=None):
        """Print all captured attention paths in descending order."""
        report = self.route_attention_report(title=title, limit=limit)
        print(report)
        return self.route_attention_summary()[:limit] if limit is not None else self.route_attention_summary()

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
        n_ancestry = len(data['ancestry'].x) if 'ancestry' in data.node_types and hasattr(data['ancestry'], 'x') else 1
        self.domain_adv = DomainAdversary(hidden, n_domains=n_ancestry)

    def forward(self, data):
        node_types = list(data.node_types)
        x_dict = {nt: data[nt].x for nt in node_types if hasattr(data[nt], 'x')}
        edge_index_dict = {
            rel: data[rel].edge_index for rel in data.edge_types if hasattr(data[rel], 'edge_index')
        }
        missing = [nt for nt in node_types if nt not in x_dict]
        if missing:
            raise KeyError(
                f"Missing node features for HGT forward pass: {missing}. "
                f"node_types={node_types}; edge_types={list(data.edge_types)[:10]}"
            )
        source_types = {src for src, _, _ in edge_index_dict.keys()}
        missing_sources = sorted(source_types - set(x_dict.keys()))
        if missing_sources:
            raise KeyError(
                f"Edges reference source node types without features: {missing_sources}. "
                f"Available x_dict keys: {sorted(x_dict.keys())}"
            )
        z = self.encoder(x_dict, edge_index_dict)
        return z

    def link_logits(self, z, edge_index, rel):
        src_t, _, dst_t = rel
        key = '__'.join(rel)
        h_src = z[src_t][edge_index[0]]
        h_dst = z[dst_t][edge_index[1]]
        return self.scorer(h_src, h_dst, key)