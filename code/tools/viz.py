import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
# import networkx as nx
# import numpy as np


def _parse_relation_tokens(value):
    if value is None or value == '':
        return set()
    raw = str(value).strip()
    if not raw or raw.lower() in {'none', 'all'}:
        return set()

    raw = raw.replace(';', '|').replace('-', ',')
    groups = [group.strip() for group in raw.split('|') if group.strip()]

    out = set()
    for group in groups:
        tokens = [token.strip().strip("()[]'\"") for token in group.split(',') if token.strip()]
        if len(tokens) == 3:
            out.add(tuple(tokens))
    return out


def audit_and_visualize_heterodata(data, output_dir, ignored_relations='', original_edge_counts=None):
    os.makedirs(output_dir, exist_ok=True)
    print("=== HETEROGENEOUS GRAPH AUDIT REPORT ===")

    # 1. Print Node Tensor Shapes
    print("\n[Node Feature Matrices]")
    node_stats = {}
    for node_type in data.node_types:
        num_nodes = data[node_type].x.shape[0]
        num_features = data[node_type].x.shape[1]
        node_stats[node_type] = num_nodes
        print(f"  - {node_type:<12}: {num_nodes} nodes, {num_features} features")

    # 2. Print Edge Tensor Shapes and Statistics
    print("\n[Edge Connections & Weights]")
    edge_counts = []
    edge_names = []
    for edge_type in data.edge_types:
        src, rel, dst = edge_type
        num_edges = data[edge_type].edge_index.shape[1]
        has_weight = hasattr(data[edge_type], 'edge_weight')
        has_attr = hasattr(data[edge_type], 'edge_attr')

        edge_label = f"{src} -> {rel} -> {dst}"
        edge_names.append(edge_label)
        edge_counts.append(num_edges)

        weight_str = f"| Weights: Yes" if has_weight else "| Weights: No"
        attr_str = f"| Attr: Yes" if has_attr else ""
        print(f"  - {edge_label:<45}: {num_edges} edges {weight_str} {attr_str}")

    ignored_set = _parse_relation_tokens(ignored_relations)
    if ignored_set and original_edge_counts is not None:
        print("\n[Relation Filters Applied]")
        ignored_labels = []
        ignored_values = []
        for rel, num_edges in sorted(original_edge_counts.items(), key=lambda x: str(x[0])):
            if tuple(rel) in ignored_set and tuple(rel) not in {tuple(edge) for edge in data.edge_types}:
                label = f"{rel[0]} -> {rel[1]} -> {rel[2]}"
                ignored_labels.append(label)
                ignored_values.append(num_edges)
                print(f"  - {label:<45}: {num_edges} edges | Status: Filtered")

        if ignored_labels:
            plt.figure(figsize=(12, 6))
            sns.barplot(x=ignored_values, y=ignored_labels, hue=ignored_labels, palette="Reds", legend=False)
            plt.xscale("log")
            plt.title("Relation Filters Applied")
            plt.xlabel("Original Number of Edges (Log Scale)")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/audit_ignored_edges.png")
            plt.close()

    # --- VISUALIZATION 1: Node Count Bar Plot ---
    plt.figure(figsize=(10, 5))
    keys = list(node_stats.keys())
    sns.barplot(x=keys, y=list(node_stats.values()), hue=keys, palette="viridis", legend=False)
    plt.yscale("log")
    plt.title("Log-Scale Count of Nodes Across Graph Types")
    plt.ylabel("Number of Nodes (Log Scale)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/audit_node_counts.png")
    plt.close()

    # --- VISUALIZATION 2: Edge Density Bar Plot ---
    if ignored_set:
        plot_labels = edge_names + [f"FILTERED: {rel[0]} -> {rel[1]} -> {rel[2]}" for rel in sorted(ignored_set, key=lambda x: str(x)) if tuple(rel) not in {tuple(edge) for edge in data.edge_types}]
        plot_counts = edge_counts + [original_edge_counts.get(tuple(rel), 0) for rel in sorted(ignored_set, key=lambda x: str(x)) if tuple(rel) not in {tuple(edge) for edge in data.edge_types}]
        title = "Log-Scale Count of Included and Filtered Edge Relations"
    else:
        plot_labels = edge_names
        plot_counts = edge_counts
        title = "Log-Scale Count of Included Edge Relations"

    if plot_labels:
        plt.figure(figsize=(12, 6))
        sns.barplot(x=plot_counts, y=plot_labels, hue=plot_labels, palette="magma", legend=False)
        plt.xscale("log")
        plt.title(title)
        plt.xlabel("Number of Edges (Log Scale)")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/audit_edge_counts.png")
        plt.close()

    print(f"\nVisualizations successfully saved to {output_dir}/")


def plot_score_histogram(pos_scores, neg_scores, output_path, title='Positive vs negative score histogram'):
    pos_scores = np.asarray(pos_scores).reshape(-1)
    neg_scores = np.asarray(neg_scores).reshape(-1)
    if pos_scores.size == 0 or neg_scores.size == 0:
        return

    plt.figure(figsize=(9, 5))
    plt.hist(pos_scores, bins=40, density=True, alpha=0.7, color='tab:green', label='positive')
    plt.hist(neg_scores, bins=40, density=True, alpha=0.7, color='tab:red', label='negative')
    plt.xlabel('score')
    plt.ylabel('density')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_roc_curve(pos_scores, neg_scores, output_path, title='ROC curve'):
    pos_scores = np.asarray(pos_scores).reshape(-1)
    neg_scores = np.asarray(neg_scores).reshape(-1)
    if pos_scores.size == 0 or neg_scores.size == 0:
        return

    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([
        np.ones(len(pos_scores), dtype=int),
        np.zeros(len(neg_scores), dtype=int),
    ])

    thresholds = np.unique(scores)
    fpr = []
    tpr = []
    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)
        tp = np.sum((pred == 1) & (labels == 1))
        fp = np.sum((pred == 1) & (labels == 0))
        tn = np.sum((pred == 0) & (labels == 0))
        fn = np.sum((pred == 0) & (labels == 1))

        tpr_val = tp / max(1, np.sum(labels == 1))
        fpr_val = fp / max(1, np.sum(labels == 0))
        tpr.append(tpr_val)
        fpr.append(fpr_val)

    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)
    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]

    fpr = np.concatenate([[0.0], fpr, [1.0]])
    tpr = np.concatenate([[0.0], tpr, [1.0]])

    plt.figure(figsize=(7, 7))
    plt.plot(fpr, tpr, color='tab:blue', linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_pr_curve(pos_scores, neg_scores, output_path, title='Precision-recall curve'):
    pos_scores = np.asarray(pos_scores).reshape(-1)
    neg_scores = np.asarray(neg_scores).reshape(-1)
    if pos_scores.size == 0 or neg_scores.size == 0:
        return

    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    order = np.argsort(-scores, kind='stable')
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    precision = tp / (tp + fp)
    recall = tp / labels.sum()
    baseline = labels.mean()
    ap_val = float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))

    plt.figure(figsize=(7, 7))
    plt.plot(recall, precision, color='tab:blue', linewidth=2, label=f'AP = {ap_val:.3f}')
    plt.axhline(baseline, linestyle='--', color='gray', linewidth=1, label=f'baseline = {baseline:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim(0, 1.02)
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_training_history(history, output_dir):

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    fig.suptitle('Training history', fontsize=14)

    axes[0].plot(history['epoch'], history['train_total'], label='train loss', color='tab:blue', linewidth=1.5)
    axes[0].plot(history['epoch'], history['val_total'], label='val loss', color='tab:red', linewidth=1.5)
    axes[0].set_ylabel('loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    alias_to_name = history.get('task_display_names', {})
    task_aliases = history.get('task_aliases', [])
    color_map = {
        'variant_phenotype': 'tab:green',
        'drug_side_effect': 'tab:purple',
    }
    for alias in task_aliases:
        display = alias_to_name.get(alias, alias.replace('_', '-'))
        color = color_map.get(alias, None)
        axes[1].plot(
            history['epoch'],
            history[f'{alias}_auroc'],
            label=f'{display} AUROC',
            color=color,
        )
        axes[1].plot(
            history['epoch'],
            history[f'{alias}_ap'],
            label=f'{display} AP',
            color=color,
            linestyle='--',
            alpha=0.8,
        )

    axes[1].set_ylabel('metric')
    axes[1].legend(loc='best', fontsize=8)
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(history['epoch'], history['epoch_time'], color='tab:orange', linewidth=1.5)
    axes[2].set_xlabel('epoch')
    axes[2].set_ylabel('time (s)')
    axes[2].grid(True, alpha=0.25)

    if len(np.unique(history['lr'])) > 1:
        fig_lr, ax_lr = plt.subplots(figsize=(12, 2.6))
        ax_lr.plot(history['epoch'], history['lr'], color='tab:blue', linewidth=1.5)
        ax_lr.set_xlabel('epoch')
        ax_lr.set_ylabel('learning rate')
        ax_lr.grid(True, alpha=0.25)
        fig_lr.tight_layout()
        fig_lr.savefig(f"{output_dir}/learning_rate.png", dpi=200)
        plt.close(fig_lr)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{output_dir}/training_history.png", dpi=200)
    plt.close(fig)