import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
# import networkx as nx
# import numpy as np

def audit_and_visualize_heterodata(data, output_dir):
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
    plt.figure(figsize=(12, 6))
    sns.barplot(x=edge_counts, y=edge_names, hue=edge_names, palette="magma", legend=False)
    plt.xscale("log")
    plt.title("Log-Scale Count of Edges Across Relational Types")
    plt.xlabel("Number of Edges (Log Scale)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/audit_edge_counts.png")
    plt.close()

    print(f"\nVisualizations successfully saved to {output_dir}/")


def plot_training_history(history, output_dir):

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    fig.suptitle('Training history', fontsize=14)

    axes[0].plot(history['epoch'], history['train_total'], label='train loss', color='tab:blue', linewidth=1.5)
    axes[0].plot(history['epoch'], history['val_total'], label='val loss', color='tab:red', linewidth=1.5)
    axes[0].set_ylabel('loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    ax2 = axes[0].twinx()
    ax2.plot(history['epoch'], history['variant_ap'], color='tab:green', alpha=0.7, linestyle='--', label='variant AP')
    ax2.plot(history['epoch'], history['drug_ap'], color='tab:purple', alpha=0.7, linestyle='--', label='drug AP')
    ax2.set_ylabel('AP')
    ax2.legend(loc='upper right')

    axes[1].plot(history['epoch'], history['variant_auroc'], label='variant AUROC', color='tab:green')
    axes[1].plot(history['epoch'], history['drug_auroc'], label='drug AUROC', color='tab:purple')
    axes[1].plot(history['epoch'], history['variant_ap'], label='variant AP', color='tab:green', linestyle='--', alpha=0.7)
    axes[1].plot(history['epoch'], history['drug_ap'], label='drug AP', color='tab:purple', linestyle='--', alpha=0.7)
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