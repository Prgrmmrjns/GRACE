import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
from params import METRIC, DATASET_NAME
import matplotlib.patches as patches

def visualize_pareto_front(study, dataset_name, selected_trial):
    """
    Visualizes the Pareto front from an Optuna study.
    """
    # Extracting data for plotting
    trials = [t for t in study.trials if t.values is not None]
    pareto_trials = study.best_trials
    
    non_pareto_trials = [t for t in trials if t not in pareto_trials]
    
    # Non-Pareto data
    non_pareto_scores = [t.values[0] for t in non_pareto_trials]
    non_pareto_complexities = [t.values[1] for t in non_pareto_trials]
    
    # Pareto data
    pareto_scores = [t.values[0] for t in pareto_trials]
    pareto_complexities = [t.values[1] for t in pareto_trials]

    plt.figure(figsize=(12, 8))
    plt.scatter(non_pareto_complexities, non_pareto_scores, c='gray', alpha=0.5, label='Dominated Trials')
    plt.scatter(pareto_complexities, pareto_scores, c='blue', s=80, label='Pareto Front')
    
    # Highlight the selected trial
    plt.scatter(
        selected_trial.values[1], 
        selected_trial.values[0], 
        c='red', 
        s=150, 
        marker='*', 
        label='Selected Best Trial',
        edgecolors='black'
    )
    plt.xlabel("#Edges")
    plt.ylabel(f"Validation {METRIC}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"manuscript_files/images/{dataset_name}_pareto_front.png", dpi=300, bbox_inches="tight")
    plt.close()


def visualize_knowledge_graph(constraints, labels, filename, title, feature_to_labels_map):
    """Visualize knowledge graph with a unified clustered layout and pie charts for multi-group nodes."""
    plt.figure(figsize=(24, 16))
    ax = plt.gca()
    ax.set_title(title, fontsize=20, fontweight='bold')
    ax.set_aspect('equal')

    G = nx.Graph()
    all_features = set(feat for group in constraints for feat in group)
    for group in constraints:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                G.add_edge(group[i], group[j])
    G.add_nodes_from(all_features)
    
    flat_labels = [l for sublist in labels for l in (sublist if isinstance(sublist, list) else [sublist])]
    all_label_names = list(set(flat_labels))
    colors = plt.cm.Set3(range(len(all_label_names)))
    label_color_map = dict(zip(all_label_names, colors))

    # Create completely separate regions for each group
    n_groups = len(constraints)
    pos = {}
    if n_groups > 0:
        # Use larger separation distances to prevent overlap
        if n_groups == 1:
            group_centers = [(0, 0)]
        else:
            angles = np.linspace(0, 2 * np.pi, n_groups, endpoint=False)
            separation_radius = 2.0  # Reduced from 3.0
            group_centers = [(separation_radius * np.cos(a), separation_radius * np.sin(a)) for a in angles]
        
        for i, group in enumerate(constraints):
            if group:
                # Create layout for this group only
                subgraph = G.subgraph(group)
                if len(group) == 1:
                    # Single node - place at group center
                    pos[group[0]] = group_centers[i]
                else:
                    # Multiple nodes - use spring layout within a smaller radius
                    sub_pos = nx.spring_layout(subgraph, k=0.5, iterations=100, seed=42)
                    center_x, center_y = group_centers[i]
                    # Scale down the internal layout to prevent overlap between groups
                    scale_factor = 0.8
                    for node, (x, y) in sub_pos.items():
                        pos[node] = (center_x + x * scale_factor, center_y + y * scale_factor)

    single_color_nodes, multi_color_nodes = {}, {}
    for node, node_labels in feature_to_labels_map.items():
        if node in G.nodes():
            unique_labels = list(set(node_labels))
            if len(unique_labels) > 1:
                multi_color_nodes[node] = [label_color_map[l] for l in unique_labels if l in label_color_map]
            elif len(unique_labels) == 1:
                label = unique_labels[0]
                if label not in single_color_nodes: single_color_nodes[label] = []
                single_color_nodes[label].append(node)

    for label, nodes in single_color_nodes.items():
        if label in label_color_map:
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=[label_color_map[label]], node_size=1200, alpha=0.8, edgecolors='black', linewidths=1)

    if pos and multi_color_nodes:
        radius = 0.08
        for node, node_colors in multi_color_nodes.items():
            if node in pos and node_colors:
                x, y = pos[node]
                angles = np.linspace(0, 360, len(node_colors) + 1)
                for i in range(len(node_colors)):
                    ax.add_patch(patches.Wedge((x, y), radius, angles[i], angles[i+1], facecolor=node_colors[i], edgecolor='black', linewidth=1, zorder=5))

    drawn_nodes = set(multi_color_nodes.keys()) | {n for nodes in single_color_nodes.values() for n in nodes}
    remaining_nodes = [n for n in G.nodes() if n not in drawn_nodes]
    if remaining_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=remaining_nodes, node_color='lightgrey', node_size=1200, alpha=0.8, edgecolors='black', linewidths=1)

    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, edge_color='gray')
    
    for node in G.nodes():
        if node in pos:
            plt.text(pos[node][0], pos[node][1], node, ha='center', va='center', zorder=10, 
                     fontdict={'size': 10, 'weight': 'bold'}, 
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', linewidth=0.5, boxstyle='round,pad=0.2'))

    # Add legend for disease mechanisms
    legend_patches = [patches.Patch(color=color, label=label.replace('/', '\n')) for label, color in sorted(label_color_map.items())]
    plt.legend(handles=legend_patches, title='Disease Mechanisms', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=12)

    plt.axis('off')
    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make space for legend
    plt.savefig(f'manuscript_files/images/{filename}', dpi=300, bbox_inches='tight')
    plt.close()
    return G