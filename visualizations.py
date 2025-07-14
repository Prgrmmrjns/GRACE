import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from params import METRIC
import matplotlib.patches as patches
import textwrap
from matplotlib.ticker import MaxNLocator

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
    plt.xlabel("# Possible Feature Interactions")
    plt.ylabel(f"Validation {METRIC}")
    plt.legend()
    plt.grid(True)

    # Force integer ticks on x-axis
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.savefig(f"manuscript_files/images/{dataset_name}_pareto_front.png", dpi=300, bbox_inches="tight")
    plt.close()


def visualize_knowledge_graph(constraints, labels, filename, feature_to_labels_map, edge_labels=None):
    """Visualize knowledge graph with a dynamic, clustered layout and better space utilization."""
    plt.figure(figsize=(22, 22)) # More square figure
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box') # Prevent stretching

    G = nx.Graph()
    all_features = set(feat for group in constraints for feat in group)
    if not all_features:
        plt.close()
        return

    for group in constraints:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                G.add_edge(group[i], group[j])
    G.add_nodes_from(all_features)
    
    # --- Dynamic Sizing ---
    num_nodes = len(all_features)
    # Scale sizes inversely with the number of nodes. More nodes = smaller sizes.
    node_scale = max(0.5, 60 / (num_nodes + 20))
    node_size = 1500 * node_scale
    # The radius of pie charts should scale with the square root of the node area.
    # The font size can scale linearly with the node scale for predictability.
    pie_radius = 0.08 * np.sqrt(node_scale)
    font_size = 10 * node_scale

    flat_labels = [l for sublist in labels for l in (sublist if isinstance(sublist, list) else [sublist])]
    all_label_names = list(set(flat_labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_label_names)))
    label_color_map = dict(zip(all_label_names, colors))

    # --- Tighter Layout ---
    n_groups = len(constraints)
    pos = {}
    if n_groups > 0:
        if n_groups == 1:
            group_centers = [(0, 0)]
            separation_radius = 0
        else:
            # Reduce separation radius to bring groups closer
            separation_radius = 1.5 if n_groups > 4 else 1.8
            angles = np.linspace(0, 2 * np.pi, n_groups, endpoint=False)
            group_centers = [(separation_radius * np.cos(a), separation_radius * np.sin(a)) for a in angles]
        
        for i, group in enumerate(constraints):
            if group:
                subgraph = G.subgraph(group)
                if len(group) == 1:
                    pos[group[0]] = group_centers[i]
                else:
                    # Use a smaller k for tighter packing within the group
                    sub_pos = nx.spring_layout(subgraph, k=0.8, iterations=100, seed=42)
                    center_x, center_y = group_centers[i]
                    # Use a larger scale factor to expand the group layout
                    scale_factor = 1.0
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
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=[label_color_map[label]], node_size=node_size, alpha=0.9, edgecolors='black', linewidths=1.5)

    if pos and multi_color_nodes:
        for node, node_colors in multi_color_nodes.items():
            if node in pos and node_colors:
                x, y = pos[node]
                angles = np.linspace(0, 360, len(node_colors) + 1)
                for i in range(len(node_colors)):
                    ax.add_patch(patches.Wedge((x, y), pie_radius, angles[i], angles[i+1], facecolor=node_colors[i], edgecolor='black', linewidth=1.5, zorder=5))

    drawn_nodes = set(multi_color_nodes.keys()) | {n for nodes in single_color_nodes.values() for n in nodes}
    remaining_nodes = [n for n in G.nodes() if n not in drawn_nodes]
    if remaining_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=remaining_nodes, node_color='lightgrey', node_size=node_size, alpha=0.9, edgecolors='black', linewidths=1.5)

    nx.draw_networkx_edges(G, pos, alpha=0.4, width=1.0, edge_color='gray')
    
    if edge_labels:
        # Ensure keys are tuples of nodes present in the graph for labeling
        valid_edge_labels = {}
        for (u, v), label in edge_labels.items():
            # networkx can store edges as (u,v) or (v,u), so check both
            if G.has_edge(u, v):
                valid_edge_labels[(u, v)] = label
            elif G.has_edge(v, u):
                valid_edge_labels[(v, u)] = label
        
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=valid_edge_labels,
            font_color='crimson',
            font_size=max(8, font_size * 0.85), # scale with node font size
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.2)
        )

    for node in G.nodes():
        if node in pos:
            plt.text(pos[node][0], pos[node][1], node, ha='center', va='center', zorder=10, 
                     fontdict={'size': font_size, 'weight': 'bold'}, 
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', linewidth=0.5, boxstyle='round,pad=0.3'))

    # --- Better Legend Placement ---
    # Wrap long labels and place legend at the bottom center
    legend_patches = [patches.Patch(color=color, label=textwrap.fill(label.replace('/', '\n'), width=40)) for label, color in sorted(label_color_map.items())]
    legend = plt.legend(handles=legend_patches, title='Disease Mechanisms', 
               bbox_to_anchor=(0.5, -0.02), loc='upper center', 
               borderaxespad=0., fontsize=14, ncol=min(3, len(legend_patches)))
    plt.setp(legend.get_title(),fontsize=16)
    
    plt.axis('off')
    plt.tight_layout(rect=[0, 0.05, 1, 1]) # Reserve space at the bottom for the legend
    plt.savefig(f'manuscript_files/images/{filename}', dpi=300, bbox_inches='tight')
    plt.close()