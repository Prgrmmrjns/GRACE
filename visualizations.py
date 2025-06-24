import matplotlib.pyplot as plt
import networkx as nx
import os
from params import DATASET_NAME, TARGET_COL
import itertools
import matplotlib.cm as cm
import numpy as np


def visualize_pareto_front(scores, graph_sizes, pareto_scores, pareto_graph_sizes, best_score, best_size):
    """
    Visualizes the Pareto front from optimization results and saves it as a PNG file.
    """
    plt.figure(figsize=(10, 8))
    
    # Plot all trials and the pareto front
    plt.scatter(graph_sizes, scores, c='gray', alpha=0.5, label='Dominated Solutions')
    plt.scatter(pareto_graph_sizes, pareto_scores, c='blue', alpha=0.9, label='Pareto Front')
    
    # Highlight the chosen best solution
    plt.scatter([best_size], [best_score], c='red', s=200, edgecolor='black', marker='*', zorder=10, label=f'Chosen Solution (Score: {best_score:.4f}, Size: {best_size})')

    plt.title(f'Pareto Front: Score vs. Graph Size')
    plt.xlabel('Graph Size (Number of Features + Number of Edges)')
    plt.ylabel('Validation Score (AUC)')
    plt.legend()
    plt.grid(True)

    os.makedirs('images', exist_ok=True)
    plot_path = f'images/{DATASET_NAME}_pareto_front.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Pareto front plot saved to {plot_path}")

def visualize_optimized_graph(final_nodes, final_constraints):
    """Visualizes the final optimized feature graph with interaction clusters."""
    
    G = nx.Graph()

    # Add Nodes: Target and Features (Inputs)
    G.add_node(TARGET_COL, node_type='target')
    G.add_nodes_from(final_nodes, node_type='input')
    
    # Add Edges
    for node in final_nodes: G.add_edge(node, TARGET_COL)
    for constraint_group in final_constraints:
        for u, v in itertools.combinations(constraint_group, 2):
            if u in G and v in G: G.add_edge(u, v)

    G.remove_nodes_from(list(nx.isolates(G)))
    plt.figure(figsize=(28, 28))

    # Spring layout will naturally cluster the densely connected interaction groups
    pos = nx.spring_layout(G, k=1.5/np.sqrt(len(G.nodes())), iterations=100, seed=42)

    target_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'target']
    input_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'input']

    nx.draw_networkx_nodes(G, pos, nodelist=target_nodes, node_color='gold', node_size=15000, edgecolors='black', linewidths=2.5)
    nx.draw_networkx_nodes(G, pos, nodelist=input_nodes, node_color='lightblue', node_size=4000, edgecolors='darkblue', linewidths=1.5)

    # --- Draw Edges in Colored Layers ---
    target_edges = [(n, TARGET_COL) for n in input_nodes if G.has_edge(n, TARGET_COL)]
    nx.draw_networkx_edges(G, pos, edgelist=target_edges, edge_color='gray', alpha=0.5, width=1.5, style='dashed')

    valid_groups = [g for g in final_constraints if len(g) > 1]
    colors = cm.get_cmap('tab10', len(valid_groups) if valid_groups else 1)
    edge_legend_handles = []
    for i, group in enumerate(valid_groups):
        group_edges = list(itertools.combinations(group, 2))
        group_edges = [(u, v) for u, v in group_edges if u in G and v in G]
        nx.draw_networkx_edges(G, pos, edgelist=group_edges, edge_color=[colors(i)], width=2.5, alpha=0.9)
        edge_legend_handles.append(plt.Line2D([0], [0], color=colors(i), lw=3, label=f'Interaction Group {i+1}'))
    
    labels = {n: str(n).replace('_', ' ').title() for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=12, font_weight='bold')

    # --- Create Legend ---
    node_legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gold', markersize=30, label='Target Node'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=20, label='Input Feature')
    ]
    if target_edges:
        edge_legend_handles.insert(0, plt.Line2D([0], [0], color='gray', lw=2, ls='dashed', label='Connection to Target'))
    
    plt.legend(handles=node_legend_handles + edge_legend_handles, loc='best', fontsize=16)
    
    plt.title('Optimized Model Structure with Interaction Clusters', fontsize=28, fontweight='bold')
    plt.axis('off'); plt.tight_layout()

    filename = f'images/{DATASET_NAME}_optimized_graph.png'
    os.makedirs('images', exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Optimized graph visualization saved to {filename}")

def visualize_kg_structure(dataset_name: str):
    """Visualize knowledge graph with intermediate and input nodes"""
    G = nx.read_graphml(f'kg/{dataset_name}_initial_agent_kg.graphml')
    
    plt.figure(figsize=(24, 24))
    
    # Separate nodes by type
    input_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'input']
    intermediate_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'intermediate']
    target_node = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'target']

    # Create a shell layout for concentric circles
    shells = []
    if target_node:
        shells.append(target_node)
    if intermediate_nodes:
        shells.append(intermediate_nodes)
    if input_nodes:
        shells.append(input_nodes)
    
    # Add any remaining nodes to the outermost shell
    all_categorized = set(target_node + intermediate_nodes + input_nodes)
    remaining_nodes = [n for n in G.nodes() if n not in all_categorized]
    if remaining_nodes:
        if shells:
            shells[-1].extend(remaining_nodes)  # Add to the last shell
        else:
            shells.append(remaining_nodes)  # Create a new shell if none exist
    
    pos = nx.shell_layout(G, nlist=shells, scale=2)

    # Manually adjust shell radius for better spacing
    if len(shells) > 1 and target_node:
        for node in shells[0]: # Target node
            pos[node] *= 0 # Center it
    if len(shells) > 2 and intermediate_nodes:
         for node in shells[1]: # Intermediate nodes
            pos[node] *= 0.5

    # Draw nodes
    if target_node:
            nx.draw_networkx_nodes(G, pos, nodelist=target_node,
                          node_color='gold', node_size=12000,
                          edgecolors='black', linewidths=2, alpha=1.0)
    
    nx.draw_networkx_nodes(G, pos, nodelist=intermediate_nodes, 
                          node_color='lightcoral', node_size=6000, 
                          edgecolors='darkred', linewidths=2, alpha=0.9)
    
    nx.draw_networkx_nodes(G, pos, nodelist=input_nodes,
                          node_color='lightblue', node_size=3000,
                          edgecolors='darkblue', linewidths=1, alpha=0.9)
    
    # Draw edges - filter out edges with nodes not in the graph
    valid_edges = [(u, v) for u, v in G.edges() if u in pos and v in pos]
    if valid_edges:
        nx.draw_networkx_edges(G, pos, edgelist=valid_edges, edge_color='gray', alpha=0.6, width=1.5)
    
    # Draw labels - only for nodes that have positions
    labels = {}
    for n in G.nodes():
        if n in pos:  # Only include nodes that have positions
            name = str(n).replace('_', ' ')
            if len(name) > 20:
                words = name.split()
                if len(words) > 2:
                    mid = len(words) // 2
                    name = ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])
            labels[n] = name
    
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
    
    # Add legend
    handles = []
    if target_node:
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gold', markersize=25, label=f'Target Node ({len(target_node)})'))
    if intermediate_nodes:
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', markersize=20, label=f'Intermediate Nodes ({len(intermediate_nodes)})'))
    if input_nodes:
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=15, label=f'Input Nodes ({len(input_nodes)})'))
    
    plt.legend(handles=handles, loc='upper right', fontsize=12)
    
    plt.title(f'{dataset_name.upper()} Knowledge Graph Structure', fontsize=18, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    os.makedirs('images', exist_ok=True)
    plt.savefig(f'images/{dataset_name}_kg_structure.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"KG structure visualization saved to images/{dataset_name}_kg_structure.png")