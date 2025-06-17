import matplotlib.pyplot as plt
import networkx as nx
import os
from params import DATASET_NAME


def visualize_pareto_front(scores, graph_sizes, pareto_scores, pareto_graph_sizes):
    """
    Visualizes the Pareto front from optimization results and saves it as a PNG file.
    """
    plt.figure(figsize=(10, 8))
    
    plt.scatter(graph_sizes, scores, c='gray', alpha=0.5, label='Dominated Solutions')
    plt.scatter(pareto_graph_sizes, pareto_scores, c='blue', alpha=0.9, label='Pareto Front')

    plt.title(f'Pareto Front: Score vs. Graph Size')
    plt.xlabel('Graph Size (Nodes + Edges)')
    plt.ylabel('Validation Score (AUC)')
    plt.legend()
    plt.grid(True)
    os.makedirs('images', exist_ok=True)
    plot_path = f'images/{DATASET_NAME}_pareto_front.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()

def visualize_optimized_graph(final_nodes, final_edges):
    """Visualizes the final optimized feature interaction graph."""
    final_graph = nx.Graph()
    final_graph.add_nodes_from(final_nodes)
    final_graph.add_edges_from(list(final_edges))

    final_graph = nx.DiGraph(final_graph)
    final_graph.remove_nodes_from(list(nx.isolates(final_graph)))

    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(final_graph, k=2, iterations=50, seed=42)
    nx.draw_networkx_nodes(final_graph, pos, node_color='lightblue', node_size=3000, edgecolors='black', linewidths=2)
    nx.draw_networkx_edges(final_graph, pos, edge_color='gray', alpha=0.7, width=2, arrows=True, arrowsize=20)
    labels = {n: str(n).replace('_', ' ').title() for n in final_graph.nodes()}
    nx.draw_networkx_labels(final_graph, pos, labels, font_size=12, font_weight='bold')
    plt.axis('off'); plt.tight_layout()

    filename = f'images/{DATASET_NAME}_optimized_graph.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_kg_structure(dataset_name: str):
    """Visualize knowledge graph with intermediate and input nodes"""
    G = nx.read_graphml(f'kg/{dataset_name}_initial_agent_kg.graphml')
    
    plt.figure(figsize=(20, 20)) # Increased figure size for better spacing
    
    # Separate nodes by type
    input_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'input']
    intermediate_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'intermediate']
    
    # Create a radial layout
    pos = {}
    # Position intermediate nodes in the center
    if intermediate_nodes:
        # Create a smaller spring layout for just the intermediate nodes to cluster them
        intermediate_subgraph = G.subgraph(intermediate_nodes)
        pos_intermediate = nx.spring_layout(intermediate_subgraph, k=0.5, iterations=100, seed=42)
        # Scale down the layout and center it
        for node, coords in pos_intermediate.items():
            pos[node] = coords * 0.2
    
    # Position input nodes in a circle around the center
    if input_nodes:
        # Use a circular layout for the input nodes
        pos_input = nx.circular_layout(G.subgraph(input_nodes), scale=1.5)
        pos.update(pos_input)

    # If no positions were calculated (e.g., all one type), default to spring layout
    if not pos:
        pos = nx.spring_layout(G, k=3, iterations=300, seed=42)

    # Draw intermediate nodes (larger, different color)
    nx.draw_networkx_nodes(G, pos, nodelist=intermediate_nodes, 
                          node_color='lightcoral', node_size=5000, # Increased size
                          edgecolors='darkred', linewidths=2, alpha=0.9)
    
    # Draw input nodes (smaller, different color)
    nx.draw_networkx_nodes(G, pos, nodelist=input_nodes,
                          node_color='lightblue', node_size=2500, # Increased size
                          edgecolors='darkblue', linewidths=1, alpha=0.9)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5, width=1.5, connectionstyle='arc3,rad=0.1')
    
    # Draw labels
    labels = {}
    for n in G.nodes():
        # Wrap long names
        name = str(n).replace('_', ' ')
        if len(name) > 20: # Adjusted for larger plot
            words = name.split()
            if len(words) > 2:
                mid = len(words) // 2
                name = ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])
        labels[n] = name
    
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
    
    # Add legend
    intermediate_patch = plt.Line2D([0], [0], marker='o', color='w', 
                                   markerfacecolor='lightcoral', markersize=20,
                                   label=f'Intermediate Nodes ({len(intermediate_nodes)})')
    input_patch = plt.Line2D([0], [0], marker='o', color='w',
                            markerfacecolor='lightblue', markersize=15,
                            label=f'Input Nodes ({len(input_nodes)})')
    plt.legend(handles=[intermediate_patch, input_patch], loc='upper right', fontsize=12)
    
    plt.title(f'{dataset_name.upper()} Knowledge Graph Structure\n'
              f'{len(intermediate_nodes)} Intermediate Nodes, {len(input_nodes)} Input Nodes, {len(G.edges())} Connections',
              fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    os.makedirs('images', exist_ok=True)
    plt.savefig(f'images/{dataset_name}_kg_structure.png', dpi=300, bbox_inches='tight')
    plt.close()