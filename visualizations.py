import matplotlib.pyplot as plt
import networkx as nx
import os
from matplotlib.patches import Patch

def get_knowledge_groups(G):
    return {}

def visualize_knowledge_graph(G: nx.Graph, dataset_name: str):
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G, k=2, iterations=200, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=3000, edgecolors='black', linewidths=2)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.7, width=2)
    
    # Draw labels
    labels = {n: str(n).replace('_', ' ').title() for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=12, font_weight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"images/{dataset_name}_feature_interaction_graph.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualized {len(G.nodes())} connected features with {len(G.edges())} interactions") 