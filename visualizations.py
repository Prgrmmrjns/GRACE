import matplotlib.pyplot as plt
import networkx as nx

def visualize_feature_interaction_graph(G: nx.Graph, dataset_name: str):
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    plt.figure(figsize=(20, 20))
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800, alpha=1)
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=1.5)
    labels = {node: str(node).replace('_', '\n') if len(str(node)) > 10 else str(node) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
    plt.axis('off')
    plt.savefig(f"images/{dataset_name}_kg.png", dpi=300, bbox_inches='tight')
    plt.close()