import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Patch
from typing import List, Dict

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

def plot_shap_waterfall(shap_values, feature_names, base_value, sample_idx=0, dataset_name="model", max_display=10):
    sample_shap = shap_values[sample_idx]
    feature_importance = [(name, val) for name, val in zip(feature_names, sample_shap)]
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    feature_names_all = [f[0] for f in feature_importance]
    shap_values_all = [f[1] for f in feature_importance]
    fig, ax = plt.subplots(figsize=(14, 8))
    cumulative = [base_value]
    for val in shap_values_all:
        cumulative.append(cumulative[-1] + val)
    final_prediction = cumulative[-1]
    ax.bar(0, base_value, color='gray', alpha=0.7, width=0.6)
    ax.text(0, base_value/2, f'Base\n{base_value:.3f}', ha='center', va='center', fontweight='bold', fontsize=9)
    colors = ['red' if val < 0 else 'blue' for val in shap_values_all]
    for i, (name, val) in enumerate(zip(feature_names_all, shap_values_all)):
        x_pos = i + 1
        bottom = cumulative[i] if val > 0 else cumulative[i+1]
        height = abs(val)
        ax.bar(x_pos, height, bottom=bottom, color=colors[i], alpha=0.7, width=0.6)
        ax.text(x_pos, bottom + height/2, f'{val:+.3f}', ha='center', va='center', fontweight='bold', fontsize=9)
        if i < len(shap_values_all) - 1:
            ax.plot([x_pos + 0.3, x_pos + 0.7], [cumulative[i+1], cumulative[i+1]], 'k--', alpha=0.5, linewidth=1)
    final_x = len(shap_values_all) + 1
    ax.bar(final_x, final_prediction, color='green', alpha=0.7, width=0.6)
    ax.text(final_x, final_prediction/2, f'Final\n{final_prediction:.3f}', ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Formatting
    x_labels = ['Base Value'] + [name.replace('_', '\n') for name in feature_names_all] + ['Final Prediction']
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_ylabel('Prediction Value')
    ax.set_title(f'SHAP Waterfall Plot - Sample {sample_idx}\n{dataset_name.upper()}')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=1)
    
    # Add legend
    
    legend_elements = [
        Patch(facecolor='gray', alpha=0.7, label='Base Value'),
        Patch(facecolor='blue', alpha=0.7, label='Positive Contribution'),
        Patch(facecolor='red', alpha=0.7, label='Negative Contribution'),
        Patch(facecolor='green', alpha=0.7, label='Final Prediction')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f"images/{dataset_name}_shap_waterfall.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_shap_interaction_network(shap_values: np.ndarray, feature_names: List[str], 
                                feature_interactions: List[Dict], dataset_name: str = "model"):
    """
    Create a network graph showing SHAP-based feature interactions using only the 
    feature interaction constraints from the knowledge graph.
    
    Args:
        shap_values: SHAP values array (n_samples, n_features)
        feature_names: List of feature names
        feature_interactions: List of feature interaction dictionaries from knowledge graph
        dataset_name: Name for saving the plot
    """
    
    # Calculate correlation matrix of SHAP values
    shap_corr = np.corrcoef(shap_values.T)
    
    # Calculate mean absolute SHAP values for node sizing
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    
    # Create network graph
    G = nx.Graph()
    
    # Add nodes with sizes based on SHAP importance
    max_shap = np.max(mean_abs_shap)
    min_size, max_size = 300, 2000
    for i, feature in enumerate(feature_names):
        # Scale node size based on SHAP importance
        normalized_importance = mean_abs_shap[i] / max_shap
        node_size = min_size + (normalized_importance * (max_size - min_size))
        G.add_node(feature, size=node_size, importance=mean_abs_shap[i])
    
    # Add edges only from feature interaction constraints
    for interaction_obj in feature_interactions:
        if interaction_obj['feature'] in feature_names:
            f1_idx = feature_names.index(interaction_obj['feature'])
            
            for target_feature in interaction_obj['interactions']:
                if target_feature in feature_names:
                    f2_idx = feature_names.index(target_feature)
                    
                    # Get SHAP correlation for this specific interaction
                    if f1_idx != f2_idx and not np.isnan(shap_corr[f1_idx, f2_idx]):
                        correlation = shap_corr[f1_idx, f2_idx]
                        abs_correlation = abs(correlation)
                        
                        # Add edge with SHAP correlation strength
                        G.add_edge(interaction_obj['feature'], target_feature, 
                                  weight=abs_correlation, correlation=correlation)
    
    # Create visualization
    plt.figure(figsize=(16, 12))
    
    # Layout
    pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
    
    # Draw nodes with uniform color and size based on importance
    node_sizes = [G.nodes[node]['size'] for node in G.nodes()]
    
    nodes = nx.draw_networkx_nodes(G, pos, 
                                  node_size=node_sizes,
                                  node_color='lightblue',  # Uniform color
                                  alpha=0.8,
                                  edgecolors='black',
                                  linewidths=1)
    
    # Draw edges with width based on interaction strength
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    
    # Scale edge widths
    if edge_weights:
        max_weight = max(edge_weights)
        min_width, max_width = 0.5, 5.0
        scaled_widths = [min_width + (w/max_weight) * (max_width - min_width) for w in edge_weights]
    else:
        scaled_widths = [1.0] * len(G.edges())
    
    edges = nx.draw_networkx_edges(G, pos,
                                    width=scaled_widths,
                                    edge_color='gray',
                                    alpha=0.6)
    
    # Draw labels
    labels = {node: node.replace('_', '\n') if len(node) > 12 else node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"images/{dataset_name}_shap_interactions.png", dpi=300, bbox_inches='tight')
    plt.close()