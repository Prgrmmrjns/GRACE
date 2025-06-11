import networkx as nx
import json
import numpy as np
from typing import List, Dict
from models import KnowledgeGraphModel

def model_to_networkx(kg_model: KnowledgeGraphModel) -> nx.Graph:
    G = nx.Graph()
    for node in kg_model.nodes:
        attrs = {'node_type': node.node_type}
        if node.attributes:
            attrs.update(node.attributes)
        G.add_node(node.id, **attrs)
    for edge in kg_model.edges:
        attrs = {'relationship': edge.relationship}
        if edge.attributes:
            attrs.update(edge.attributes)
        G.add_edge(edge.source, edge.target, **attrs)
    return G

def networkx_to_json(G: nx.Graph) -> str:
    return json.dumps(nx.node_link_data(G), indent=2)

def get_knowledge_groups(G: nx.Graph) -> Dict[str, List[str]]:
    """Extracts feature groups (features connected to the same mechanism) from the knowledge graph."""
    groups = {}
    # Iterate through all nodes and find the ones that are disease mechanisms
    for node_id, data in G.nodes(data=True):
        if data.get('node_type') == 'disease_mechanism':
            # For each mechanism, find all neighboring feature nodes
            groups[node_id] = [
                neighbor for neighbor in G.neighbors(node_id) 
                if G.nodes[neighbor].get('node_type') == 'feature'
            ]
    return groups

def select_features_by_shap_contribution(feature_names: list, feature_contributions: np.ndarray, threshold=0.95) -> list:
    """Selects the top features that contribute to a certain percentage of the total SHAP importance."""
    sorted_indices = np.argsort(feature_contributions)[::-1]
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_contributions = feature_contributions[sorted_indices]
    cumulative_contributions = np.cumsum(sorted_contributions)
    total_contribution = np.sum(sorted_contributions)
    n_features_threshold = np.argmax(cumulative_contributions / total_contribution >= threshold) + 1
    return sorted_features[:n_features_threshold]