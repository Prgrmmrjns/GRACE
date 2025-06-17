import networkx as nx
import os
import joblib
import shap
import numpy as np
from collections import defaultdict
import copy
from itertools import combinations
from typing import List, Optional
from pydantic import BaseModel, Field
from params import ML_MODEL

class Node(BaseModel):
    name: str = Field(..., description="Unique name of the node (can be a feature or an intermediate mechanism).")
    node_type: str = Field(..., description="Type of the node, e.g., 'input' or 'intermediate'.")
    description: Optional[str] = Field(None, description="Detailed description of what the node represents.")

class Edge(BaseModel):
    source: str = Field(..., description="Name of the source node.")
    target: str = Field(..., description="Name of the target node.")
    relationship: str = Field(..., description="Detailed description of the relationship between the source and target nodes.")

class KnowledgeGraphModel(BaseModel):
    nodes: List[Node] = Field(..., description="List of all nodes in the graph.")
    edges: List[Edge] = Field(..., description="List of all edges connecting the nodes.")
    rationale: str = Field(..., description="A high-level explanation for the current structure of the knowledge graph.") 


def get_effective_edges(mechanism_to_features):
    effective_edges = set()
    for features in mechanism_to_features.values():
        for edge in combinations(sorted(features), 2):
            effective_edges.add(edge)
    return effective_edges

def model_to_networkx(kg_model: KnowledgeGraphModel) -> nx.Graph:
    G = nx.Graph()
    for node in kg_model.nodes:
        G.add_node(node.id)
    for edge in kg_model.edges:
        G.add_edge(edge.source, edge.target, relationship=edge.relationship)
    return G

def get_constraints_from_graph(nodes: List[str], edges: List[tuple[str, str]], X_train):
    feature_to_idx = {name: i for i, name in enumerate(X_train.columns)}
    valid_nodes = [node for node in nodes if node in feature_to_idx]

    if not valid_nodes:
        return [], [], []

    nodes = sorted(valid_nodes)
    feature_indices = [feature_to_idx[f] for f in nodes]
    node_to_local_idx = {node: i for i, node in enumerate(nodes)}

    parent = list(range(len(nodes)))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for u, v in edges:
        if u in node_to_local_idx and v in node_to_local_idx:
            union(node_to_local_idx[u], node_to_local_idx[v])

    groups = defaultdict(list)
    for i, node in enumerate(nodes):
        groups[find(i)].append(i)

    interaction_constraints = [sorted(group) for group in groups.values() if len(group) > 1]
    return interaction_constraints, feature_indices, nodes

def graph_to_ml_model(graph, X_train):
    nodes = list(graph.nodes())
    edges = list(graph.edges())
    interaction_constraints, feature_indices, valid_nodes = get_constraints_from_graph(nodes, edges, X_train)
    valid_edges = [(u, v) for u, v in edges if u in valid_nodes and v in valid_nodes]
    return interaction_constraints, feature_indices, valid_nodes, valid_edges

def get_mechanism_to_features(kg_model, feature_names):
    intermediate_nodes = [node for node in kg_model.nodes if node.node_type == 'intermediate']
    mechanism_to_features = {}
    for intermediate in intermediate_nodes:
        connected_features = []
        for edge in kg_model.edges:
            if edge.source == intermediate.name and edge.target in feature_names:
                connected_features.append(edge.target)
            elif edge.target == intermediate.name and edge.source in feature_names:
                connected_features.append(edge.source)
        if connected_features:
            mechanism_to_features[intermediate.name] = connected_features
    return mechanism_to_features

def networkx_to_model(G: nx.Graph) -> KnowledgeGraphModel:
    """Converts a NetworkX graph to a KnowledgeGraphModel."""
    nodes = []
    for node_id, attrs in G.nodes(data=True):
        nodes.append(Node(
            name=node_id,
            node_type=attrs.get('node_type', 'input'),
            description=attrs.get('description', '')
        ))

    edges = []
    for u, v, attrs in G.edges(data=True):
        edges.append(Edge(
            source=u,
            target=v,
            relationship=attrs.get('relationship', 'interacts')
        ))
    
    rationale = G.graph.get('rationale', 'Graph loaded from file.')
    return KnowledgeGraphModel(nodes=nodes, edges=edges, rationale=rationale)

def save_model(model, dataset_name: str):
    os.makedirs('models', exist_ok=True)
    model_path = f'models/{dataset_name}_model.joblib'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def calculate_shap_contributions(kg_model, X_train, y_train, X_val, y_val):
    # Train model with current graph constraints
    graph = model_to_networkx(kg_model)
    interaction_constraints, feature_indices_kg, _, _ = graph_to_ml_model(graph, X_train)
    X_train_kg = X_train.iloc[:, feature_indices_kg]
    X_val_kg = X_val.iloc[:, feature_indices_kg]
    
    model = copy.deepcopy(ML_MODEL)
    model.set_params(interaction_constraints=interaction_constraints)
    model.fit(X_train_kg, y_train, eval_set=[(X_val_kg, y_val)])

    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train_kg)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Calculate interaction values
    shap_interaction_values = explainer.shap_interaction_values(X_train_kg)
    if isinstance(shap_interaction_values, list):
        shap_interaction_values = shap_interaction_values[1]
        
    feature_names = X_train_kg.columns.tolist()
    
    # Get feature contributions
    feature_contributions = []
    for i, f in enumerate(feature_names):
        feature_contributions.append((f, np.abs(shap_values[:, i]).mean()))
    
    # Sort by contribution strength
    feature_contributions.sort(key=lambda x: x[1], reverse=True)

    # Get interaction pairs
    interaction_pairs = []
    for i in range(shap_interaction_values.shape[1]):
        for j in range(shap_interaction_values.shape[1]):
            if i < j:
                interaction_strength = np.abs(shap_interaction_values[:, i, j]).mean()
                if interaction_strength > 0:
                    interaction_pairs.append(
                        (feature_names[i], feature_names[j], interaction_strength)
                    )
    interaction_pairs.sort(key=lambda x: x[2], reverse=True)
    return feature_contributions, interaction_pairs

def save_knowledge_graph(kg_model: KnowledgeGraphModel, filename: str):
    """Saves a KnowledgeGraphModel to a GraphML file."""
    G = model_to_networkx(kg_model)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    nx.write_graphml(G, filename)

def load_knowledge_graph(filename: str) -> KnowledgeGraphModel:
    """Loads a KnowledgeGraphModel from a GraphML file."""
    G = nx.read_graphml(filename)
    nodes = [Node(id=node_id) for node_id in G.nodes()]
    edges = [Edge(source=u, target=v, relationship="interacts") for u, v in G.edges()]
    return KnowledgeGraphModel(nodes=nodes, edges=edges)

def create_interaction_constraints(mechanism_to_features, feature_names):
    constraint_indices = []
    for mechanism, features in mechanism_to_features.items():
        if len(features) > 1:
            # Convert feature names to indices
            feature_indices = []
            for feat in features:
                if feat in feature_names:
                    feature_indices.append(feature_names.index(feat))
            if len(feature_indices) > 1:
                constraint_indices.append(feature_indices)
    return constraint_indices