import warnings
import copy
import os
import numpy as np
import shap
from sklearn.metrics import roc_auc_score, accuracy_score
from itertools import combinations
from params import DATASET_NAME
from utils import model_to_networkx
import os
import sys
import pandas as pd
from agno.agent import Agent
from agno.tools import tool
from typing import List
from models import Edge, Node, KnowledgeGraphModel, KnowledgeGraphRefinement
from params import (
    LLM_MODEL, 
    TARGET_COL, 
    LLM_CACHE_DIR, 
    DATASET_NAME, 
    LOAD_LLM_CACHE,
    ML_MODEL,
    METRIC,
    PREDICT_FN,
)

# Global variables for agent access
g_kg_model: KnowledgeGraphModel = None
g_X_train_orig: pd.DataFrame = None
g_X_val_orig: pd.DataFrame = None
g_X_test_orig: pd.DataFrame = None
g_y_train: pd.Series = None
g_y_val: pd.Series = None
g_final_model = None

# Define agent tools
@tool
def show_top_features_and_interactions() -> str:
    """
    Shows the top 15 most important features and feature interactions based on SHAP values from the current best model.
    Use this tool to get a general overview of what is driving the model's predictions.
    This function takes no arguments.
    """
    global g_final_model, g_kg_model, g_X_train_orig

    # Get current features from the graph
    graph = model_to_networkx(g_kg_model)
    _, feature_indices, nodes, _ = graph_to_ml_model(graph, g_X_train_orig)

    X_train_kg = g_X_train_orig.iloc[:, feature_indices]
    
    feature_names = list(X_train_kg.columns)
    explainer = shap.TreeExplainer(model=g_final_model)
    shap_interaction_values = explainer.shap_interaction_values(X_train_kg)
    shap_values = explainer.shap_values(X_train_kg)

    if isinstance(shap_interaction_values, list):
        best_class_idx = np.argmax([np.mean(np.abs(sv)) for sv in shap_values])
        interaction_values = np.abs(shap_interaction_values[best_class_idx]).mean(axis=0)
        feature_contributions = np.abs(shap_values[best_class_idx]).mean(axis=0)
    else:
        interaction_values = np.abs(shap_interaction_values).mean(axis=0)
        feature_contributions = np.abs(shap_values[1] if isinstance(shap_values, list) else shap_values).mean(axis=0)

    interaction_pairs = [(feature_names[i], feature_names[j], np.abs(interaction_values[i, j]).mean()) 
                        for i, j in combinations(range(len(feature_names)), 2)]
    interaction_pairs.sort(key=lambda x: x[2], reverse=True)
    top_interactions_str = "\\n".join([f"- {f1} <-> {f2} (strength: {strength:.4f})" for f1, f2, strength in interaction_pairs[:15]])
    
    feature_importance_data = [{'feature': name, 'importance': float(contrib) if np.isscalar(contrib) else np.mean(contrib)} 
                              for i, (name, contrib) in enumerate(zip(feature_names, feature_contributions))]
    feature_importance_data.sort(key=lambda x: x['importance'], reverse=True)
    top_feature_contributions_str = "\\n".join([f"- {item['feature']}: {item['importance']:.4f}" for item in feature_importance_data[:15]])
    return f"Top 15 Feature Contributions:\\n{top_feature_contributions_str}\\n\\nTop 15 Feature Interactions:\\n{top_interactions_str}"

@tool
def add_node(feature_name: str) -> str:
    """
    Test adding a single feature node to the knowledge graph and return the resulting validation score.
    Use this to see if adding a specific feature improves model performance.
    
    Args:
        feature_name: The name of the feature to add to the graph.
        
    Returns:
        A string with the new validation score if the feature was added successfully.
    """
    global g_kg_model, g_X_train_orig, g_X_val_orig, g_y_train, g_y_val, g_final_model
    print(f"Adding node: {feature_name}")
    
    # Create temporary copy and add the node
    temp_kg_model = copy.deepcopy(g_kg_model)
    new_node = Node(id=feature_name, node_type="feature", attributes={"description": f"Added feature: {feature_name}"})
    temp_kg_model.add_node(new_node)
    
    # Convert to ML model and evaluate
    temp_graph = model_to_networkx(temp_kg_model)
    new_interaction_constraints, new_feature_indices, new_nodes, new_edges = graph_to_ml_model(temp_graph, g_X_train_orig)
    
    X_train_kg_temp = g_X_train_orig.iloc[:, new_feature_indices]
    X_val_kg_temp = g_X_val_orig.iloc[:, new_feature_indices]
    
    temp_model = copy.deepcopy(ML_MODEL)
    temp_model.set_params(interaction_constraints=new_interaction_constraints)
    new_val_score = evaluate_model(temp_model, X_train_kg_temp, X_val_kg_temp, g_y_train, g_y_val)
    return f"Adding '{feature_name}' would result in validation score: {new_val_score:.4f}"

# --- Agent-based Knowledge Graph Functions ---
def create_initial_knowledge_graph(top_feature_contributions: List[str], top_interactions: str) -> KnowledgeGraphModel:
    cache_path = os.path.join(LLM_CACHE_DIR, f"{DATASET_NAME}_comprehensive_kg.json")
    if LOAD_LLM_CACHE and os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        print(f"Loading cached comprehensive KG from {cache_path}")
        with open(cache_path, 'r') as f:
            kg_model = KnowledgeGraphModel.model_validate_json(f.read())
        return kg_model

    print("Generating Comprehensive Knowledge Graph from LLM...")
    agent = Agent(model=LLM_MODEL, response_model=KnowledgeGraphModel)
    
    prompt = f"""
    As an expert biomedical researcher, create a comprehensive knowledge graph explaining how features relate to '{TARGET_COL}'.
    Use both domain knowledge and the provided SHAP analysis results.

    **Instructions:**
    - Connect two features if they have a high SHAP interaction value or if they are known to be related in the literature.
    - Provide a rationale for each edge. This rationale should be based on the SHAP values and/or the literature.
    - Make sure that each feature is connected to at least one other feature.
    - Only create edges between two features. Dont create edges between features and the target.
    - It is up to you decide which features to connect and which should not be connected. Make it dependent on the SHAP values and the literature.

    **Top Important Features (SHAP Analysis):**
    {top_feature_contributions}

    **Key Feature Interactions (SHAP Analysis):**
    {top_interactions}
    """
    original_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    try:
        response = agent.run(prompt)
    finally:
        sys.stderr.close()
        sys.stderr = original_stderr
    kg_model = response.content
    with open(cache_path, "w") as f:
        f.write(kg_model.model_dump_json(indent=4))
    print(f"Cached comprehensive KG to {cache_path}")
    return kg_model

def refine_knowledge_graph(graph: KnowledgeGraphModel, features: List[str], best_val_score: float) -> KnowledgeGraphModel:
    agent = Agent(
        model=LLM_MODEL, 
        response_model=KnowledgeGraphRefinement, 
        tools=[add_node],
    )
    prompt = f"""
    As an expert biomedical researcher, refine the knowledge graph to improve the model's performance.
    Your goal is to propose changes that will increase the validation score. The current best validation score is {best_val_score:.4f}.

    You have tools to help you analyze and test changes:
    - `show_top_features_and_interactions()`: See the top 15 most important features and feature interactions from the current model. Takes no arguments.
    - `add_node(feature_name: str)`: Test adding a specific feature to the graph and see the resulting validation score.

    Use these tools to understand the model and test potential improvements. Then propose changes to the graph.
    The final output of your response must be a `KnowledgeGraphRefinement` object.
    You must provide a rationale for all the changes you propose.

    The current graph is:
    {graph.model_dump_json(indent=2)}

    Available features you can add to the graph:
    {sorted(list(set(features) - set(n.id for n in graph.nodes)))}
    """
    original_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    try:
        response = agent.run(prompt)
    finally:
        sys.stderr.close()
        sys.stderr = original_stderr
    return response.content

def calculate_shap_interactions(model, X_train):
    feature_names = list(X_train.columns)
    explainer = shap.TreeExplainer(model=model)
    shap_interaction_values = explainer.shap_interaction_values(X_train)
    shap_values = explainer.shap_values(X_train)

    if isinstance(shap_interaction_values, list):
        best_class_idx = np.argmax([np.mean(np.abs(sv)) for sv in shap_values])
        interaction_values = np.abs(shap_interaction_values[best_class_idx]).mean(axis=0)
        feature_contributions = np.abs(shap_values[best_class_idx]).mean(axis=0)
    else:
        interaction_values = np.abs(shap_interaction_values).mean(axis=0)
        feature_contributions = np.abs(shap_values[1] if isinstance(shap_values, list) else shap_values).mean(axis=0)

    interaction_pairs = [(feature_names[i], feature_names[j], np.abs(interaction_values[i, j]).mean()) 
                        for i, j in combinations(range(len(feature_names)), 2)]
    interaction_pairs.sort(key=lambda x: x[2], reverse=True)
    top_interactions_str = "\n".join([f"- {f1} <-> {f2} (strength: {strength:.4f})" for f1, f2, strength in interaction_pairs[:50]])
    feature_importance_data = [{'feature': name, 'importance': float(contrib) if np.isscalar(contrib) else np.mean(contrib), 'rank': i + 1} 
                              for i, (name, contrib) in enumerate(zip(feature_names, feature_contributions))]
    feature_importance_data.sort(key=lambda x: x['importance'], reverse=True)
    top_feature_contributions_str = "\n".join([f"- {item['feature']}: {item['importance']:.4f}" for item in feature_importance_data[:30]])
    return top_interactions_str, top_feature_contributions_str

def graph_to_ml_model(graph, X_train):
    feature_to_idx = {name: i for i, name in enumerate(X_train.columns)}
    nodes = {node for node in graph.nodes() if node in feature_to_idx}
    feature_indices = sorted([feature_to_idx[f] for f in nodes])
    edges = list(set(graph.edges()))
    old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(feature_indices)}
    filtered_interactions = [(old_to_new_idx[u], old_to_new_idx[v]) for u, v in edges 
                           if u in old_to_new_idx and v in old_to_new_idx]
    interaction_constraints_dict = {}
    for u, v in filtered_interactions:
        if u not in interaction_constraints_dict:
            interaction_constraints_dict[u] = []
        interaction_constraints_dict[u].append(v)
    
    interaction_constraints = [[key] + value for key, value in interaction_constraints_dict.items()]
    return interaction_constraints, feature_indices, nodes, edges

def evaluate_model(model, X_train, X_val, y_train, y_val):
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    val_pred = PREDICT_FN(model, X_val)
    val_score = accuracy_score(y_val, val_pred) if METRIC == 'accuracy' else roc_auc_score(y_val, val_pred)
    return val_score

def run_grace(X_train, X_val, X_test, y_train, y_val, y_test):
    global g_kg_model, g_X_train_orig, g_X_val_orig, g_X_test_orig, g_y_train, g_y_val, g_final_model
    g_X_train_orig = X_train
    g_X_val_orig = X_val
    g_X_test_orig = X_test
    g_y_train = y_train
    g_y_val = y_val

    # Get shap values
    model = ML_MODEL.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    top_interactions, top_feature_contributions = calculate_shap_interactions(model, X_train)

    # Creating initial knowledge graph
    kg_model = create_initial_knowledge_graph(top_feature_contributions, top_interactions)
    g_kg_model = kg_model
    graph = model_to_networkx(kg_model)
    interaction_constraints, feature_indices, nodes, edges = graph_to_ml_model(graph, X_train)
    X_train_kg, X_val_kg, X_test_kg = X_train.iloc[:, feature_indices], X_val.iloc[:, feature_indices], X_test.iloc[:, feature_indices]
    final_model = copy.deepcopy(ML_MODEL)
    final_model.set_params(interaction_constraints=interaction_constraints)
    best_val_score = evaluate_model(final_model, X_train_kg, X_val_kg, y_train, y_val)
    g_final_model = final_model
    print(f"Val Score after creating initial knowledge graph: {best_val_score:.4f}")

    # Refining knowledge graph
    max_iterations = 10
    for i in range(max_iterations):
        print(f"--- Refinement Iteration {i+1}/{max_iterations} ---")
        graph_refinement = refine_knowledge_graph(kg_model, X_train.columns.tolist(), best_val_score)
        if not graph_refinement or not any([graph_refinement.nodes_to_add, graph_refinement.edges_to_add, graph_refinement.nodes_to_remove, graph_refinement.edges_to_remove]):
            print("Agent suggests no further improvements. Stopping refinement.")
            break

        print(f"Agent proposed changes: {graph_refinement.rationale}")
        new_kg_model = copy.deepcopy(kg_model)

        for node in graph_refinement.nodes_to_add:
            new_kg_model.add_node(node)
        for node_id in graph_refinement.nodes_to_remove:
            new_kg_model.remove_node(node_id)
        for edge in graph_refinement.edges_to_add:
            new_kg_model.add_edge(edge)
        for edge in graph_refinement.edges_to_remove:
            new_kg_model.remove_edge(edge)

        temp_graph = model_to_networkx(new_kg_model)
        new_interaction_constraints, new_feature_indices, new_nodes, new_edges = graph_to_ml_model(temp_graph, X_train)
        
        if not new_feature_indices:
            print("Refinement led to a model with no features. Stopping.")
            break
            
        X_train_kg_temp, X_val_kg_temp = X_train.iloc[:, new_feature_indices], X_val.iloc[:, new_feature_indices]
        
        temp_model = copy.deepcopy(ML_MODEL)
        temp_model.set_params(interaction_constraints=new_interaction_constraints)
        new_val_score = evaluate_model(temp_model, X_train_kg_temp, X_val_kg_temp, y_train, y_val)
        print(f"New validation score: {new_val_score:.4f}")

        if new_val_score > best_val_score:
            print("Validation score improved. Accepting changes.")
            best_val_score = new_val_score
            kg_model = new_kg_model
            g_kg_model = kg_model
            graph = temp_graph
            interaction_constraints = new_interaction_constraints
            feature_indices = new_feature_indices
            nodes = new_nodes
            edges = new_edges
            X_train_kg, X_val_kg, X_test_kg = X_train_kg_temp, X_val_kg_temp, g_X_test_orig.iloc[:, new_feature_indices]
            final_model = temp_model
            g_final_model = final_model
        else:
            print("Validation score did not improve. Rejecting changes and stopping refinement.")
            break


    # Evaluating model on test set
    test_pred = PREDICT_FN(final_model, X_test_kg)
    test_score = accuracy_score(y_test, test_pred) if METRIC == 'accuracy' else roc_auc_score(y_test, test_pred)
    results = {'val_score': best_val_score, 'test_score': test_score, 'n_features': len(nodes), 
            'n_edges': len(edges), 'model': final_model, 'features': [X_train.columns[i] for i in feature_indices]}
    print(f"GRACE Complete: Val {best_val_score:.4f}, Test {test_score:.4f}, Features {len(nodes)}/{len(X_train.columns)}, Edges {len(edges)}")
    return results, graph 