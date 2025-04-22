import os
import csv
import networkx as nx
from params import DATASET_PATH, DATASET_NAME, TARGET_COL, MODEL, LLM_PROVIDER
from typing import List, Dict
from visualizations import visualize_graph_structure
from graph_utils import NodeType, KnowledgeGraph, IntermediateNodes, SelectedFeatures, MissingFeatures
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.models.ollama import Ollama

def initial_prompt(feature_names: List[str], dataset_description: str, target_name: str) -> str:
    """Build the prompt for the LLM to list intermediate node names as strings"""
    return f"""Based on the following dataset description and features, identify the most relevant mechanisms, organ systems, factors or other entities that can best explain the target variable ({target_name}).
Dataset Description:
{dataset_description}
Available Features:
{feature_names}
Instructions:
Propose around 6-10 entities that best capture the key mechanisms connecting the features to {target_name}.
Examples may be: Cardiovascular System, Endocrine System, Inflammation, Laboratory markers, Cognitive function markers.  
Return a list of string with the names of the entities.
"""

def add_features_to_intermediate_node_prompt(entity_name: str, candidate_features: List[str], target_name: str) -> str:
    """Build a general prompt to select relevant features for a given entity from a list of candidates."""
    return f"""Given the following entity and a list of candidate features, select only those features that are relevant to the entity.
Entity: {entity_name}
Candidate features: {candidate_features}
Instructions:
 - From the candidate features, select only those relevant to the entity.
 - Use the exact feature names provided.
 - Select up to 20 features.
Return a list of the selected features.
"""

def add_missing_features_prompt(entity_names: List[str], remaining_features: List[str], target_name: str) -> str:
    """Build a general prompt to assign remaining features to a list of entities."""
    return f"""Given the following entities and a list of remaining features, assign each feature to one or more relevant entities. Use the exact feature names provided. 
The entities are mechanisms, organ systems, factors or other entities that best connect the input features to the target variable ({target_name}).
Entities: {entity_names}
Remaining features: {remaining_features}
Return a dictionary where the keys are entity names and the values are lists of feature names assigned to each entity. Make sure to assign all features to at least one entity.
"""

def build_kg(DATASET_PATH: str, DATASET_NAME: str, TARGET_COL: str, MODEL: str, valid_features: list = None) -> KnowledgeGraph:
    # Initialize the appropriate model based on the provider
    if LLM_PROVIDER == "openai":
        model = OpenAIChat(id=MODEL)
    elif LLM_PROVIDER == "ollama":
        model = Ollama(id=MODEL)
    
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        feature_names = [col for col in header if col.lower() != TARGET_COL.lower()]

    with open(f"dataset_info/{DATASET_NAME.lower()}_info.txt", 'r') as f:
        dataset_description = f.read().strip()

    # Step 1: Get intermediate nodes only
    prompt = initial_prompt(feature_names, dataset_description, TARGET_COL)
    
    initial_agent = Agent(model=model, response_model=IntermediateNodes, use_json_mode=True)
    
    # Run agent and get the response
    response: RunResponse = initial_agent.run(prompt)
    
    # The content field contains the structured output
    intermediate_nodes = response.content.nodes
    print(f"Intermediate nodes: {intermediate_nodes}")

    # Build empty DiGraph and add target node
    G = nx.DiGraph()
    G.add_node(TARGET_COL, entity_type=NodeType.TARGET.value)

    # Add intermediate nodes to the graph
    for node in intermediate_nodes:
        G.add_node(node, entity_type=NodeType.INTERMEDIATE.value)
        G.add_edge(node, TARGET_COL)

    # Step 2: Assign input nodes to intermediate nodes (allowing multiple assignments)
    for inter_node in intermediate_nodes:
        prompt = add_features_to_intermediate_node_prompt(inter_node, feature_names, TARGET_COL)
        add_features_agent = Agent(model=model, response_model=SelectedFeatures, use_json_mode=True)
        
        # Run agent and get the response
        response: RunResponse = add_features_agent.run(prompt)
        
        # Get the features from the content and filter to valid features
        selected_features = [f for f in response.content.features if valid_features is None or f in valid_features]
        print(f"Intermediate node: {inter_node}. Selected features: {selected_features}")
        # Add each selected feature as an input node connected to the intermediate node
        for feature in selected_features:
            if feature not in G.nodes:
                G.add_node(feature, entity_type=NodeType.INPUT.value)
            # Connect feature to intermediate node
            G.add_edge(feature, inter_node)

    # Final refinement: assign any remaining missing features
    input_nodes_set = {n for n, d in G.nodes(data=True) if d.get('entity_type') == NodeType.INPUT.value}
    remaining_features = [f for f in feature_names if f not in input_nodes_set]
    if remaining_features:
        prompt = add_missing_features_prompt(intermediate_nodes, remaining_features, TARGET_COL)
        add_missing_features_agent = Agent(model=model, response_model=MissingFeatures, use_json_mode=True)
        
        # Run agent and get the response
        response: RunResponse = add_missing_features_agent.run(prompt)
        
        # Get the assignments from the content and filter to valid features
        missing_features_dict = response.content.assignments
        print(f"Missing features: {missing_features_dict}")
        for inter_node, features in missing_features_dict.items():
            filtered_features = [f for f in features if valid_features is None or f in valid_features]
            for feature in filtered_features:
                G.add_node(feature, entity_type=NodeType.INPUT.value)
                G.add_edge(feature, inter_node)

    output_path = f"kg/{DATASET_NAME}.graphml"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nx.write_graphml(G, output_path)

    print("\nNode Types:")
    node_type_counts = {}
    for n, d in G.nodes(data=True):
        etype = d.get("entity_type", "unknown")
        node_type_counts[etype] = node_type_counts.get(etype, 0) + 1
    for k, v in node_type_counts.items():
        print(f"{k}: {v}")
    print("\nIntermediate Nodes:")
    for n, d in G.nodes(data=True):
        if d.get("entity_type") == NodeType.INTERMEDIATE.value:
            in_degree = G.in_degree(n)
            out_degree = G.out_degree(n)
            print(f"{n} Input connections: {in_degree}. Output connections: {out_degree}")

    input_nodes = set(n for n, d in G.nodes(data=True) if d.get("entity_type") == NodeType.INPUT.value)
    missing_features = set(feature_names) - input_nodes
    print(f"Missing input features: {missing_features}")
    visualize_graph_structure(G, DATASET_NAME)
    return G

if __name__ == "__main__":
    build_kg(DATASET_PATH, DATASET_NAME, TARGET_COL, MODEL)