from agno.agent import Agent
import os
import json
import networkx as nx
import pandas as pd
from pydantic import BaseModel
from typing import List
from agno.knowledge.arxiv import ArxivKnowledgeBase
from agno.vectordb.chroma import ChromaDb
from visualizations import visualize_knowledge_graph
from params import (TARGET_COL, KEYWORDS, LLM_MODEL, DATASET_NAME, PLOT_IMAGES, VERBOSE, USE_KNOWLEDGE_BASE)

class Mechanism(BaseModel):
    name: str
    description: str
    assigned_features: List[str]

class MechanismList(BaseModel):
    mechanisms: List[Mechanism]

def create_kg(df):
    """Creates a knowledge graph based on disease mechanisms with rich feature-mechanism connections."""
    
    if VERBOSE:
        print(f"\n=== Creating Knowledge Graph for {DATASET_NAME} ===")
        print(f"Dataset shape: {df.shape}")
        print(f"Target column: {TARGET_COL}")
    
    with open(f'dataset_info/{DATASET_NAME}_info.txt', 'r') as f:
        dataset_info = f.read()
    
    feature_names = [col for col in df.columns.tolist() if col != TARGET_COL]
    
    if VERBOSE:
        print(f"Total features: {len(feature_names)}")
        print(f"Dataset info length: {len(dataset_info)} characters")
    
    # Initialize agent outputs storage
    agent_outputs = {
        "dataset_name": DATASET_NAME,
        "target_column": TARGET_COL,
        "total_features": len(feature_names),
        "mechanisms": [],
        "final_graph_stats": {}
    }

    # --- Agent Setup ---
    if USE_KNOWLEDGE_BASE:
        vector_db = ChromaDb(
            collection=f"{DATASET_NAME}_mechanisms_collection",
        )
        knowledge_base = ArxivKnowledgeBase(
            queries=KEYWORDS,
            vector_db=vector_db,
        )
    
        if VERBOSE:
            print(f"Keywords for knowledge base: {KEYWORDS}")
    
    print(f"Creating Knowledge Graph for {DATASET_NAME}")
    print(f"Target: {TARGET_COL}")
    print(f"Total features: {len(feature_names)}")
    
    # --- Step 1: Create Disease Mechanisms and Assign All Features ---
    
    mechanism_agent = Agent(
        model=LLM_MODEL, 
        response_model=MechanismList,
        knowledge=knowledge_base if USE_KNOWLEDGE_BASE else None,
        search_knowledge=True,
        instructions="""You are a medical expert creating a comprehensive knowledge graph of disease mechanisms from clinical features. Your goal is to define 5-10 core mechanisms and exhaustively assign all relevant features to them.

For each mechanism, provide:
- name: A descriptive name for the biological pathway (e.g., "Inflammatory Cascade").
- description: A detailed explanation of the process.
- assigned_features: A comprehensive list of ALL features from the dataset related to this mechanism. A feature can be a cause, effect, biomarker, or part of the same biological pathway. Features can be assigned to multiple mechanisms."""
    )
    
    if USE_KNOWLEDGE_BASE:
        mechanism_agent.knowledge.load(recreate=False)
    
    mechanism_prompt = f"""
    **Target Condition:** {TARGET_COL}
    **Clinical Context:**
    {dataset_info}
    **Available Features:**
    {feature_names}
    
    **Task:**
    1. Define 6-10 key disease mechanisms explaining how features contribute to {TARGET_COL}.
    2. For each mechanism, assign ALL potentially related features from the list. Be exhaustive.
    
    Your goal is to create a complete map between features and mechanisms in a single step.
    """
    
    mechanisms = mechanism_agent.run(mechanism_prompt).content.mechanisms
    
    if VERBOSE:
        print(f"Generated {len(mechanisms)} mechanisms in a single step:")
        for i, mech in enumerate(mechanisms):
            print(f"  {i+1}. {mech.name}: {len(mech.assigned_features)} features")

    final_mechanisms = mechanisms
    
    # --- Step 2: Ensure All Features Are Assigned ---
    
    # Find unassigned features
    all_assigned_features = set()
    for m in final_mechanisms:
        all_assigned_features.update(m.assigned_features)
    unassigned_features = set(feature_names) - all_assigned_features
    
    if VERBOSE:
        print(f"Features assigned: {len(all_assigned_features)}")
        print(f"Features unassigned: {len(unassigned_features)}")
        if unassigned_features:
            print(f"Unassigned features: {list(unassigned_features)[:10]}{'...' if len(unassigned_features) > 10 else ''}")
    
    if unassigned_features:
        print(f"Found {len(unassigned_features)} unassigned features. Assigning them to most relevant mechanisms...")
        
        # Create an agent to assign remaining features
        assignment_agent = Agent(
            model=LLM_MODEL,
            knowledge=knowledge_base if USE_KNOWLEDGE_BASE else None,
            search_knowledge=True,
            response_model=MechanismList,
            instructions="""You are assigning unassigned features to the most relevant disease mechanisms.
            Each feature should be assigned to at least one mechanism based on potential relevance."""
        )
        
        if USE_KNOWLEDGE_BASE:
            assignment_agent.knowledge.load(recreate=False)
        
        # Create a prompt with all mechanisms and unassigned features
        assignment_prompt = f"""
        **Target Condition:** {TARGET_COL}
        
        **Existing Mechanisms:**
        {chr(10).join([f"- {m.name}: {m.description}" for m in final_mechanisms])}
        
        **Unassigned Features:**
        {list(unassigned_features)}
        
        **Task:** For each unassigned feature, assign it to one or more of the existing mechanisms based on potential relevance.
        Return the mechanisms with the unassigned features added to their assigned_features lists.
        """
        
        updated_mechanisms = assignment_agent.run(assignment_prompt).content.mechanisms
        
        # Merge the assignments
        mechanism_map = {m.name: m for m in final_mechanisms}
        for updated_mech in updated_mechanisms:
            if updated_mech.name in mechanism_map:
                # Add new features to existing mechanism
                existing_features = set(mechanism_map[updated_mech.name].assigned_features)
                new_features = set(updated_mech.assigned_features) - existing_features
                mechanism_map[updated_mech.name].assigned_features.extend(list(new_features))
                all_assigned_features.update(new_features)
                
                if VERBOSE and new_features:
                    print(f"  Added {len(new_features)} features to {updated_mech.name}")
        
        # Verify all features are now assigned
        still_unassigned = set(feature_names) - all_assigned_features
        if still_unassigned:
            if VERBOSE:
                print(f"Warning: {len(still_unassigned)} features still unassigned: {list(still_unassigned)}")
            print(f"Warning: {len(still_unassigned)} features still unassigned. Assigning to first mechanism.")
            final_mechanisms[0].assigned_features.extend(list(still_unassigned))
    
    print(f"All {len(feature_names)} features are now assigned to mechanisms.")
    
    # Validation step: remove features not in the dataset
    valid_feature_set = set(feature_names)
    invalid_features_found = set()
    
    for mechanism in final_mechanisms:
        valid_features = []
        for feature in mechanism.assigned_features:
            if feature in valid_feature_set:
                valid_features.append(feature)
            else:
                invalid_features_found.add(feature)
        mechanism.assigned_features = valid_features
    
    if VERBOSE and invalid_features_found:
        print(f"Removed {len(invalid_features_found)} invalid features: {list(invalid_features_found)[:5]}{'...' if len(invalid_features_found) > 5 else ''}")
    
    # --- Step 4: Create NetworkX Graph ---
    
    G = nx.DiGraph()
    
    # Add target node
    G.add_node(TARGET_COL, node_type='target')
    
    # Add mechanism nodes and edges
    mechanism_nodes = []
    for mechanism in final_mechanisms:
        mechanism_name = mechanism.name
        G.add_node(mechanism_name, node_type='mechanism', 
                  description=mechanism.description,
                  feature_count=len(mechanism.assigned_features))
        G.add_edge(TARGET_COL, mechanism_name)
        mechanism_nodes.append(mechanism_name)
        
        # Add feature nodes and edges
        for feature in mechanism.assigned_features:
            if feature not in G.nodes():
                G.add_node(feature, node_type='feature')
            G.add_edge(mechanism_name, feature)
    
    if VERBOSE:
        print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        print(f"Mechanism nodes: {len(mechanism_nodes)}")
        print(f"Feature nodes: {len([n for n in G.nodes() if G.nodes[n].get('node_type') == 'feature'])}")
    
    # Allow features to be in multiple groups - create feature-to-groups mapping
    feature_to_groups = {}
    for i, mechanism in enumerate(final_mechanisms):
        if len(mechanism.assigned_features) > 1:
            for feature in mechanism.assigned_features:
                if feature not in feature_to_groups:
                    feature_to_groups[feature] = []
                feature_to_groups[feature].append(i)
    
    # Create mechanism groups for model stacking
    mechanism_groups = []
    group_labels = []
    for mechanism in final_mechanisms:
        if len(mechanism.assigned_features) > 1:
            mechanism_groups.append(mechanism.assigned_features)
            group_labels.append(mechanism.name)
    
    if VERBOSE:
        print(f"Created {len(mechanism_groups)} mechanism groups for interaction constraints")
        multi_group_features = sum(1 for groups in feature_to_groups.values() if len(groups) > 1)
        print(f"Features in multiple groups: {multi_group_features}")
    
    # --- Step 6: Save Everything ---
    
    os.makedirs('kg', exist_ok=True)
    
    # Save as GraphML
    nx.write_graphml(G, f'kg/{DATASET_NAME}_initial_agent_kg.graphml')
    
    # Save constraints data in the format expected by other scripts
    constraints_data = {
        "interaction_constraints": mechanism_groups,
        "constraint_labels": group_labels
    }
    
    with open(f'kg/{DATASET_NAME}_interaction_constraints.json', 'w') as f:
        json.dump(constraints_data, f, indent=2)

    # Save detailed agent outputs
    agent_outputs["mechanisms"] = [
        {
            "name": m.name,
            "description": m.description,
            "assigned_features": m.assigned_features,
            "feature_count": len(m.assigned_features),
        } for m in final_mechanisms
    ]
    
    agent_outputs["final_graph_stats"] = {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "target_nodes": 1,
        "mechanism_nodes": len(mechanism_nodes),
        "feature_nodes": len([n for n in G.nodes() if G.nodes[n].get('node_type') == 'feature']),
        "avg_edges_per_feature": G.number_of_edges() / len(feature_names)
    }
    
    with open(f'kg/{DATASET_NAME}_agent_outputs.json', 'w') as f:
        json.dump(agent_outputs, f, indent=2)

    # Visualize and save the initial KG
    if PLOT_IMAGES:
        feature_to_labels_map = {}
        for i, group in enumerate(mechanism_groups):
            label_for_group = group_labels[i]
            for feature in group:
                if feature not in feature_to_labels_map:
                    feature_to_labels_map[feature] = []
                feature_to_labels_map[feature].append(label_for_group)

        visualize_knowledge_graph(
            constraints=mechanism_groups,
            labels=group_labels,
            filename=f'{DATASET_NAME}_initial_kg.png',
            feature_to_labels_map=feature_to_labels_map
        )
  
    return mechanism_groups, group_labels

if __name__ == "__main__":
    df = pd.read_csv(f'datasets/{DATASET_NAME}.csv')
    G, mechanisms, mechanism_groups, group_labels = create_kg(df)