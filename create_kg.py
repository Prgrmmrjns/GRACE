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
from params import (TARGET_COL, EMBEDDING_MODEL, KEYWORDS, LLM_MODEL, DATASET_NAME, DATASET_PATH, PLOT_IMAGES, VERBOSE)

class Mechanism(BaseModel):
    name: str
    description: str
    assigned_features: List[str]
    label: str

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
    vector_db = ChromaDb(
        collection=f"{DATASET_NAME}_mechanisms_collection",
        embedder=EMBEDDING_MODEL,
    )
    knowledge_base = ArxivKnowledgeBase(
        queries=KEYWORDS,
        vector_db=vector_db,
    )
    
    if VERBOSE:
        print(f"Keywords for knowledge base: {KEYWORDS}")
    
    print(f"\n=== Creating Knowledge Graph for {DATASET_NAME} ===")
    print(f"Target: {TARGET_COL}")
    print(f"Total features: {len(feature_names)}")
    
    # --- Step 1: Create Initial Disease Mechanisms ---
    print("\n--- Step 1: Creating Disease Mechanisms ---")
    
    mechanism_agent = Agent(
        model=LLM_MODEL, 
        response_model=MechanismList,
        knowledge=knowledge_base,
        search_knowledge=True,
        instructions="""You are a medical expert creating disease mechanisms. Your task is to identify 6-10 key biological/pathological mechanisms that contribute to the target condition.
        Create comprehensive mechanisms that cover different aspects:
        - Molecular/cellular processes
        - Physiological systems  
        - Clinical manifestations
        - Risk factors
        - Diagnostic markers
        
        For each mechanism, provide:
        - name: A descriptive name for the mechanism
        - description: Detailed explanation of the mechanism
        - assigned_features: Relevant features based on domain knowledge (be generous)
        - label: A short 1-3 word overarching term that captures the essence of this mechanism (e.g., "Cardiovascular", "Metabolic", "Inflammatory", "Respiratory")"""
    )
    
    if VERBOSE:
        print("Loading knowledge base...")
    mechanism_agent.knowledge.load(recreate=False)
    
    mechanism_prompt = f"""
    **Target Condition:** {TARGET_COL}
    **Clinical Context:**
    {dataset_info}
    **Available Features:**
    {feature_names}
    
    Create 6-10 disease mechanisms that comprehensively explain how different features contribute to {TARGET_COL}.
    Assign features generously to mechanisms based on potential biological relevance.
    """
    
    if VERBOSE:
        print("Generating initial mechanisms...")
    
    mechanisms = mechanism_agent.run(mechanism_prompt).content.mechanisms
    
    if VERBOSE:
        print(f"Generated {len(mechanisms)} initial mechanisms:")
        for i, mech in enumerate(mechanisms):
            print(f"  {i+1}. {mech.name} ({mech.label}): {len(mech.assigned_features)} features")
            print(f"     Description: {mech.description[:100]}...")
    
    # --- Step 2: Enrich Connections by Asking Each Mechanism to Claim More Features ---
    print("\n--- Step 2: Enriching Feature-Mechanism Connections ---")
    
    enrichment_agent = Agent(
        model=LLM_MODEL, 
        knowledge=knowledge_base,
        search_knowledge=True,
        response_model=MechanismList,
        instructions="""You are reviewing each disease mechanism and identifying features that could potentially be related to it.
        
        Suggest up to 30 features that could be related to the mechanism."""
    )
    enrichment_agent.knowledge.load(recreate=False)
    
    # For each mechanism, ask agent to identify ALL potentially relevant features
    enriched_mechanisms = []
    all_assigned_features = set()
    
    for i, mechanism in enumerate(mechanisms):
        if VERBOSE:
            print(f"Enriching mechanism {i+1}/{len(mechanisms)}: {mechanism.name}")
        
        enrichment_prompt = f"""
        **Focus Mechanism:** {mechanism.name}
        **Description:** {mechanism.description}
        
        **All Available Features:**
        {feature_names}
        
        **Task:** Identify features that could be related to this mechanism. 
        
        Consider features that could be:
        - Direct effects of this mechanism
        - Causes or triggers of this mechanism  
        - Biomarkers or indicators of this mechanism
        - Associated with this mechanism through biological pathways
        
        Return the mechanism with an expanded list of assigned features.
        """
        
        enriched_response = enrichment_agent.run(enrichment_prompt)
        enriched_mechanism = enriched_response.content.mechanisms[0]  # Should return single mechanism
        
        # Keep original mechanism info but update features
        enriched_mechanism.name = mechanism.name
        enriched_mechanism.description = mechanism.description
        enriched_mechanism.label = mechanism.label
        
        enriched_mechanisms.append(enriched_mechanism)
        all_assigned_features.update(enriched_mechanism.assigned_features)
        
        if VERBOSE:
            print(f"  Assigned {len(enriched_mechanism.assigned_features)} features (was {len(mechanism.assigned_features)})")
        else:
            print(f"Assigned {len(enriched_mechanism.assigned_features)} features to {mechanism.name}")
    
    final_mechanisms = enriched_mechanisms
    
    # --- Step 3: Ensure All Features Are Assigned ---
    print("\n--- Step 3: Ensuring Complete Feature Coverage ---")
    
    # Find unassigned features
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
            knowledge=knowledge_base,
            search_knowledge=True,
            response_model=MechanismList,
            instructions="""You are assigning unassigned features to the most relevant disease mechanisms.
            Each feature should be assigned to at least one mechanism based on potential relevance."""
        )
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
        
        if VERBOSE:
            print("Running assignment agent for unassigned features...")
        
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
    if VERBOSE:
        print("\n--- Step 4: Creating NetworkX Graph ---")
    
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
        print(f"Target nodes: 1")
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
            group_labels.append(mechanism.label)
    
    if VERBOSE:
        print(f"Created {len(mechanism_groups)} mechanism groups for interaction constraints")
        multi_group_features = sum(1 for groups in feature_to_groups.values() if len(groups) > 1)
        print(f"Features in multiple groups: {multi_group_features}")
    
    # --- Step 6: Save Everything ---
    if VERBOSE:
        print("\n--- Step 6: Saving Results ---")
    
    os.makedirs('kg', exist_ok=True)
    
    # Save as GraphML
    nx.write_graphml(G, f'kg/{DATASET_NAME}_initial_agent_kg.graphml')
    if VERBOSE:
        print(f"Saved GraphML to kg/{DATASET_NAME}_initial_agent_kg.graphml")
    
    # Save mechanism data for model stacking
    mechanism_data = {
        "dataset_name": DATASET_NAME,
        "target_column": TARGET_COL,
        "mechanism_groups": mechanism_groups,
        "group_labels": group_labels,
        "feature_to_groups": feature_to_groups,
        "mechanisms": [
            {
                "name": m.name,
                "description": m.description,
                "assigned_features": m.assigned_features,
                "label": m.label
            } for m in final_mechanisms
        ],
        "statistics": {
            "num_mechanisms": len(final_mechanisms),
            "num_groups": len(mechanism_groups),
            "total_features": len(feature_names),
            "total_edges": G.number_of_edges(),
            "avg_features_per_mechanism": sum(len(m.assigned_features) for m in final_mechanisms) / len(final_mechanisms),
            "features_covered": len(all_assigned_features)
        },
        "interaction_constraints": mechanism_groups,
    }
    
    with open(f'kg/{DATASET_NAME}_mechanism_data.json', 'w') as f:
        json.dump(mechanism_data, f, indent=2)
    
    if VERBOSE:
        print(f"Saved mechanism data to kg/{DATASET_NAME}_mechanism_data.json")
    
    # Save detailed agent outputs
    agent_outputs["mechanisms"] = [
        {
            "name": m.name,
            "description": m.description,
            "assigned_features": m.assigned_features,
            "feature_count": len(m.assigned_features),
            "label": m.label
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
            label_for_group = labels[i]
            for feature in group:
                if feature not in feature_to_labels_map:
                    feature_to_labels_map[feature] = []
                feature_to_labels_map[feature].append(label_for_group)

        print("Generating initial knowledge graph visualization...")
        visualize_knowledge_graph(
            constraints=mechanism_groups,
            labels=labels,
            title=f'Initial {DATASET_NAME.upper()} Knowledge Graph',
            filename=f'{DATASET_NAME}_initial_kg.png',
            feature_to_labels_map=feature_to_labels_map
        )

    return G, mechanisms, mechanism_groups, labels

if __name__ == "__main__":
    df = pd.read_csv(f'datasets/{DATASET_NAME}.csv')
    G, mechanisms, mechanism_groups, labels = create_kg(df)