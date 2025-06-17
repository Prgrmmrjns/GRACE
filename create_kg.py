from agno.agent import Agent
import os
import networkx as nx
from utils import Node, KnowledgeGraphModel
from agno.knowledge.arxiv import ArxivKnowledgeBase
from agno.vectordb.chroma import ChromaDb
from params import (TARGET_COL, EMBEDDING_MODEL, KEYWORDS, LLM_MODEL, DATASET_NAME)

def create_kg(df):
    """Creates a comprehensive, knowledge-driven knowledge graph in a multi-step process."""
    with open(f'dataset_info/{DATASET_NAME}_info.txt', 'r') as f:
        dataset_info = f.read()
    
    feature_names = [col for col in df.columns.tolist() if col != TARGET_COL]
    feature_list = "\n".join([f"- {feat}" for feat in feature_names])

    # --- Agent Setup ---
    vector_db = ChromaDb(
        collection_name=f"{DATASET_NAME}_kg_collection",
        embedding_model=EMBEDDING_MODEL,
        )
    knowledge_base = ArxivKnowledgeBase(
        queries=KEYWORDS,
        vector_db=vector_db,
        )
    agent = Agent(
        model=LLM_MODEL, 
        knowledge=knowledge_base,
        search_knowledge=True,
        response_model=KnowledgeGraphModel,
        instructions="""You are a medical expert building a knowledge graph for clinical predictions.
        Your goal is to create a comprehensive and structured graph of causative mechanisms and their relationships to clinical features.
        Follow the instructions precisely at each step.
        """
    )
    agent.knowledge.load(recreate=False)
    
    # --- Step 1: Derive Causative Mechanisms (Intermediate Nodes) ---
    print("--- Step 1: Deriving Causative Mechanisms ---")
    mechanism_prompt = f"""
        **Clinical Context:** The model will predict '{TARGET_COL}' based on the following information:
        {dataset_info}

        **Example for Heart Failure Prediction:**
        - Atherosclerosis: Hardening of arteries, a primary cause.
        - High Blood Pressure: Damages arteries and the heart muscle.
        - Diabetes: High blood sugar damages blood vessels.
        - Sedentary Lifestyle: Contributes to obesity and poor cardiovascular health.

        **Instructions:**
        1. Identify 5-7 key causative mechanisms for the condition described.
        2. For each mechanism, create an 'intermediate' node.
        3. Provide a detailed description for each mechanism, explaining its role.
        4. Your response should ONLY contain these intermediate nodes and a brief rationale.
        """
    initial_response = agent.run(mechanism_prompt)
    kg = initial_response.content
    intermediate_nodes = [node for node in kg.nodes if node.node_type == 'intermediate']
    print(f"Identified {len(intermediate_nodes)} initial mechanisms.")

    # --- Step 2: Connect Features to Mechanisms ---
    print("\n--- Step 2: Connecting Features to Mechanisms ---")
    
    # Create a base graph with all features as input nodes
    for feature in feature_names:
        # Check if node already exists to avoid duplicates
        if not any(node.name == feature for node in kg.nodes):
            kg.nodes.append(Node(name=feature, node_type='input', description='Dataset feature.'))
            
    intermediate_names = [node.name for node in intermediate_nodes]
    
    # Loop through each mechanism and find relevant features
    for mechanism in intermediate_names:
        feature_prompt = f"""
        **Causative Mechanism:**
        {mechanism}

        **Dataset Features:**
        {feature_list}

        **Instructions:**
        1. Identify ALL features from the list that are directly related to the '{mechanism}'.
        2. For each related feature, create an edge connecting it to the mechanism.
        3. The 'relationship' for each edge should explain HOW the feature relates to the mechanism.
        
        Example:
        - Source: 'High Blood Pressure' (mechanism)
        - Target: 'ABPmmmHg' (feature)
        - Relationship: 'Represents the direct measurement of arterial blood pressure, a key indicator of hypertension.'
        """
        try:
            feature_response = agent.run(feature_prompt)
            # Add new edges to our graph
            if feature_response.content.edges:
                kg.edges.extend(feature_response.content.edges)
                print(f"Connected {len(feature_response.content.edges)} features to '{mechanism}'.")
        except Exception as e:
            print(f"Error connecting features for '{mechanism}': {e}")
            
    # --- Step 3: Validate and Refine for Missing Features ---
    print("\n--- Step 3: Validating Feature Coverage ---")
    
    # Find features not yet connected to any intermediate nodes
    intermediate_names_set = {node.name for node in kg.nodes if node.node_type == 'intermediate'}
    connected_features = set()
    
    for edge in kg.edges:
        # If the source is an intermediate node, the target is a connected feature
        if edge.source in intermediate_names_set:
            connected_features.add(edge.target)
        # If the target is an intermediate node, the source is a connected feature  
        elif edge.target in intermediate_names_set:
            connected_features.add(edge.source)
    
    missing_features = [f for f in feature_names if f not in connected_features]
    print(f"Connected features: {len(connected_features)}, Missing: {len(missing_features)}")

    if missing_features:
        print(f"Found {len(missing_features)} features not connected to any mechanism. Refining...")
        missing_list = "\n".join([f"- {f}" for f in missing_features])
        
        refine_prompt = f"""
        **Unassigned Features:**
        {missing_list}

        **Existing Mechanisms:**
        {intermediate_names}

        **Instructions:**
        1. For EACH unassigned feature, determine the most relevant mechanism(s).
        2. Create edges to connect these features to the appropriate mechanisms.
        3. Provide a clear 'relationship' description for each new edge.
        """
        try:
            refine_response = agent.run(refine_prompt)
            if refine_response.content.edges:
                kg.edges.extend(refine_response.content.edges)
                print(f"Connected {len(refine_response.content.edges)} missing features.")
        except Exception as e:
            print(f"Error during refinement: {e}")

    # --- Finalize and Save ---
    print("\n--- Finalizing Knowledge Graph ---")
    G = nx.Graph()
    for node in kg.nodes:
        G.add_node(node.name, node_type=node.node_type, description=node.description or '')
    for edge in kg.edges:
        G.add_edge(edge.source, edge.target, relationship=edge.relationship)

    # Final validation with same logic
    final_intermediate_names = {node.name for node in kg.nodes if node.node_type == 'intermediate'}
    final_connected_features = set()
    
    for edge in kg.edges:
        if edge.source in final_intermediate_names:
            final_connected_features.add(edge.target)
        elif edge.target in final_intermediate_names:
            final_connected_features.add(edge.source)
    
    final_missing_count = len([f for f in feature_names if f not in final_connected_features])

    print(f"Final KG has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    print(f"Features missing from graph: {final_missing_count}")
    
    os.makedirs('kg', exist_ok=True)
    nx.write_graphml(G, f'kg/{DATASET_NAME}_initial_agent_kg.graphml')
    print(f"Saved final KG to 'kg/{DATASET_NAME}_initial_agent_kg.graphml'")
    
    return kg

if __name__ == "__main__":
    create_kg()