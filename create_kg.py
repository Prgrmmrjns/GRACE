from agno.agent import Agent
import os
import json
import networkx as nx
from utils import Node, KnowledgeGraphModel
from agno.knowledge.arxiv import ArxivKnowledgeBase
from agno.vectordb.chroma import ChromaDb
from params import (TARGET_COL, EMBEDDING_MODEL, KEYWORDS, LLM_MODEL, DATASET_NAME)

def create_kg(df):
    """Creates a comprehensive, knowledge-driven knowledge graph in a multi-step process."""
    kg = KnowledgeGraphModel(nodes=[], edges=[], rationale="Initial empty graph.")
    
    with open(f'dataset_info/{DATASET_NAME}_info.txt', 'r') as f:
        dataset_info = f.read()
    
    feature_names = [col for col in df.columns.tolist() if col != TARGET_COL]
    feature_list = "\n".join([f"- {feat}" for feat in feature_names])
    
    # Initialize agent outputs storage
    agent_outputs = {
        "dataset_name": DATASET_NAME,
        "target_column": TARGET_COL,
        "total_features": len(feature_names),
        "prompts_and_responses": []
    }

    # --- Agent Setup ---
    vector_db = ChromaDb(
        collection=f"{DATASET_NAME}_kg_collection",
        embedder=EMBEDDING_MODEL,
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
    
    # --- Step 2: Generating Full Knowledge Graph ---
    print("\n--- Step 2: Generating Full Knowledge Graph ---")
    
    # Create a base graph with all features as input nodes for the agent to see
    for feature in feature_names:
        if not any(node.name == feature for node in kg.nodes):
            kg.nodes.append(Node(name=feature, node_type='input', description='Dataset feature.'))

    # Single, powerful prompt to build the entire graph structure
    graph_creation_prompt = f"""
    **Task:** Create a complete knowledge graph by identifying mechanisms and connecting features.

    **Clinical Context:** The model will predict '{TARGET_COL}'.
    {dataset_info}

    **Dataset Features:**
    {feature_list}
    
    **Instructions:**
    1.  **Identify Mechanisms:** First, identify 5-7 key causative biological/clinical mechanisms for the condition. Create an 'intermediate' node for each.
    2.  **Connect Features:** For each feature in the dataset, connect it to the most relevant mechanism(s) by creating an edge. A feature can connect to multiple mechanisms if justified.
    3.  **Return Full Graph:** Your response MUST be a single, complete `KnowledgeGraphModel` containing:
        - All intermediate nodes (mechanisms).
        - All input nodes (features from the dataset).
        - All edges connecting features to mechanisms.
    4.  Do NOT create direct feature-to-feature edges.
    """
    
    try:
        full_graph_response = agent.run(graph_creation_prompt)
        kg = full_graph_response.content
        print(f"Generated graph with {len(kg.nodes)} nodes and {len(kg.edges)} edges in a single step.")
        
        agent_outputs["prompts_and_responses"].append({
            "step": "2_generate_full_graph",
            "prompt": graph_creation_prompt,
            "response": {
                "nodes": [{"name": n.name, "type": n.node_type} for n in kg.nodes],
                "edges": [{"source": e.source, "target": e.target} for e in kg.edges],
            }
        })
    except Exception as e:
        print(f"Error during full graph creation: {e}")
        agent_outputs["prompts_and_responses"].append({
            "step": "2_generate_full_graph",
            "prompt": graph_creation_prompt,
            "error": str(e)
        })

    # --- Finalize and Save ---
    print("\n--- Finalizing Knowledge Graph ---")
    G = nx.Graph()
    for node in kg.nodes:
        G.add_node(node.name, node_type=node.node_type, description=node.description or '')
    for edge in kg.edges:
        G.add_edge(edge.source, edge.target, relationship=edge.relationship)

    # --- Manually add Target Node ---
    print("Adding target node and connecting to all other nodes...")
    G.add_node(TARGET_COL, node_type='target', description='The prediction target variable.')
    all_other_nodes = [node for node in G.nodes() if node != TARGET_COL]
    for node in all_other_nodes:
        node_type = G.nodes[node].get('node_type', 'unknown')
        if node_type == 'input':
            G.add_edge(node, TARGET_COL, relationship='predicts')
        elif node_type == 'intermediate':
            G.add_edge(node, TARGET_COL, relationship='influences')

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
    
    # Add final statistics to agent outputs
    agent_outputs["final_statistics"] = {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "target_node_connections": len([n for n in G.neighbors(TARGET_COL)]),
        "features_missing_from_graph": final_missing_count,
        "intermediate_nodes_final": len([n for n, d in G.nodes(data=True) if d.get('node_type') == 'intermediate']),
        "input_nodes_final": len([n for n, d in G.nodes(data=True) if d.get('node_type') == 'input'])
    }
    
    os.makedirs('kg', exist_ok=True)
    nx.write_graphml(G, f'kg/{DATASET_NAME}_initial_agent_kg.graphml')
    print(f"Saved final KG to 'kg/{DATASET_NAME}_initial_agent_kg.graphml'")
    
    # Save agent outputs to JSON
    with open(f'kg/{DATASET_NAME}_agent_outputs.json', 'w') as f:
        json.dump(agent_outputs, f, indent=2)
    print(f"Saved agent outputs to 'kg/{DATASET_NAME}_agent_outputs.json'")
    
    return kg

if __name__ == "__main__":
    import pandas as pd
    from params import DATASET_PATH
    df = pd.read_csv(DATASET_PATH)
    create_kg(df)