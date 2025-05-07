from sklearn.model_selection import train_test_split
import networkx as nx
import pandas as pd
from create_kg import run_kg_workflows
from graph_utils import nx_to_knowledgegraph, NodeType
from train import graph_based_optimization
from visualizations import visualize_knowledge_graph
from params import DATASET_PATH, DATASET_NAME, TARGET_COL, METRIC, LOAD_KG, VISUALIZE_KG, get_base_model, get_callbacks

def main():
    df = pd.read_csv(DATASET_PATH, encoding='utf-8')
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    base_model = get_base_model()
    callbacks = get_callbacks()
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1, random_state=42)
    if LOAD_KG:
        graph_path = f"kg/{DATASET_NAME}.graphml"
        raw_graph_nx = nx.read_graphml(graph_path)
    else:
        raw_graph_nx = run_kg_workflows()
    knowledge_graph = nx_to_knowledgegraph(raw_graph_nx)
    if VISUALIZE_KG:
        # visualize initial KG
        initial_node_groups_for_viz = {}
        for node in knowledge_graph.nodes:
            if node.node_type == NodeType.INTERMEDIATE:
                initial_node_groups_for_viz[node.name] = [e.source for e in node.edges if any(n.name == e.source and n.node_type==NodeType.INPUT for n in knowledge_graph.nodes)]
        visualize_knowledge_graph(raw_graph_nx, DATASET_NAME + "_initial", initial_node_groups_for_viz, initial_node_groups_for_viz)

    # This node_groups is the initial configuration for the optimization process
    node_groups_pre_optimization = {}
    for node in knowledge_graph.nodes:
        if node.node_type == NodeType.INPUT:
            for edge in node.edges:
                target_node = next((n for n in knowledge_graph.nodes if n.name == edge.target), None)
                if target_node and target_node.node_type == NodeType.INTERMEDIATE:
                    if edge.target not in node_groups_pre_optimization:
                        node_groups_pre_optimization[edge.target] = []
                    node_groups_pre_optimization[edge.target].append(node.name)

    optimized_node_groups = graph_based_optimization(
        X_train, y_train, X_val, y_val, X_test, y_test, 
        node_groups_pre_optimization, 
        base_model, METRIC, callbacks,
        raw_graph_nx,
        DATASET_NAME
    )

    if VISUALIZE_KG:
        optimized_kg_path = f"kg/{DATASET_NAME}_optimized.graphml"
        optimized_graph_for_viz = nx.read_graphml(optimized_kg_path)
        final_groups_to_pass_to_viz = optimized_node_groups if optimized_node_groups is not None else node_groups_pre_optimization
        visualize_knowledge_graph(optimized_graph_for_viz, DATASET_NAME+"_post_optimization", node_groups_pre_optimization, final_groups_to_pass_to_viz)

if __name__ == "__main__":
    main() 