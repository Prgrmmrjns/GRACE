from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import networkx as nx
import pandas as pd
from create_kg import run_kg_workflows
from graph_utils import nx_to_knowledgegraph, NodeType
from train import graph_based_optimization
from visualizations import visualize_knowledge_graph
from params import DATASET_PATH, DATASET_NAME, TARGET_COL, METRIC, LOAD_KG, VISUALIZE_KG, get_base_model, model_fit
from sklearn.base import clone

def main():
    print(f"Running {DATASET_NAME}")
    df = pd.read_csv(DATASET_PATH, encoding='utf-8')
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    model = get_base_model()
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    if LOAD_KG:
        graph_path = f"kg/{DATASET_NAME}.graphml"
        raw_graph_nx = nx.read_graphml(graph_path)
    else:
        raw_graph_nx = run_kg_workflows()
    knowledge_graph = nx_to_knowledgegraph(raw_graph_nx)
    if VISUALIZE_KG:
        initial_node_groups_for_viz = {}
        for u, v, data in raw_graph_nx.edges(data=True):
            node_u_data = raw_graph_nx.nodes.get(u, {})
            node_v_data = raw_graph_nx.nodes.get(v, {})
            if node_u_data.get('entity_type') == 'INPUT_NODE' and node_v_data.get('entity_type') == 'INTERMEDIATE_NODE':
                inter_node_name = v
                input_node_name = u
                if inter_node_name not in initial_node_groups_for_viz:
                    initial_node_groups_for_viz[inter_node_name] = []
                if input_node_name not in initial_node_groups_for_viz[inter_node_name]:
                    initial_node_groups_for_viz[inter_node_name].append(input_node_name)
        visualize_knowledge_graph(raw_graph_nx, f"{DATASET_NAME}_initial", initial_node_groups_for_viz)

    node_groups_pre_optimization = {}
    # Process all edges from input nodes to intermediate nodes
    for u, v, data in raw_graph_nx.edges(data=True):
        node_u_data = raw_graph_nx.nodes.get(u, {})
        node_v_data = raw_graph_nx.nodes.get(v, {})
        
        # If this is an edge from input node to intermediate node
        if node_u_data.get('entity_type') == 'INPUT_NODE' and node_v_data.get('entity_type') == 'INTERMEDIATE_NODE':
            # Initialize the intermediate node's group if needed
            if v not in node_groups_pre_optimization:
                node_groups_pre_optimization[v] = []
            
            # Add the input node to the intermediate node's group
            if u not in node_groups_pre_optimization[v]:
                node_groups_pre_optimization[v].append(u)
                
    optimized_node_groups = graph_based_optimization(X_train, y_train, X_val, y_val, node_groups_pre_optimization, model, METRIC)

    if VISUALIZE_KG:
        visualize_knowledge_graph(raw_graph_nx, f"{DATASET_NAME}_optimized", optimized_node_groups, store_graph=True)
    X_train_graph = pd.DataFrame(index=X_train.index)
    X_val_graph = pd.DataFrame(index=X_val.index)
    X_test_graph = pd.DataFrame(index=X_test.index)
    active_node_names_final = []
    for node_name, node_features in optimized_node_groups.items():
        if node_features:
            node_model_final = clone(model)
            model_fit(node_model_final, X_train[node_features], y_train, X_val[node_features], y_val)
            X_train_graph[node_name] = node_model_final.predict(X_train[node_features]) if METRIC == 'accuracy' else node_model_final.predict_proba(X_train[node_features])[:, 1]
            X_val_graph[node_name] = node_model_final.predict(X_val[node_features]) if METRIC == 'accuracy' else node_model_final.predict_proba(X_val[node_features])[:, 1]
            X_test_graph[node_name] = node_model_final.predict(X_test[node_features]) if METRIC == 'accuracy' else node_model_final.predict_proba(X_test[node_features])[:, 1]
            active_node_names_final.append(node_name)

    X_train_graph = X_train_graph[active_node_names_final]
    X_val_graph = X_val_graph[active_node_names_final]
    X_test_graph = X_test_graph[active_node_names_final]
    final_ensemble_model = clone(model)
    model_fit(final_ensemble_model, X_train_graph, y_train, X_val_graph, y_val)
    y_pred = final_ensemble_model.predict(X_test_graph) if METRIC == "accuracy" else final_ensemble_model.predict_proba(X_test_graph)[:, 1]
    score = accuracy_score(y_test, y_pred) if METRIC == "accuracy" else roc_auc_score(y_test, y_pred)
    print(f"Final score: {score:.4f}")

if __name__ == "__main__":
    main() 