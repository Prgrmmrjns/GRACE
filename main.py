from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import networkx as nx
import pandas as pd
import os
from create_kg import run_kg_workflows
from train import graph_based_optimization
from visualizations import visualize_knowledge_graph
from params import DATASET_PATH, DATASET_NAME, TARGET_COL, METRIC, LOAD_KG, VISUALIZE_KG, LOAD_OPTIMIZATION_RESULTS, get_model
from graph_utils import nx_to_node_groups

def main():
    print(f"Running {DATASET_NAME}")
    df = pd.read_csv(DATASET_PATH, encoding='utf-8')
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    model = get_model()
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

    if LOAD_OPTIMIZATION_RESULTS and os.path.exists(f"kg/{DATASET_NAME}_optimized.graphml"):
        graph_path = f"kg/{DATASET_NAME}_optimized.graphml"
        raw_graph_nx = nx.read_graphml(graph_path)
        optimized_node_groups = nx_to_node_groups(raw_graph_nx)
    else:
        if LOAD_KG and os.path.exists(f"kg/{DATASET_NAME}.graphml"):
            graph_path = f"kg/{DATASET_NAME}.graphml"
            raw_graph_nx = nx.read_graphml(graph_path)
        else:
            raw_graph_nx = run_kg_workflows()
        node_groups = nx_to_node_groups(raw_graph_nx)
        optimized_node_groups = graph_based_optimization(X_train, y_train, X_val, y_val, node_groups, model, METRIC)

    X_train_graph = pd.DataFrame(index=X_train.index)
    X_val_graph = pd.DataFrame(index=X_val.index)
    X_test_graph = pd.DataFrame(index=X_test.index)
    for node, features in optimized_node_groups.items():
        if features:
            if len(features) == 1:
                X_train_graph[node] = X_train[features[0]]
                X_val_graph[node] = X_val[features[0]]
                X_test_graph[node] = X_test[features[0]]
            else:
                node_model = get_model()
                node_model.fit(X_train[features], y_train, eval_set=[(X_val[features], y_val)])
                X_train_graph[node] = node_model.predict(X_train[features]) if METRIC == 'accuracy' else node_model.predict_proba(X_train[features])[:, 1]
                X_val_graph[node] = node_model.predict(X_val[features]) if METRIC == 'accuracy' else node_model.predict_proba(X_val[features])[:, 1]
                X_test_graph[node] = node_model.predict(X_test[features]) if METRIC == 'accuracy' else node_model.predict_proba(X_test[features])[:, 1]
    ensemble_model = get_model()
    ensemble_model.fit(X_train_graph, y_train, eval_set=[(X_val_graph, y_val)])
    y_pred = ensemble_model.predict(X_test_graph) if METRIC == 'accuracy' else ensemble_model.predict_proba(X_test_graph)[:, 1]
    score = accuracy_score(y_test, y_pred) if METRIC == 'accuracy' else roc_auc_score(y_test, y_pred)
    print(f"Score ({METRIC.upper()}): {score:.4f}")
        
    if VISUALIZE_KG:
        if not LOAD_OPTIMIZATION_RESULTS:
            visualize_knowledge_graph(raw_graph_nx, f"{DATASET_NAME}_initial", node_groups)
        visualize_knowledge_graph(raw_graph_nx, f"{DATASET_NAME}_optimized", optimized_node_groups, store_graph=True)

if __name__ == "__main__":
    main() 