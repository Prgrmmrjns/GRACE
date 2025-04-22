from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import networkx as nx
import lightgbm as lgb
from graph_utils import implement_graph_changes, nx_to_knowledgegraph
from train import optimize_intermediate_nodes, analyze_node_relationships, evaluate_final_model
from create_kg import build_kg
from params import DATASET_PATH, DATASET_NAME, TARGET_COL, MODEL, METRIC, VERBOSE, LOAD_KG

def main():
    
    # Run the refinement process
    df = pd.read_csv(DATASET_PATH, encoding='utf-8')
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    unique_classes = len(np.unique(y))
    
    # Set up model parameters
    model_params = {
        'objective': 'binary' if unique_classes == 2 else 'multiclass',
        'learning_rate': 0.1,
        'n_estimators': 1000,
        'min_split_gain': 0,
        'random_state': 42,
        'num_threads': 6,
        'data_sample_strategy': 'goss',
        'use_quantized_grad': True,
        'verbosity': -1
    }
    
    model = lgb.LGBMClassifier(**model_params)
    callbacks = [lgb.early_stopping(stopping_rounds=5, verbose=False)]
    
    # Create train/val/test splits
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1, random_state=42)

    # Load initial graph structure
    if LOAD_KG:
        graph_path = f"kg/{DATASET_NAME}.graphml"
        raw_graph_nx = nx.read_graphml(graph_path)
    else:
        raw_graph_nx = build_kg(DATASET_PATH, DATASET_NAME, TARGET_COL, MODEL, valid_features=list(X.columns))

    # Convert to Pydantic KnowledgeGraph model for optimization functions
    current_graph = nx_to_knowledgegraph(raw_graph_nx)
    removed_nodes = set()
        
    # First round optimization - Select best features for each intermediate node
    X_train_intermediate, X_val_intermediate, X_test_intermediate, features_to_remove, intermediate_to_selected_features, intermediate_models = optimize_intermediate_nodes(
        current_graph, removed_nodes, X_train, y_train, X_val, y_val, X_test, model, callbacks, METRIC, VERBOSE
    )

    # Update model and graph with first round results
    removed_nodes.update(features_to_remove)
    current_graph = implement_graph_changes(current_graph, {"remove_nodes": list(features_to_remove)})
        
    # Second round optimization - Perform error correction between intermediate nodes
    X_train_intermediate, X_val_intermediate, X_test_intermediate, removed_intermediate_nodes = analyze_node_relationships(
        y_train, X_val, y_val, X_train_intermediate, X_val_intermediate, X_test_intermediate, intermediate_to_selected_features, intermediate_models, model, callbacks, METRIC, VERBOSE
    )
    
    # Also update removed_nodes with any removed intermediate nodes
    removed_nodes.update(removed_intermediate_nodes)
    
    # Evaluate on test set
    test_score = evaluate_final_model(X_train_intermediate, X_val_intermediate, X_test_intermediate, y_train, y_val, y_test, model, callbacks, METRIC)
    print(f"Test {METRIC.upper()}: {test_score:.4f}")
    
    # isualize the knowledge graph and modeling process
    #visualize_post_training_graph(graphml_template, removed_nodes,dataset)

if __name__ == "__main__":
    main() 