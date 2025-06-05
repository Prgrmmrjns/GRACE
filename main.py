from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
import networkx as nx
from params import (
    DATASET_PATH, DATASET_NAME, TARGET_COL, TEST_SIZE, VAL_SIZE,
    VISUALIZE_KG, LOAD_KG, ML_MODEL, METRIC, EARLY_STOPPING_ROUNDS
)
from create_kg import run_kg_workflows
from graph_utils import nx_to_feature_interactions
from db import setup_lancedb_knowledge_base
from visualizations import visualize_feature_interaction_graph
from grace_shap import build_constraints_from_interactions, calculate_shap_values, shap_based_selection, optimize_thresholds, set_interaction_constraints

def main():
    print(f"Running GRACE pipeline for {DATASET_NAME}")
    
    # Load and split data
    df = pd.read_csv(DATASET_PATH, encoding='utf-8')
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    
    # Handle class labels for XGBoost (needs 0-based labels)
    if len(y.unique()) > 2:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        print(f"Encoded labels for XGBoost: {label_encoder.classes_} -> {list(range(len(label_encoder.classes_)))}")
    
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=VAL_SIZE, random_state=42)
    
    print(f"Dataset: {len(X)} samples, {len(X.columns)} features")
    
    # Load or create knowledge graph
    if LOAD_KG and os.path.exists(f"kg/{DATASET_NAME}.graphml"):
        graph_nx = nx.read_graphml(f"kg/{DATASET_NAME}.graphml")
        feature_interactions = nx_to_feature_interactions(graph_nx)
    else:
        arxiv_kb_instance = setup_lancedb_knowledge_base(queries=[], dataset_name=DATASET_NAME, recreate_db=True) 
        graph_nx = run_kg_workflows(arxiv_kb=arxiv_kb_instance, recreate_search=True)
        feature_interactions = nx_to_feature_interactions(graph_nx)
        # Clean up resources after KG creation
        del arxiv_kb_instance
        import gc
        gc.collect()

    original_edge_pairs = set()
    for interaction in feature_interactions:
        for target in interaction['interactions']:
            # Store edge in canonical form (sorted) to avoid duplicates
            edge = tuple(sorted([interaction['feature'], target]))
            original_edge_pairs.add(edge)
    original_edges = len(original_edge_pairs)
    print(f"Original unique edges: {original_edges}")

    # Initialize variables for the best iteration
    best_score = 0
    best_model = None  # renamed from final_model
    best_selected_features = None  # renamed from final_selected_features
    best_filtered_edges = 0  # renamed from final_filtered_edges
    
    while True:
        constraints = build_constraints_from_interactions(feature_interactions, X_train.columns.tolist())
        model = clone(ML_MODEL)
        model = set_interaction_constraints(model, constraints, X_train.columns.tolist())
        model.set_params(early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        shap_values = calculate_shap_values(model, X_train, X_val)
        # Optimize thresholds
        params, score = optimize_thresholds(
            X_train, X_val, y_train, y_val, shap_values, feature_interactions,
            X_train.columns.tolist(), ML_MODEL, METRIC, EARLY_STOPPING_ROUNDS
        )
        
        if score > best_score:
            best_score = score
            
            # Use optimized thresholds for final selection
            selected_features, filtered_graph, filtered_edges = shap_based_selection(
                shap_values, feature_interactions, X_train.columns.tolist(), 
                params['min_shap_threshold'], params['min_interaction_threshold'], DATASET_NAME
            )
            
            # Train final model with selected features
            X_train_filtered = X_train[selected_features]
            X_val_filtered = X_val[selected_features]
            
            final_constraints = build_constraints_from_interactions(nx_to_feature_interactions(filtered_graph), selected_features)
            final_model = clone(ML_MODEL)
            final_model = set_interaction_constraints(final_model, final_constraints, selected_features)
            final_model.set_params(early_stopping_rounds=EARLY_STOPPING_ROUNDS)
            final_model.fit(X_train_filtered, y_train, eval_set=[(X_val_filtered, y_val)], verbose=False)
            
            # Store the best results
            best_model = final_model  # fixed shadowing bug
            best_selected_features = selected_features
            best_filtered_edges = filtered_edges
        else:
            break

    # Use the best model and features for final evaluation
    X_test_filtered = X_test[best_selected_features]  # using renamed variable
    
    # Evaluate
    if METRIC == 'accuracy':
        y_pred = best_model.predict(X_test_filtered)  # using renamed variable
        test_score = accuracy_score(y_test, y_pred)
    else:
        y_pred_proba = best_model.predict_proba(X_test_filtered)[:, 1]  # using renamed variable
        test_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Test Score ({METRIC.upper()}): {test_score:.4f}")
    print(f"Features reduced from {len(X.columns)} to {len(best_selected_features)}")  # using renamed variable
    print(f"Edges reduced from {original_edges} to {best_filtered_edges}")  # using renamed variable
    
    if VISUALIZE_KG:
        #visualize_feature_interaction_graph(graph_nx, DATASET_NAME)
        # Use the filtered graph from the best iteration
        best_filtered_graph = nx.read_graphml(f"kg/{DATASET_NAME}_filtered.graphml")
        visualize_feature_interaction_graph(best_filtered_graph, DATASET_NAME + "_filtered")

if __name__ == "__main__":
    main() 