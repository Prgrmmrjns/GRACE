from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
import networkx as nx
from params import (
    DATASET_PATH, DATASET_NAME, TARGET_COL, TEST_SIZE, VAL_SIZE,
    VISUALIZE_KG, LOAD_KG, ML_MODEL, METRIC, EARLY_STOPPING_CALLBACK, MODEL_TYPE, LOAD_OPTIMIZATION_RESULTS, EXPLAIN_WITH_LLM
)
from create_kg import run_kg_workflows
from graph_utils import nx_to_feature_interactions
from db import setup_lancedb_knowledge_base
from visualizations import visualize_feature_interaction_graph, plot_shap_waterfall, plot_shap_interaction_network
from explainability import run_explainability_analysis, save_explainability_report, run_explainability_analysis
from grace_shap import build_constraints_from_interactions, calculate_shap_values, shap_based_selection, optimize_thresholds, set_interaction_constraints, fit_model
import shap
import numpy as np

def main():
    print(f"Running GRACE pipeline for {DATASET_NAME}")
    
    # Load and split data
    df = pd.read_csv(DATASET_PATH, encoding='utf-8')
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    
    # Handle class labels for XGBoost (needs 0-based labels)
    if MODEL_TYPE == "xgboost" and len(y.unique()) > 2:
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
    
    # Train initial model and get SHAP values
    # Count unique undirected edges in original graph
    original_edge_pairs = set()
    for interaction in feature_interactions:
        for target in interaction['interactions']:
            # Store edge in canonical form (sorted) to avoid duplicates
            edge = tuple(sorted([interaction['feature'], target]))
            original_edge_pairs.add(edge)
    original_edges = len(original_edge_pairs)
    
    print(f"Original unique edges: {original_edges}")
    constraints = build_constraints_from_interactions(feature_interactions, X_train.columns.tolist())
    model = clone(ML_MODEL)
    model = set_interaction_constraints(model, constraints, X_train.columns.tolist())
    model = fit_model(model, X_train, y_train, X_val, y_val, EARLY_STOPPING_CALLBACK)
    
    shap_values, _ = calculate_shap_values(model, X_train, X_val)
    
    # Optimize thresholds
    if LOAD_OPTIMIZATION_RESULTS and os.path.exists(f"results/{DATASET_NAME}_results.csv"):
        results = pd.read_csv(f"results/{DATASET_NAME}_results.csv")
        best_params = {
            'min_shap_threshold': results['best_shap_threshold'].iloc[0],
            'min_interaction_threshold': results['best_interaction_threshold'].iloc[0]
        }
    else:
        best_params = optimize_thresholds(
            X_train, X_val, y_train, y_val, shap_values, feature_interactions,
            X_train.columns.tolist(), ML_MODEL, METRIC, EARLY_STOPPING_CALLBACK
        )
    
    print(f"Best thresholds: {best_params}")
    
    # Use optimized thresholds for final selection
    selected_features, filtered_graph, filtered_edges = shap_based_selection(
        shap_values, feature_interactions, X_train.columns.tolist(), 
        best_params['min_shap_threshold'], best_params['min_interaction_threshold'], DATASET_NAME
    )
    
    # Train final model with selected features
    X_train_filtered = X_train[selected_features]
    X_val_filtered = X_val[selected_features]
    X_test_filtered = X_test[selected_features]
    
    final_constraints = build_constraints_from_interactions(nx_to_feature_interactions(filtered_graph), selected_features)
    model = clone(ML_MODEL)
    model = set_interaction_constraints(model, final_constraints, selected_features)
    model = fit_model(model, X_train_filtered, y_train, X_val_filtered, y_val, EARLY_STOPPING_CALLBACK)
    
    # Evaluate
    if METRIC == 'accuracy':
        y_pred = model.predict(X_test_filtered)
        test_score = accuracy_score(y_test, y_pred)
    else:
        y_pred_proba = model.predict_proba(X_test_filtered)[:, 1]
        test_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Test Score ({METRIC.upper()}): {test_score:.4f}")
    print(f"Features reduced from {len(X.columns)} to {len(selected_features)}")
    print(f"Edges reduced from {original_edges} to {filtered_edges}")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    pd.DataFrame({'feature': selected_features}).to_csv(f"results/{DATASET_NAME}_selected_features.csv", index=False)
    results_df = pd.DataFrame([{
        'test_score': test_score, 
        'features_used': len(selected_features),
        'edges_used': filtered_edges,
        'original_edges': original_edges,
        'best_shap_threshold': best_params['min_shap_threshold'],
        'best_interaction_threshold': best_params['min_interaction_threshold']
    }])
    results_df.to_csv(f"results/{DATASET_NAME}_results.csv", index=False)
    
    if VISUALIZE_KG:
        visualize_feature_interaction_graph(graph_nx, DATASET_NAME)
        visualize_feature_interaction_graph(filtered_graph, DATASET_NAME + "_filtered")
    
    # Run explainability analysis
    model_performance = {
        'accuracy': test_score * 100,  # Convert to percentage
        'features_used': len(selected_features),
        'edges_used': filtered_edges
    }
    
    # Get SHAP values for selected features only
    selected_shap_values = shap_values[:, [X_train.columns.tolist().index(f) for f in selected_features]]
    
    # Get or create knowledge base for explainability
    if 'arxiv_kb_instance' in locals():
        arxiv_kb = arxiv_kb_instance
    else:
        arxiv_kb = setup_lancedb_knowledge_base(queries=[], dataset_name=DATASET_NAME, recreate_db=False)
    
    if EXPLAIN_WITH_LLM:
        explainability_report = run_explainability_analysis(
            shap_values=selected_shap_values, 
            selected_features=selected_features, 
            model_performance=model_performance, 
            arxiv_kb=arxiv_kb, 
            recreate_analysis=True
        )
        save_explainability_report(explainability_report)
    
    # Get base value from SHAP explainer for waterfall plot
    explainer = shap.TreeExplainer(model)
    base_value = explainer.expected_value
    
    # Handle multiclass case
    if isinstance(base_value, np.ndarray):
        base_value = base_value[0]  # Use first class for simplicity
    
    # Create waterfall plot for first sample
    plot_shap_waterfall(
        shap_values=selected_shap_values,
        feature_names=selected_features,
        base_value=base_value,
        sample_idx=0,
        dataset_name=DATASET_NAME,
        max_display=10
    )
    
    # Create SHAP interaction network
    filtered_feature_interactions = nx_to_feature_interactions(filtered_graph)
    plot_shap_interaction_network(
        shap_values=selected_shap_values,
        feature_names=selected_features,
        feature_interactions=filtered_feature_interactions,
        dataset_name=DATASET_NAME
    )

if __name__ == "__main__":
    main() 