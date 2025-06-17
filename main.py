import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
import os
from sklearn.metrics import roc_auc_score, accuracy_score
from create_kg import create_kg
from params import (DATASET_NAME, TARGET_COL, TEST_SIZE, VAL_SIZE,
                    ML_MODEL, METRIC, CALLBACKS, TARGET_COL,
                    LOAD_AGENT_KG, AGENT_KG_PATH)
import networkx as nx
import joblib
from graph_reduction import optimize_graph
from utils import (create_interaction_constraints, get_mechanism_to_features,
                   networkx_to_model)
from visualizations import visualize_kg_structure

warnings.filterwarnings('ignore')

def main():
    """Train basic LightGBM with KG-derived interaction constraints"""
    df = pd.read_csv(f'datasets/{DATASET_NAME}.csv')
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=VAL_SIZE, random_state=42)
    if LOAD_AGENT_KG and os.path.exists(AGENT_KG_PATH):
        # Load from GraphML
        G = nx.read_graphml(AGENT_KG_PATH)
        kg = networkx_to_model(G)
    else:
        kg = create_kg(df)
        visualize_kg_structure(DATASET_NAME)
    
    # Create interaction constraints based on shared disease mechanisms
    feature_names = list(X.columns)
    mechanism_to_features = get_mechanism_to_features(kg, feature_names)
    constraint_indices = create_interaction_constraints(mechanism_to_features, feature_names)
    print(f"Created {len(constraint_indices)} interaction constraints (one per disease mechanism)")
    print(f"Constraint sizes: {[len(c) for c in constraint_indices]}")
    
    final_mechanism_to_features = optimize_graph(X_train, y_train, X_val, y_val, mechanism_to_features)
    
    # --- Final Evaluation ---
    print("\n--- Training final model on optimized graph ---")
    
    # Get final nodes from the optimized mechanism groups
    final_nodes = set()
    for features in final_mechanism_to_features.values():
        final_nodes.update(features)
    final_nodes_list = list(final_nodes)
    
    X_train_reduced = X_train[final_nodes_list]
    X_val_reduced = X_val[final_nodes_list]
    X_test_reduced = X_test[final_nodes_list]

    # Create constraints using the same logic as initial model
    final_constraints = create_interaction_constraints(final_mechanism_to_features, final_nodes_list)
    
    print(f"Final number of nodes: {len(final_nodes_list)}")
    print(f"Final number of interaction constraints: {len(final_constraints)}")
    print(f"Final constraint sizes: {[len(c) for c in final_constraints]}")

    model = ML_MODEL
    model.set_params(interaction_constraints=final_constraints)
    model.fit(X_train_reduced, y_train, eval_set=[(X_val_reduced, y_val)], callbacks=CALLBACKS)
    
    # Evaluate
    val_pred = model.predict(X_val_reduced) if METRIC == 'accuracy' else model.predict_proba(X_val_reduced)[:, 1]
    val_score = accuracy_score(y_val, val_pred) if METRIC == 'accuracy' else roc_auc_score(y_val, val_pred)
    print(f"Validation {METRIC} after optimization: {val_score:.4f}")
    
    test_pred = model.predict(X_test_reduced) if METRIC == 'accuracy' else model.predict_proba(X_test_reduced)[:, 1]
    test_score = accuracy_score(y_test, test_pred) if METRIC == 'accuracy' else roc_auc_score(y_test, test_pred)
    print(f"Test {METRIC}: {test_score:.4f}") 
    
    # Save model
    joblib.dump(model, f'models/{DATASET_NAME}_kg_constrained_model.joblib')
    
    return model, test_score

if __name__ == "__main__":
    main()