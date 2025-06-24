import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
import os
from sklearn.metrics import roc_auc_score, accuracy_score
from create_kg import create_kg
from params import (DATASET_NAME, TARGET_COL, TEST_SIZE, VAL_SIZE,
                    ML_MODEL, METRIC, TARGET_COL,
                    LOAD_AGENT_KG, AGENT_KG_PATH)
import networkx as nx
import joblib
from sklearn.preprocessing import LabelEncoder
from graph_reduction import optimize_graph
from utils import (networkx_to_model, get_mechanism_to_features)
from visualizations import visualize_kg_structure
import copy

warnings.filterwarnings('ignore')

def main():
    """Train and optimize a model using a knowledge graph."""
    df = pd.read_csv(f'datasets/{DATASET_NAME}.csv')
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]
    if len(y.unique()) > 2:
        le = LabelEncoder()
        y = le.fit_transform(y)
    # Keep a single train/test split. The optimization will handle internal validation.
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=0)

    if LOAD_AGENT_KG and os.path.exists(AGENT_KG_PATH):
        G = nx.read_graphml(AGENT_KG_PATH)
        kg = networkx_to_model(G)
    else:
        kg = create_kg(df)
        visualize_kg_structure(DATASET_NAME)
    
    feature_names = list(X.columns)
    mechanism_to_features = get_mechanism_to_features(kg, feature_names)
    print(f"Found {len(mechanism_to_features)} mechanism groups to guide optimization.")
    
    # Optimize graph structure to find the best feature set and interaction constraints using CV
    print("\n--- Optimizing graph structure with 5-fold Stratified CV ---")
    final_nodes, final_constraints = optimize_graph(X_train_full, y_train_full, mechanism_to_features)
    
    # --- Final Evaluation ---
    print("\n--- Training final model on optimized graph ---")
    final_nodes_list = sorted(list(final_nodes))
    
    # Create a small holdout from the full training data for final model's early stopping
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=VAL_SIZE, random_state=42, stratify=y_train_full
    )
    
    X_train_reduced = X_train[final_nodes_list]
    X_val_reduced = X_val[final_nodes_list]
    X_test_reduced = X_test[final_nodes_list]
    print(f"Final number of features: {len(final_nodes_list)}")
    print(f"Final interaction constraints ({len(final_constraints)} groups): {final_constraints}")

    model = copy.deepcopy(ML_MODEL)
    model.set_params(interaction_constraints=final_constraints)
    model.fit(X_train_reduced, y_train, eval_set=[(X_val_reduced, y_val)], verbose=False)
    
    # Evaluate
    val_pred = model.predict_proba(X_val_reduced)[:, 1] if METRIC != 'accuracy' else model.predict(X_val_reduced)
    val_score = roc_auc_score(y_val, val_pred) if METRIC != 'accuracy' else accuracy_score(y_val, val_pred)
    print(f"Final holdout validation {METRIC}: {val_score:.4f}")
    
    test_pred = model.predict_proba(X_test_reduced)[:, 1] if METRIC != 'accuracy' else model.predict(X_test_reduced)
    test_score = roc_auc_score(y_test, test_pred) if METRIC != 'accuracy' else accuracy_score(y_test, test_pred)
    print(f"Test {METRIC}: {test_score:.4f}")
    
    joblib.dump(model, f'models/{DATASET_NAME}_kg_constrained_model.joblib')
    return model, test_score

if __name__ == "__main__":
    main()