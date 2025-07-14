from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
from params import *
import json
import os
from graph_optimization import optimize_graph
from create_kg import create_kg
from visualizations import visualize_knowledge_graph
import warnings
import joblib

warnings.filterwarnings("ignore", category=FutureWarning)

def grace_feature_selection(X_train, y_train, X_val, y_val, model, dataset_name=None):
    """
    GRACE feature selection using knowledge graph optimization.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        model: ML model to use (from params.py)
        dataset_name: Optional dataset name (defaults to DATASET_NAME from params)
    
    Returns:
        Fitted model with optimized constraints
    """
    if dataset_name is None:
        dataset_name = DATASET_NAME
    
    # Load initial constraints from KG creation
    if LOAD_AGENT_KG and os.path.exists(f'kg/{dataset_name}_interaction_constraints.json'):
        with open(f'kg/{dataset_name}_interaction_constraints.json', 'r') as f:
            constraints_data = json.load(f)
            initial_constraints = constraints_data['interaction_constraints']
            labels = constraints_data['constraint_labels']
    else:
        df = pd.read_csv(f'datasets/{dataset_name}.csv')
        initial_constraints, labels = create_kg(df)
        
    if PLOT_IMAGES:
        # Visualize and save the initial KG
        feature_to_labels_map = {}
        for i, group in enumerate(initial_constraints):
            label_for_group = labels[i]
            for feature in group:
                if feature not in feature_to_labels_map:
                    feature_to_labels_map[feature] = []
                feature_to_labels_map[feature].append(label_for_group)
        visualize_knowledge_graph(
            constraints=initial_constraints, 
            labels=labels,
            filename=f'{dataset_name}_initial_kg.png',
            feature_to_labels_map=feature_to_labels_map
        )
    
    # Run graph optimization
    optimized_constraints, optimized_labels, best_params = optimize_graph(X_train, y_train, X_val, y_val, N_TRIALS, initial_constraints, labels)

    if PLOT_IMAGES: 
        # Create a new feature_to_labels_map for the OPTIMIZED graph
        optimized_feature_to_labels_map = {}
        for i, group in enumerate(optimized_constraints):
            label_for_group = optimized_labels[i]
            for feature in group:
                if feature not in optimized_feature_to_labels_map:
                    optimized_feature_to_labels_map[feature] = []
                optimized_feature_to_labels_map[feature].append(label_for_group)
        
        visualize_knowledge_graph(
            constraints=optimized_constraints,
            labels=optimized_labels,
            filename=f'{dataset_name}_optimized_kg.png',
            feature_to_labels_map=optimized_feature_to_labels_map
        )

    # Convert feature names to indices for LightGBM constraints
    col_to_idx = {col: idx for idx, col in enumerate(X_train.columns)}
    model_hyperparams = {k: v for k, v in best_params.items() if k in HPARAMS.keys()}
    if VERBOSE:
        print(f"Best ML model parameters: {model_hyperparams}")
    lgbm_constraints = [[col_to_idx[feat] for feat in group] for group in optimized_constraints]
    model.set_params(interaction_constraints=lgbm_constraints, **model_hyperparams)
    
    # Fit the model
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    
    # Store the removed features so they can be removed from test set
    model.removed_features_ = []
    model.feature_names_ = X_train.columns.tolist()
    model.constraint_groups_ = optimized_constraints
    model.constraint_labels_ = optimized_labels
    return model

def apply_grace_to_test(model, X_test):
    """Apply GRACE model to test set by removing the same features."""
    if hasattr(model, 'removed_features_') and model.removed_features_:
        return X_test.drop(model.removed_features_, axis=1)
    return X_test

if __name__ == "__main__":
    # Original main.py functionality for testing
    df = pd.read_csv(f'datasets/{DATASET_NAME}.csv')
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]
    
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=VAL_SIZE, random_state=42, stratify=y_train_full)
    
    model = grace_feature_selection(X_train, y_train, X_val, y_val, ML_MODEL)
    
    # Save the trained model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, f'models/{DATASET_NAME}_grace_model.joblib')
    
    # Apply to test set
    X_test_processed = apply_grace_to_test(model, X_test)
    
    # Make predictions
    test_preds = PREDICT_FN(model, X_test_processed)
    test_score = roc_auc_score(y_test, test_preds) if METRIC == 'roc_auc' else accuracy_score(y_test, test_preds)
    print(f"Final Test Score: {test_score:.4f}")
    
    # Run interaction analysis if enabled
    if EXPLAIN_RESULTS:
        best_constraints = model.constraint_groups_
        best_labels = model.constraint_labels_
        optimized_feature_to_labels_map = {
            col: [best_labels[i] for i, group in enumerate(best_constraints) if col in group]
            for col in X_test.columns
        }
        # Create the results dictionary for the explainer even when not optimizing
        optimized_results = {
            'best_constraints': best_constraints,
            'best_labels': best_labels,
            'feature_to_labels_map': optimized_feature_to_labels_map
        }

        if VERBOSE:
            print(f"Final model using {len(best_constraints)} constraint groups")
            print(f"Constraint group sizes: {[len(group) for group in best_constraints]}")
        
        from analysis import run_interaction_analysis
        run_interaction_analysis(model, X_test, model.feature_names_, DATASET_NAME, optimized_results)