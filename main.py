from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
from params import *
import json
import os
from graph_optimization import optimize_graph
from create_kg import create_kg
from visualizations import visualize_knowledge_graph
import copy
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
        G, mechanisms, initial_constraints, labels = create_kg(df)
        
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
            title=f'Initial {dataset_name.upper()} Knowledge Graph',
            filename=f'{dataset_name}_initial_kg.png',
            feature_to_labels_map=feature_to_labels_map
        )
    
    # Run graph optimization
    best_constraints, initial_labels = optimize_graph(X_train, y_train, X_val, y_val, N_TRIALS, initial_constraints)
    
    # Visualize the optimized KG
    optimized_feature_to_labels_map = {}
    optimized_labels = []
    
    for i, group in enumerate(best_constraints):
        if not group:  # Skip empty groups
            continue
            
        # For each group, find which original groups contributed features
        contributing_original_groups = set()
        for feature in group:
            for orig_idx, orig_group in enumerate(initial_constraints):
                if feature in orig_group:
                    contributing_original_groups.add(orig_idx)
        
        # Create merged label using "/" to separate original labels
        if contributing_original_groups:
            contributing_labels = [initial_labels[idx] for idx in sorted(contributing_original_groups) if idx < len(initial_labels)]
            merged_label = "/".join(contributing_labels) if len(contributing_labels) > 1 else contributing_labels[0] if contributing_labels else f"Group_{i+1}"
        else:
            merged_label = f"Group_{i+1}"
        
        optimized_labels.append(merged_label)
        
        # Assign this merged label to all features in the group
        for feature in group:
            if feature not in optimized_feature_to_labels_map:
                optimized_feature_to_labels_map[feature] = []
            optimized_feature_to_labels_map[feature].append(merged_label)

    if PLOT_IMAGES: 
        visualize_knowledge_graph(
            constraints=best_constraints,
            labels=optimized_labels,
            title=f'Optimized {dataset_name.upper()} Knowledge Graph',
            filename=f'{dataset_name}_optimized_kg.png',
            feature_to_labels_map=optimized_feature_to_labels_map
        )

    # Remove isolated features (features not in any constraint group)
    unique_features_in_constraints = set(feature for group in best_constraints for feature in group)
    isolated_features = [feature for feature in X_train.columns if feature not in unique_features_in_constraints]
    
    if isolated_features:
        X_train_reduced = X_train.drop(isolated_features, axis=1)
        X_val_reduced = X_val.drop(isolated_features, axis=1)
    else:
        X_train_reduced = X_train
        X_val_reduced = X_val
    
    # Convert feature names to indices for interaction constraints
    final_constraints = [[X_train_reduced.columns.get_loc(feature) for feature in group] for group in best_constraints if group]
    
    # Clone model and set constraints
    model = copy.deepcopy(model)
    model.set_params(interaction_constraints=final_constraints)
    
    # Fit the model
    model.fit(X_train_reduced, y_train, eval_set=[(X_val_reduced, y_val)])
    
    # Store the removed features so they can be removed from test set
    model.removed_features_ = isolated_features
    model.feature_names_ = X_train_reduced.columns.tolist()
    model.constraint_groups_ = best_constraints
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
    print(f"Model saved to models/{DATASET_NAME}_grace_model.joblib")
    
    # Apply to test set
    X_test = apply_grace_to_test(model, X_test)
    
    # Make predictions
    test_preds = PREDICT_FN(model, X_test)
    test_score = roc_auc_score(y_test, test_preds) if METRIC == 'roc_auc' else accuracy_score(y_test, test_preds)
    print(f"Final Test Score: {test_score:.4f}")