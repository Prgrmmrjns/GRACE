import shap
import numpy as np
import networkx as nx
import pandas as pd
import optuna
from sklearn.base import clone
from sklearn.metrics import accuracy_score, roc_auc_score
from params import MIN_SHAP_THRESHOLD_RANGE, MIN_INTERACTION_THRESHOLD_RANGE, N_TRIALS, FEATURE_PENALTY_COEFF, EDGE_PENALTY_COEFF

def build_constraints_from_interactions(feature_interactions, features):
    constraints_set = set()
    
    for interaction_obj in feature_interactions:
        if len(interaction_obj['interactions']) > 0:
            f1_name = interaction_obj['feature']
            current_group_names = {f1_name}
            for f2_name in interaction_obj['interactions']:
                current_group_names.add(f2_name)
            
            current_group_indices = {features.index(name) for name in current_group_names if name in features}
            if current_group_indices:
                sorted_group_tuple = tuple(sorted(list(current_group_indices)))
                constraints_set.add(sorted_group_tuple)
    
    return [list(group) for group in constraints_set]

def set_interaction_constraints(model, constraints, feature_names=None):
    """Set interaction constraints for both LightGBM and XGBoost models."""
    xgb_constraints = []
    for constraint_group in constraints:
        feature_name_group = [feature_names[idx] for idx in constraint_group if idx < len(feature_names)]
        if feature_name_group:
            xgb_constraints.append(feature_name_group)
    model.set_params(interaction_constraints=xgb_constraints)
    return model

def calculate_shap_values(model, X_train, X_val):
    explainer = shap.TreeExplainer(model)
    X_train_val = pd.concat([X_train, X_val])
    shap_values = explainer(X_train_val)
    shap_vals = shap_values.values
    
    # Handle multiclass: take the class with highest average absolute SHAP values
    if len(shap_vals.shape) == 3:
        class_shap_importance = [np.mean(np.abs(shap_vals[:, :, i])) for i in range(shap_vals.shape[2])]
        best_class = np.argmax(class_shap_importance)
        shap_vals = shap_vals[:, :, best_class]
    
    return shap_vals
    
def shap_based_selection(shap_values, feature_interactions, feature_names, min_shap_threshold, min_interaction_threshold, dataset_name):
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    selected_features = [
        feature_names[i] for i, shap_importance in enumerate(mean_abs_shap)
        if shap_importance > min_shap_threshold
    ]
    
    # Fallback: if no features selected, take top 5
    if len(selected_features) == 0:
        top_indices = np.argsort(mean_abs_shap)[-5:]
        selected_features = [feature_names[i] for i in top_indices]
    
    # Calculate SHAP interaction matrix for selected features
    selected_indices = [feature_names.index(f) for f in selected_features]
    selected_shap_values = shap_values[:, selected_indices]
    interaction_matrix = np.corrcoef(selected_shap_values.T)
    G_filtered = nx.Graph()
    for feature in selected_features:
        G_filtered.add_node(feature, entity_type='INPUT_NODE')
    
    feature_has_edge = {feature: False for feature in selected_features}
    
    # Create filtered interactions based on SHAP values
    filtered_interactions = []
    
    for interaction_obj in feature_interactions:
        if interaction_obj['feature'] in selected_features:
            valid_interactions = []
            for interacting_feature in interaction_obj['interactions']:
                if (interacting_feature in selected_features and 
                    interaction_obj['feature'] != interacting_feature):
                    feature_has_edge[interaction_obj['feature']] = True
                    feature_has_edge[interacting_feature] = True
                    f1_idx = selected_features.index(interaction_obj['feature'])
                    f2_idx = selected_features.index(interacting_feature)
                    
                    # Check SHAP interaction strength
                    if f1_idx != f2_idx and not np.isnan(interaction_matrix[f1_idx, f2_idx]):
                        interaction_strength = abs(interaction_matrix[f1_idx, f2_idx])
                        if interaction_strength > min_interaction_threshold:
                            G_filtered.add_edge(interaction_obj['feature'], interacting_feature, relationship='interaction')
                            valid_interactions.append(interacting_feature)
            
            # If this feature has valid interactions, count it as a constraint
            if valid_interactions:
                filtered_interactions.append({
                    'feature': interaction_obj['feature'],
                    'interactions': valid_interactions
                })
    
    # Count unique undirected edges (each pair counted only once)
    edge_pairs = set()
    for interaction in filtered_interactions:
        for target in interaction['interactions']:
            # Store edge in canonical form (sorted) to avoid duplicates
            edge = tuple(sorted([interaction['feature'], target]))
            edge_pairs.add(edge)
    
    # The number of unique undirected edges
    filtered_edge_count = len(edge_pairs)
    
    # Remove features without any edges
    features_with_edges = [f for f in selected_features if feature_has_edge[f]]
    isolated_features = [f for f in selected_features if not feature_has_edge[f]]
    
    if isolated_features:
        for f in isolated_features:
            G_filtered.remove_node(f)
    
    nx.write_graphml(G_filtered, f"kg/{dataset_name}_filtered.graphml")
    return features_with_edges, G_filtered, filtered_edge_count

def optimize_thresholds(X_train, X_val, y_train, y_val, shap_values, feature_interactions, 
                       feature_names, ml_model, metric, early_stopping_rounds):
    
    # Pre-compute mean absolute SHAP values
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    
    # Pre-compute feature interaction mappings for faster lookup
    feature_to_interactions = {}
    for interaction_obj in feature_interactions:
        feature_to_interactions[interaction_obj['feature']] = set(interaction_obj['interactions'])
    
    # Pre-compute feature name to index mapping
    feature_name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    
    def objective(trial):
        min_shap_threshold = trial.suggest_float('min_shap_threshold', *MIN_SHAP_THRESHOLD_RANGE)
        min_interaction_threshold = trial.suggest_float('min_interaction_threshold', *MIN_INTERACTION_THRESHOLD_RANGE)
        
        # Fast feature selection using numpy masking
        selected_mask = mean_abs_shap > min_shap_threshold
        selected_indices = np.where(selected_mask)[0]
        
        # Use numpy indexing for feature names and SHAP values
        selected_features = [feature_names[i] for i in selected_indices]
        selected_features_set = set(selected_features)
        selected_shap_values = shap_values[:, selected_indices]
        
        # Fast correlation matrix computation
        interaction_matrix = np.corrcoef(selected_shap_values.T)
        
        # Vectorized interaction processing
        valid_edges = []
        features_with_edges = set()
        
        for i, feature in enumerate(selected_features):
            if feature in feature_to_interactions:
                # Fast set intersection
                valid_targets = selected_features_set & feature_to_interactions[feature]
                
                for target in valid_targets:
                    j = selected_features.index(target)
                    if i != j and not np.isnan(interaction_matrix[i, j]):
                        interaction_strength = abs(interaction_matrix[i, j])
                        if interaction_strength > min_interaction_threshold:
                            valid_edges.append((feature, target))
                            features_with_edges.add(feature)
                            features_with_edges.add(target)
        
        # Fast constraint building - group features that interact
        constraint_groups = []
        
        for feature in features_with_edges:
            # Find all features connected to this one
            group = {feature}
            stack = [feature]
            
            while stack:
                current = stack.pop()
                for edge_feature, edge_target in valid_edges:
                    if edge_feature == current and edge_target not in group:
                        group.add(edge_target)
                        stack.append(edge_target)
                    elif edge_target == current and edge_feature not in group:
                        group.add(edge_feature)
                        stack.append(edge_feature)
            
            # Convert to indices
            group_indices = [feature_name_to_idx[f] for f in group if f in feature_name_to_idx]
            constraint_groups.append(sorted(group_indices))
        
        # Get final feature list and indices
        final_features = list(features_with_edges)
        final_indices = np.array([feature_name_to_idx[f] for f in final_features])
        
        # Train model with selected features and constraints
        X_train_filtered = X_train.iloc[:, final_indices]
        X_val_filtered = X_val.iloc[:, final_indices]
        
        model = clone(ml_model)
        
        # Fast constraint setting
        xgb_constraints = [[final_features[final_indices.tolist().index(idx)] for idx in group if idx in final_indices] for group in constraint_groups]
        model.set_params(interaction_constraints=xgb_constraints, early_stopping_rounds=early_stopping_rounds)
        model.fit(X_train_filtered, y_train, eval_set=[(X_val_filtered, y_val)], verbose=False)       
        
        # Evaluate
        if metric == 'accuracy':
            y_pred = model.predict(X_val_filtered)
            val_score = accuracy_score(y_val, y_pred)
        else:
            y_pred_proba = model.predict_proba(X_val_filtered)[:, 1]
            val_score = roc_auc_score(y_val, y_pred_proba)
        
        # Multi-objective: maximize val_score, minimize features and edges
        n_features = len(final_features)
        n_edges = len(valid_edges)
        return val_score - FEATURE_PENALTY_COEFF * n_features - EDGE_PENALTY_COEFF * n_edges
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=-1, show_progress_bar=True)
    
    return study.best_params, study.best_value