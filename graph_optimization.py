from sklearn.metrics import accuracy_score, roc_auc_score
import params
import optuna
from sklearn.base import clone
import numpy as np
from visualizations import visualize_pareto_front
import json
import networkx as nx

optuna.logging.set_verbosity(optuna.logging.WARNING)

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_j] = root_i

def get_feature_importances(X_train, y_train, X_val, y_val):
    """Calculate and normalize feature importances using a baseline model"""

    model = clone(params.ML_MODEL)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    
    # Get feature importances and normalize them
    importances = model.feature_importances_
    normalized_importances = importances / np.sum(importances)
    
    # Create mapping from feature name to normalized importance
    importance_dict = dict(zip(X_train.columns, normalized_importances))
    return importance_dict

def objective(trial, X_train, y_train, X_val, y_val, col_to_idx, 
              feature_importances, max_importance, feature_to_original_groups, feature_names, num_initial_groups):
    
    # Step 1: Decide pairwise merges only - each group can merge with at most one other group
    uf = UnionFind(num_initial_groups)
    
    # For each group, decide if it should merge with another group (or stay alone)
    for i in range(num_initial_groups):
        # Create choices: stay alone (i) or merge with another group (0 to num_initial_groups-1, excluding i)
        merge_choices = ['stay_alone'] + [j for j in range(num_initial_groups) if j != i]
        chosen_action = trial.suggest_categorical(f'group_{i}_merge_with', merge_choices)
        
        if chosen_action != 'stay_alone':
            # Only merge if the other group hasn't already been merged
            if uf.find(i) != uf.find(chosen_action):
                uf.union(i, chosen_action)
    
    # Create mapping from original group to merged group representative
    group_representatives = {}
    merged_group_ids = set()
    for i in range(num_initial_groups):
        rep = uf.find(i)
        group_representatives[i] = rep
        merged_group_ids.add(rep)
    
    merged_group_list = sorted(list(merged_group_ids))
    
    # Step 2: Assign each feature to exactly one merged group (or exclude it)
    merged_groups = {rep: [] for rep in merged_group_list}
    included_features = 0
    
    for feature in feature_names:
        importance = feature_importances.get(feature, 0.0)
        normalized_importance = importance / max_importance
        inclusion_prob = params.INCLUSION_BASE_PROB + (normalized_importance * params.INCLUSION_IMPORTANCE_SCALE)

        if trial.suggest_float(f'include_{feature}', 0, 1) < inclusion_prob:
            included_features += 1
            original_groups = feature_to_original_groups.get(feature, [])
            if original_groups:
                # Use consistent choice space: all possible group representatives (0 to num_initial_groups-1)
                # This ensures Optuna sees the same choices for each feature across all trials
                chosen_group_idx = trial.suggest_categorical(f'assign_{feature}', list(range(num_initial_groups)))
                
                # Map the chosen index to the actual merged group representative
                chosen_merged_group = group_representatives[chosen_group_idx]
                
                # Only assign if this merged group is actually available for this feature
                available_merged_groups = set(group_representatives[g] for g in original_groups)
                if chosen_merged_group in available_merged_groups:
                    merged_groups[chosen_merged_group].append(feature)

    # Step 3: Create final constraints
    merged_constraints = [list(set(group)) for group in merged_groups.values() if len(set(group)) > 1]
    
    # If no interaction constraints are created, don't select any features
    if not merged_constraints:
        lgbm_constraints = []
        # Set a penalty score to discourage this solution
        val_score = 0.0
        total_edges = 0
    else:
        lgbm_constraints = [[col_to_idx[feat] for feat in group] for group in merged_constraints]
        
        model = clone(params.ML_MODEL)
        model.set_params(interaction_constraints=lgbm_constraints)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        
        val_preds = model.predict_proba(X_val)[:, 1] if params.METRIC == 'roc_auc' else model.predict(X_val)
        val_score = roc_auc_score(y_val, val_preds) if params.METRIC == 'roc_auc' else accuracy_score(y_val, val_preds)
        
        # Calculate total number of possible interactions (edges) within all constraint groups
        total_edges = 0
        for group in merged_constraints:
            n_features = len(group)
            if n_features > 1:
                # Number of possible interactions in a group of n features = n*(n-1)/2
                total_edges += n_features * (n_features - 1) // 2
    
    return val_score, total_edges

def optimize_graph(X_train, y_train, X_val, y_val, n_trials, initial_constraints):
    if params.VERBOSE:
        print(f"Dataset: {params.DATASET_NAME}")
        print(f"Total features: {len(X_train.columns)}")
        print(f"Initial constraint groups: {len(initial_constraints)}")
    
    with open(f'kg/{params.DATASET_NAME}_interaction_constraints.json', 'r') as f:
        constraints_data = json.load(f)
        initial_labels = constraints_data.get('constraint_labels', ['Unknown'] * len(initial_constraints))
    
    feature_importances = get_feature_importances(X_train, y_train, X_val, y_val)
    max_importance = max(feature_importances.values()) if feature_importances else 1
    
    feature_to_original_groups = {}
    for group_idx, group in enumerate(initial_constraints):
        for feature in group:
            feature_to_original_groups.setdefault(feature, []).append(group_idx)

    X_train_np, y_train_np = X_train.to_numpy(), y_train.to_numpy()
    X_val_np, y_val_np = X_val.to_numpy(), y_val.to_numpy()
    feature_names = X_train.columns.to_list()
    col_to_idx = {col: idx for idx, col in enumerate(feature_names)}
    num_initial_groups = len(initial_constraints)
    
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(directions=['maximize', 'minimize'], pruner=pruner)
    study.optimize(lambda t: objective(t, X_train_np, y_train_np, X_val_np, y_val_np, col_to_idx, 
                                       feature_importances, max_importance, feature_to_original_groups, 
                                       feature_names, num_initial_groups), 
                   n_trials=n_trials, show_progress_bar=False)

    selected_trial = max(study.best_trials, key=lambda t: t.values[0] - t.values[1] * params.EDGE_PENALTY)
    if params.VERBOSE:
        print(f"\nSelected trial with: Score={selected_trial.values[0]:.4f}, Edges={selected_trial.values[1]}")
    if params.PLOT_IMAGES:
        visualize_pareto_front(study, params.DATASET_NAME, selected_trial)

    # Step 1: Reconstruct which groups were merged
    uf = UnionFind(num_initial_groups)
    
    for i in range(num_initial_groups):
        chosen_action = selected_trial.params.get(f'group_{i}_merge_with', 'stay_alone')
        if chosen_action != 'stay_alone':
            # Only merge if the other group hasn't already been merged
            if uf.find(i) != uf.find(chosen_action):
                uf.union(i, chosen_action)
    
    # Create mapping from original group to merged group representative
    group_representatives = {}
    merged_group_ids = set()
    for i in range(num_initial_groups):
        rep = uf.find(i)
        group_representatives[i] = rep
        merged_group_ids.add(rep)
    
    merged_group_list = sorted(list(merged_group_ids))
    
    # Step 2: Assign features to merged groups
    merged_groups = {rep: [] for rep in merged_group_list}
    included_features = 0
    
    for feature in feature_names:
        importance = feature_importances.get(feature, 0.0)
        normalized_importance = importance / max_importance
        inclusion_prob = params.INCLUSION_BASE_PROB + (normalized_importance * params.INCLUSION_IMPORTANCE_SCALE)

        if selected_trial.params.get(f'include_{feature}', 1.0) < inclusion_prob:
            included_features += 1
            original_groups = feature_to_original_groups.get(feature, [])
            if original_groups:
                # Get the chosen group index from the trial
                chosen_group_idx = selected_trial.params.get(f'assign_{feature}')
                if chosen_group_idx is not None:
                    # Map to the actual merged group representative
                    chosen_merged_group = group_representatives[chosen_group_idx]
                    
                    # Only assign if this merged group is actually available for this feature
                    available_merged_groups = set(group_representatives[g] for g in original_groups)
                    if chosen_merged_group in available_merged_groups:
                        merged_groups[chosen_merged_group].append(feature)

    # Create final constraints
    merged_constraints = [list(set(group)) for group in merged_groups.values() if len(set(group)) > 1]

    if params.VERBOSE:
        print(f"Final included features: {included_features}/{len(feature_names)}")
        print(f"Final constraint groups: {len(merged_constraints)}")
        print("Final constraint group sizes:")
        for i, group in enumerate(merged_constraints):
            print(f"  Group {i+1}: {len(group)} features")

    optimized_kg = nx.Graph()
    for feature in X_train.columns:
        optimized_kg.add_node(feature)
    for group in merged_constraints:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                optimized_kg.add_edge(group[i], group[j])
                
    return merged_constraints, initial_labels