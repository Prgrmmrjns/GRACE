import copy
import itertools
import networkx as nx
import numpy as np
import optuna
import shapiq
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd

from params import METRIC, ML_MODEL, N_TRIALS
from visualizations import visualize_pareto_front, visualize_optimized_graph


def get_shapiq_contributions(model, X_train):
    """Calculates global feature and interaction contributions using SHAP-IQ."""
    if len(X_train) > 10:
        X_sample = X_train.sample(n=10, random_state=42)
        print(f"Calculating SHAP-IQ values for {len(X_sample)} random samples to save time...")
    else:
        X_sample = X_train
        print(f"Calculating SHAP-IQ values for all {len(X_sample)} training samples...")

    explainer_params = {"model": model, "max_order": 2, "index": "k-SII"}
    if METRIC == "accuracy": # multiclass
        explainer_params["class_index"] = 1
    explainer = shapiq.TreeExplainer(**explainer_params)

    feature_names = X_train.columns.tolist()
    node_contributions = {name: 0.0 for name in feature_names}
    edge_contributions = {}

    for i in range(len(X_sample)):
        interaction_values = explainer.explain(X_sample.iloc[i])
        for interaction_indices, value in interaction_values.dict_values.items():
            value = abs(value)
            if len(interaction_indices) == 1:
                node_name = feature_names[interaction_indices[0]]
                node_contributions[node_name] += value
            elif len(interaction_indices) == 2:
                node1, node2 = feature_names[interaction_indices[0]], feature_names[interaction_indices[1]]
                edge = tuple(sorted((node1, node2)))
                edge_contributions[edge] = edge_contributions.get(edge, 0.0) + value

    n_samples = len(X_sample)
    node_contributions = {n: v / n_samples for n, v in node_contributions.items()}
    edge_contributions = {e: v / n_samples for e, v in edge_contributions.items()}
    return node_contributions, edge_contributions


def optimize_graph(X_train_full, y_train_full, mechanism_to_features):
    """Optimizes the feature graph using SHAP-IQ, topology, and CV."""
    print("Pre-calculating SHAP-IQ values on model with NO constraints (using full training data)...")
    base_model = copy.deepcopy(ML_MODEL)
    base_model.set_params(early_stopping_rounds=None)
    base_model.fit(X_train_full, y_train_full, verbose=False) # Fit on all data
    node_contributions, edge_contributions = get_shapiq_contributions(base_model, X_train_full)

    G = nx.Graph()
    G.add_nodes_from(X_train_full.columns)
    mechanism_groups = list(mechanism_to_features.values())
    for group in mechanism_groups:
        G.add_edges_from(itertools.combinations(group, 2))
    G.add_edges_from(edge_contributions.keys())
    
    betweenness_centrality = nx.betweenness_centrality(G)
    clustering_coefficient = nx.clustering(G)
    node_values = list(node_contributions.values())
    edge_values = list(edge_contributions.values())
    
    def objective(trial):
        node_thresh = trial.suggest_float("node_contribution_threshold", 1e-10, max(node_values) if node_values else 1e-10, log=True)
        edge_thresh = trial.suggest_float("edge_contribution_threshold", 1e-10, max(edge_values) if edge_values else 1e-10, log=True)
        betweenness_thresh = trial.suggest_float("betweenness_centrality_thresh", 0, max(betweenness_centrality.values()) if betweenness_centrality else 0, log=False)
        clustering_thresh = trial.suggest_float("clustering_coefficient_thresh", 0, max(clustering_coefficient.values()) if clustering_coefficient else 0, log=False)

        active_nodes_base = {
            n for n, c in node_contributions.items() if c >= node_thresh and 
            betweenness_centrality.get(n, 0) >= betweenness_thresh and 
            clustering_coefficient.get(n, 0) >= clustering_thresh
        }
        active_edges = {e for e, c in edge_contributions.items() if c >= edge_thresh}
        features_in_edges = {f for edge in active_edges for f in edge}
        final_active_features = active_nodes_base.union(features_in_edges)
        
        interaction_graph = nx.Graph()
        interaction_graph.add_nodes_from(final_active_features)
        mechanism_edges = {
            edge for group in mechanism_groups 
            for edge in itertools.combinations(sorted([f for f in group if f in final_active_features]), 2)
        }
        interaction_graph.add_edges_from(mechanism_edges.union(active_edges))
        interaction_graph.remove_nodes_from(list(nx.isolates(interaction_graph)))

        if interaction_graph.number_of_nodes() > 0:
            communities = nx.community.greedy_modularity_communities(interaction_graph)
            filtered_constraints = [list(c) for c in communities if len(c) > 1]
        else:
            filtered_constraints = []

        if not final_active_features:
            return 0.0, 0

        # --- Stratified K-Fold Cross-Validation ---
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        X_train_sub = X_train_full[sorted(list(final_active_features))]
        
        y_train_full_np = y_train_full.values if isinstance(y_train_full, pd.Series) else y_train_full

        for train_idx, val_idx in skf.split(X_train_sub, y_train_full_np):
            X_train_fold, X_val_fold = X_train_sub.iloc[train_idx], X_train_sub.iloc[val_idx]
            y_train_fold, y_val_fold = y_train_full_np[train_idx], y_train_full_np[val_idx]
            
            model = copy.deepcopy(ML_MODEL)
            model.set_params(interaction_constraints=filtered_constraints)
            model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], verbose=False)
            
            if METRIC == "auc":
                val_pred = model.predict_proba(X_val_fold)[:, 1]
                score = roc_auc_score(y_val_fold, val_pred)
            else: # accuracy
                val_pred = model.predict(X_val_fold)
                score = accuracy_score(y_val_fold, val_pred)
            cv_scores.append(score)
        val_score = np.mean(cv_scores)
        # --- End CV ---

        num_nodes = len(final_active_features)
        num_edges = sum(len(group) * (len(group) - 1) // 2 for group in filtered_constraints)
        graph_size = num_nodes + num_edges
        trial.set_user_attr("active_nodes", list(final_active_features))
        trial.set_user_attr("filtered_constraints", filtered_constraints)
        return val_score, graph_size

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(directions=["maximize", "minimize"])
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=-1, show_progress_bar=True)
    print(f"\nFound {len(study.best_trials)} best trials on the Pareto front:")
    best_trial = max(study.best_trials, key=lambda t: t.values[0] - t.values[1] * 0.00002)
    best_score, best_size = best_trial.values
    print(f"Selected trial: {best_trial.number}, Score: {best_score:.4f}, Size: {best_size}")

    scores = [t.values[0] for t in study.trials if t.values]
    graph_sizes = [t.values[1] for t in study.trials if t.values]
    pareto_scores = [t.values[0] for t in study.best_trials]
    pareto_graph_sizes = [t.values[1] for t in study.best_trials]
    visualize_pareto_front(scores, graph_sizes, pareto_scores, pareto_graph_sizes, best_score, best_size)

    active_nodes = set(best_trial.user_attrs["active_nodes"])
    final_constraints = best_trial.user_attrs["filtered_constraints"]
    visualize_optimized_graph(active_nodes, final_constraints)
    
    return active_nodes, final_constraints