from visualizations import visualize_optimized_graph, visualize_pareto_front
import copy
from utils import (get_effective_edges, 
                   get_constraints_from_graph, create_interaction_constraints)
import shap
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import optuna
import networkx as nx
from params import ML_MODEL, CALLBACKS, METRIC

def get_shap_contributions(X_train, y_train, X_val, y_val, edges_for_model):
    """Calculates SHAP values for a model constrained by a specific set of edges."""
    feature_names = X_train.columns.tolist()
    feature_to_idx = {name: i for i, name in enumerate(feature_names)}

    model = copy.deepcopy(ML_MODEL)
    constraints, _, _ = get_constraints_from_graph(feature_names, list(edges_for_model), X_train)
    model.set_params(interaction_constraints=constraints)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=CALLBACKS)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
        
    shap_interaction_values = explainer.shap_interaction_values(X_val)
    if isinstance(shap_interaction_values, list):
        shap_interaction_values = shap_interaction_values[1]

    feature_contributions = []
    for i, f in enumerate(feature_names):
        feature_contributions.append((f, np.abs(shap_values[:, i]).mean()))
    feature_contributions.sort(key=lambda x: x[1])

    interaction_pairs = []
    for feat1, feat2 in edges_for_model:
        idx1 = feature_to_idx.get(feat1)
        idx2 = feature_to_idx.get(feat2)
        if idx1 is not None and idx2 is not None:
            interaction_strength = np.abs(shap_interaction_values[:, idx1, idx2]).mean()
            interaction_pairs.append((feat1, feat2, interaction_strength))
    interaction_pairs.sort(key=lambda x: x[2])
    return feature_contributions, interaction_pairs

def optimize_graph(X_train, y_train, X_val, y_val, mechanism_to_features):
    initial_edges = get_effective_edges(mechanism_to_features)
    feature_contributions, interaction_strengths = get_shap_contributions(X_train, y_train, X_val, y_val, initial_edges)
    min_node_imp = min(c[1] for c in feature_contributions) if feature_contributions else 0
    max_node_imp = max(c[1] for c in feature_contributions) if feature_contributions else 0
    min_edge_int = min(s[2] for s in interaction_strengths) if interaction_strengths else 0
    max_edge_int = max(s[2] for s in interaction_strengths) if interaction_strengths else 0

    # --- Pre-calculate Graph Metrics ---
    node_strengths = {}
    for node in X_train.columns:
        node_strengths[node] = 0.0
    for f1, f2, strength in interaction_strengths:
        node_strengths[f1] += strength
        node_strengths[f2] += strength
    min_node_strength = min(node_strengths.values()) if node_strengths else 0
    max_node_strength = max(node_strengths.values()) if node_strengths else 0

    community_edge_threshold = np.percentile([s[2] for s in interaction_strengths if s[2] > 0], 50) if any(s[2] > 0 for s in interaction_strengths) else 0
    G = nx.Graph()
    for f1, f2, strength in interaction_strengths:
        if strength >= community_edge_threshold:
            G.add_edge(f1, f2, weight=strength)
    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)
    
    communities = nx.community.louvain_communities(G, seed=42) if len(G.nodes()) > 0 else []
    communities = [c for c in communities if len(c) > 1]
    node_to_community = {node: i for i, comm in enumerate(communities) for node in comm}
    for node in isolated: node_to_community[node] = -1
        
    betweenness = nx.betweenness_centrality(G, weight='weight') if len(G.nodes()) > 0 else {}
    for node in X_train.columns:
        if node not in betweenness: betweenness[node] = 0.0
    min_betweenness = min(betweenness.values())
    max_betweenness = max(betweenness.values())
    print(f"Detected {len(communities)} meaningful communities.")
    
    # --- Optuna Objective ---
    def objective(trial):
        edge_thresh = trial.suggest_float("edge_interaction_threshold", min_edge_int, max_edge_int)
        max_edges = trial.suggest_int("max_edges_per_node", 2, 15)
        strength_thresh = trial.suggest_float("node_strength_threshold", min_node_strength, max_node_strength)
        community_thresh = trial.suggest_float("community_threshold", 0.0, 1.0)
        betweenness_thresh = trial.suggest_float("betweenness_threshold", min_betweenness, max_betweenness)
        node_imp_thresh = trial.suggest_float("node_importance_threshold", min_node_imp, max_node_imp)

        active_nodes = {
            feat for feat, imp in feature_contributions 
            if imp >= node_imp_thresh
            and node_strengths.get(feat, 0) >= strength_thresh 
            and betweenness.get(feat, 0) >= betweenness_thresh
        }
        
        updated_mech_to_features = {}
        for mech, features in mechanism_to_features.items():
            updated_features = [f for f in features if f in active_nodes]
            if len(updated_features) > 1: updated_mech_to_features[mech] = updated_features
        all_possible_edges = get_effective_edges(updated_mech_to_features)
        
        candidate_edges = []
        for f1, f2, strength in interaction_strengths:
            edge = tuple(sorted((f1, f2)))
            if (strength >= edge_thresh and edge in all_possible_edges):
                comm1 = node_to_community.get(f1, -1); comm2 = node_to_community.get(f2, -1)
                same_community = (comm1 == comm2) and (comm1 != -1)
                if same_community or np.random.random() < community_thresh:
                    candidate_edges.append((f1, f2, strength))
        
        node_edge_count = {node: 0 for node in active_nodes}
        strong_edges = set()
        candidate_edges.sort(key=lambda x: x[2], reverse=True)
        for f1, f2, strength in candidate_edges:
            if f1 in node_edge_count and f2 in node_edge_count:
                if node_edge_count[f1] < max_edges and node_edge_count[f2] < max_edges:
                    strong_edges.add(tuple(sorted((f1, f2)))); node_edge_count[f1] += 1; node_edge_count[f2] += 1
        
        final_mech_features = {}
        for mech, features in updated_mech_to_features.items():
            features_with_edges = {f for edge in strong_edges for f in edge if f in features}
            final_features = [f for f in features if f in features_with_edges]
            if len(final_features) > 1: final_mech_features[mech] = final_features
        
        final_nodes = {f for features in final_mech_features.values() for f in features}
        graph_size = len(final_nodes) + len(strong_edges)

        if not final_mech_features or not final_nodes:
            trial.set_user_attr("final_nodes", [])
            trial.set_user_attr("strong_edges", set())
            trial.set_user_attr("final_mech_features", {})
            return 0.0, len(X_train.columns) + len(initial_edges)
        
        constraints = create_interaction_constraints(final_mech_features, list(final_nodes))
        
        model = copy.deepcopy(ML_MODEL)
        model.set_params(interaction_constraints=constraints)
        X_train_sub = X_train[list(final_nodes)]; X_val_sub = X_val[list(final_nodes)]
        model.fit(X_train_sub, y_train, eval_set=[(X_val_sub, y_val)], callbacks=CALLBACKS)
        val_pred = model.predict_proba(X_val_sub)[:, 1] if METRIC != 'accuracy' else model.predict(X_val_sub)
        val_score = roc_auc_score(y_val, val_pred) if METRIC != 'accuracy' else accuracy_score(y_val, val_pred)
        trial.set_user_attr("final_nodes", list(final_nodes))
        trial.set_user_attr("strong_edges", strong_edges)
        trial.set_user_attr("final_mech_features", final_mech_features)
        return val_score, graph_size

    # --- Run Optuna Study ---
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    study = optuna.create_study(directions=["maximize", "minimize"], sampler=optuna.samplers.GPSampler())
    study.optimize(objective, n_trials=100, n_jobs=-1, show_progress_bar=True)

    # Visualize Pareto front
    all_trials = [t for t in study.trials if t.values is not None]
    pareto_trials = study.best_trials
    visualize_pareto_front(
        scores=[t.values[0] for t in all_trials],
        graph_sizes=[t.values[1] for t in all_trials],
        pareto_scores=[t.values[0] for t in pareto_trials],
        pareto_graph_sizes=[t.values[1] for t in pareto_trials],
    )
        
    best_trial = max(study.best_trials, key=lambda t: (t.values[0], -t.values[1]))
    best_val_score, graph_size = best_trial.values
    print(f"\nBest trial score: {best_val_score:.4f} with Graph Size {int(graph_size)}.")
    
    # --- Finalization ---
    final_nodes = best_trial.user_attrs["final_nodes"]
    final_edges = best_trial.user_attrs["strong_edges"]
    final_mechanisms = best_trial.user_attrs["final_mech_features"]
    visualize_optimized_graph(final_nodes, final_edges)
    return final_mechanisms