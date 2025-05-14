from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
from sklearn.base import clone
import multiprocessing
import functools
import optuna

def objective(trial, param_names, intermediate_nodes, all_features, X_train, y_train, X_val, y_val, model, METRIC, node_model_predictions_cache):
    """Optuna objective function to evaluate a specific graph configuration."""
    # Get binary parameters from optuna
    current_params_dict = {p_name: trial.suggest_categorical(p_name, [0, 1]) for p_name in param_names}
    
    # Build node groups from active parameters
    current_node_groups = {node: [] for node in intermediate_nodes}
    active_edges_count = 0
    for p_name, is_active in current_params_dict.items():
        if is_active:
            feature, node = p_name.rsplit("_", 1)
            if feature in all_features and node in intermediate_nodes:
                current_node_groups[node].append(feature)
                active_edges_count += 1
    
    # Initialize DataFrames for node outputs
    X_train_graph = pd.DataFrame(index=X_train.index)
    X_val_graph = pd.DataFrame(index=X_val.index)

    # Process each node sequentially
    for node_key, features_list in current_node_groups.items():
        if not features_list:
            continue
            
        if len(features_list) == 1:
            # Single feature just gets passed through
            feature_name = features_list[0]
            X_train_graph[node_key] = X_train[feature_name]
            X_val_graph[node_key] = X_val[feature_name]
        else:
            # Multiple features require a model
            cache_key = tuple(sorted(features_list))
            
            # Use cached predictions if available
            if cache_key in node_model_predictions_cache:
                train_pred, val_pred = node_model_predictions_cache[cache_key]
            else:
                # Train the model sequentially (no parallelization)
                model = clone(model)
                model.fit(X_train[features_list], y_train, eval_set=[(X_val[features_list], y_val)], verbose=False)
    
                train_pred = model.predict(X_train[features_list]) if METRIC == 'accuracy' else model.predict_proba(X_train[features_list])[:, 1]
                val_pred = model.predict(X_val[features_list]) if METRIC == 'accuracy' else model.predict_proba(X_val[features_list])[:, 1]
                # Cache the results
                node_model_predictions_cache[cache_key] = (train_pred, val_pred)
                
            X_train_graph[node_key] = train_pred
            X_val_graph[node_key] = val_pred

    active_edges_count += len(X_train_graph.columns)
    # Train and evaluate the ensemble model
    ensemble_model = clone(model)
    ensemble_model.fit(X_train_graph, y_train, eval_set=[(X_val_graph, y_val)], verbose=False)
    
    val_preds = ensemble_model.predict(X_val_graph) if METRIC == 'accuracy' else ensemble_model.predict_proba(X_val_graph)[:, 1]
    performance = accuracy_score(y_val, val_preds) if METRIC == 'accuracy' else roc_auc_score(y_val, val_preds)
    
    # Return multiple objectives: maximize performance, minimize edges
    return performance, -active_edges_count

def graph_based_optimization(X_train, y_train, X_val, y_val, initial_node_groups, model, METRIC, n_trials_optuna=2000):
    """Optimize the graph structure using Optuna with multi-objective optimization."""
    # Create shared cache for node model predictions
    manager = multiprocessing.Manager()
    node_model_predictions_cache = manager.dict()
    
    # Extract features and intermediate nodes
    all_features = list(X_train.columns)
    intermediate_nodes = list(initial_node_groups.keys())
    
    # Find valid feature-node connections
    valid_connections = set()
    for node, features in initial_node_groups.items():
        for feature in features:
            if feature in all_features:
                valid_connections.add((feature, node))
            
    # Create parameter names for optimization
    param_names = []
    for feature in all_features:
        for node in intermediate_nodes:
            if (feature, node) in valid_connections:
                param_names.append(f"{feature}_{node}")

    # Set Optuna logging level
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Prepare objective function with partial
    objective_fn = functools.partial(
        objective,
        param_names=param_names,
        intermediate_nodes=intermediate_nodes,
        all_features=all_features,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        model=model, METRIC=METRIC,
        node_model_predictions_cache=node_model_predictions_cache
    )
    
    # Create and run multi-objective Optuna study
    study = optuna.create_study(directions=['maximize', 'maximize'])
    study.optimize(objective_fn, n_trials=n_trials_optuna, show_progress_bar=True, n_jobs=-1)
    
    # Get best trials (Pareto front)
    best_trials = study.best_trials
    
    # Define a combined metric to balance performance and edge reduction
    # Formula: score = performance - edge_penalty * (edge_count / max_edges)
    # Higher edge_penalty values favor sparser graphs
    edge_penalty = 0.2  # Adjust this to control performance vs. sparsity tradeoff
    max_edges = max(-t.values[1] for t in best_trials) if best_trials else 1
    
    # Calculate combined scores for each solution
    combined_scores = []
    for trial in best_trials:
        perf = trial.values[0]
        edges = -trial.values[1]
        normalized_edges = edges / max_edges if max_edges > 0 else 0
        combined_score = perf - edge_penalty * normalized_edges
        combined_scores.append((trial, combined_score))
    
    # Sort by combined score and select best solution
    sorted_combined = sorted(combined_scores, key=lambda x: x[1], reverse=True)
    best_trial = sorted_combined[0][0]
    best_params = best_trial.params
    performance_score = best_trial.values[0]
    edge_count = -best_trial.values[1]
    
    # Build optimized node groups from best parameters
    optimized_node_groups = {node: [] for node in intermediate_nodes}
    used_features = set()
    
    for p_name, is_active in best_params.items():
        if is_active:
            feature, node = p_name.rsplit("_", 1)
            if feature in all_features and node in intermediate_nodes:
                optimized_node_groups[node].append(feature)
                used_features.add(feature)
    
    # Remove intermediate nodes without any features
    optimized_node_groups = {node: features for node, features in optimized_node_groups.items() if features}
    
    print(f"Val {METRIC.upper()} Score: {performance_score:.4f}, Features: {len(used_features)}, Edges: {edge_count}")
    
    # For information, show the Pareto front
    print("\nPareto front solutions:")
    for i, (trial, combined_score) in enumerate(sorted_combined[:min(5, len(sorted_combined))]):
        perf = trial.values[0]
        edges = -trial.values[1]
        print(f"  Solution {i+1}: {METRIC.upper()}={perf:.4f}, Edges={edges}, Combined={combined_score:.4f}")
        
    # Explain the choice
    print(f"\nSelected solution with best balance using edge_penalty={edge_penalty}")
    print(f"This favors solutions with fewer edges while maintaining good performance")
        
    return optimized_node_groups