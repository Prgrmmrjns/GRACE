import optuna
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
from sklearn.base import clone
from params import model_fit

def graph_based_optimization(X_train, y_train, X_val, y_val, initial_node_groups, model, METRIC, n_trials_optuna=2000):
    original_feature_map = {name: list(feats) for name, feats in initial_node_groups.items()}

    def evaluate_graph(current_config):
        X_train_graph = pd.DataFrame(index=X_train.index)
        X_val_graph = pd.DataFrame(index=X_val.index)
        all_input_features_used_in_trial = set()
        active_node_names_eval = []
        for name, feats in current_config.items():
            if feats: 
                all_input_features_used_in_trial.update(feats)
                node_model = clone(model) 
                model_fit(node_model, X_train[feats], y_train, X_val[feats], y_val)
                X_train_graph.loc[:, name] = node_model.predict(X_train[feats]) if METRIC == 'accuracy' else node_model.predict_proba(X_train[feats])[:, 1]
                X_val_graph.loc[:, name] = node_model.predict(X_val[feats]) if METRIC == 'accuracy' else node_model.predict_proba(X_val[feats])[:, 1]
                active_node_names_eval.append(name)
            
        num_unique_input_features = len(all_input_features_used_in_trial)
        if not active_node_names_eval or X_train_graph.shape[1] == 0:
             return (0.0, num_unique_input_features) 
        
        X_train_graph = X_train_graph[active_node_names_eval]
        X_val_graph = X_val_graph[active_node_names_eval]
        ensemble_model = clone(model) 
        model_fit(ensemble_model, X_train_graph, y_train, X_val_graph, y_val)
        val_preds = ensemble_model.predict(X_val_graph) if METRIC == 'accuracy' else ensemble_model.predict_proba(X_val_graph)[:, 1]
        score = accuracy_score(y_val, val_preds) if METRIC == 'accuracy' else roc_auc_score(y_val, val_preds)
        return score, num_unique_input_features

    initial_config_for_eval = initial_node_groups 
    initial_val_score, initial_num_unique_input_features = evaluate_graph(initial_config_for_eval)
    print(f"Initial Configuration - Score ({METRIC.upper()}): {initial_val_score:.4f}, Input Features: {initial_num_unique_input_features}")

    def optuna_objective(trial):
        current_trial_node_groups = {} 
        for group_name, original_features_in_group in original_feature_map.items():
            selected_features_for_group = []
            if original_features_in_group: 
                for feature in original_features_in_group:
                    if trial.suggest_categorical(f"feature_{group_name}_{feature}", [True, False]):
                        selected_features_for_group.append(feature)
            current_trial_node_groups[group_name] = selected_features_for_group
        return evaluate_graph(current_trial_node_groups)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10) 
    study = optuna.create_study(directions=['maximize', 'minimize'], pruner=pruner) 
    study.optimize(optuna_objective, n_trials=n_trials_optuna, show_progress_bar=True, n_jobs=-1)
    
    best_trials = study.best_trials
    for i, trial_info in enumerate(best_trials):
        print(f"  Pareto Solution #{i+1} (Trial {trial_info.number}): Score={trial_info.values[0]:.4f}, Unique Input Features={trial_info.values[1]}")

    # Select best overall trial (highest score, then fewest features)
    best_overall_trial = best_trials[0]
    for trial_info in best_trials[1:]:
        if trial_info.values[0] > best_overall_trial.values[0]:
            best_overall_trial = trial_info
        elif trial_info.values[0] == best_overall_trial.values[0] and trial_info.values[1] < best_overall_trial.values[1]:
            best_overall_trial = trial_info
                
    print(f"\nSelected Best Overall Trial (Highest Score, then Fewest Unique Input Features): Trial {best_overall_trial.number}") # Restored print
    print(f"  Score ({METRIC.upper()}): {best_overall_trial.values[0]:.4f}")
    print(f"  Num Unique Input Features: {best_overall_trial.values[1]}")

    # Reconstruct optimized node groups from the best trial
    optimized_node_groups = {}
    for group_name, original_features_in_group in original_feature_map.items():
        selected_features = []
        if original_features_in_group: 
            for feature in original_features_in_group:
                if best_overall_trial.params.get(f"feature_{group_name}_{feature}", False):
                    selected_features.append(feature)
        # Store the result for the group (might be empty list)
        optimized_node_groups[group_name] = selected_features
    print(f"  Resulting Optimized Node Groups Configuration: {optimized_node_groups}") # Restored print
    return optimized_node_groups