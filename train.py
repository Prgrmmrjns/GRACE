from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
import copy
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from joblib import Parallel, delayed
import os
import networkx as nx
import numpy as np

# --- Optimization Hyperparameters ---
# Penalty for graph complexity (relative to initial unique nodes)
RELATIVE_PENALTY_STRENGTH = 0.01
# Probabilities for hyperopt search space generation
PROB_KEEP_EXISTING_FEATURE = 0.5 
PROB_ADD_NON_EXISTING_FEATURE = 0.05
MAX_HYPEROPT_EVALS = 300 # Increased as requested
# ----------------------------------

# --- LightGBM Hyperparameter Search Space (used for per-model definitions) ---
LGBM_MAX_DEPTH_CHOICES = [3, 4, 5, 6, 7, 8, 9, 10]
LGBM_REG_LAMBDA_LOW = 1e-3
LGBM_REG_LAMBDA_HIGH = 10.0
LGBM_REG_ALPHA_LOW = 1e-3
LGBM_REG_ALPHA_HIGH = 10.0
LGBM_MIN_CHILD_WEIGHT_CHOICES = [1, 5, 10, 15, 20]
# ------------------------------------------

def _extract_lgbm_params_from_model(model_instance):
    return {
        'max_depth': model_instance.get_params().get('max_depth', 6), # Default from typical LGBM
        'reg_lambda': model_instance.get_params().get('reg_lambda', 0.0),
        'reg_alpha': model_instance.get_params().get('reg_alpha', 0.0),
        'min_child_weight': model_instance.get_params().get('min_child_weight', 1e-3)
    }

def graph_based_optimization(X_train, y_train, X_val, y_val, X_test, y_test,
                             initial_node_groups, base_model, METRIC, callbacks,
                             raw_graph_base, dataset_name_for_saving):

    all_features = list(X_train.columns)
    
    prob_remove_existing = 1.0 - PROB_KEEP_EXISTING_FEATURE
    prob_no_add_non_existing = 1.0 - PROB_ADD_NON_EXISTING_FEATURE

    def _calculate_graph_stats(node_groups_config):
        total_edges = sum(len(features) for features in node_groups_config.values() if features)
        unique_input_nodes = set()
        for features in node_groups_config.values():
            if features: unique_input_nodes.update(features)
        return total_edges, len(unique_input_nodes)

    def _report_edge_differences(initial_groups, final_groups, final_config_source_name):
        print(f"\n--- Edge Changes based on {final_config_source_name} Configuration ---")
        initial_edges_set = set()
        for node, feats in initial_groups.items():
            if feats: initial_edges_set.update([(feat, node) for feat in feats])
        final_edges_set = set()
        if final_groups:
            for node, feats in final_groups.items():
                if feats: final_edges_set.update([(feat, node) for feat in feats])
        added_edges = final_edges_set - initial_edges_set
        removed_edges = initial_edges_set - final_edges_set
        if added_edges: print(f"Added Edges ({len(added_edges)}): {sorted(list(added_edges))}")
        else: print("Added Edges: None")
        if removed_edges: print(f"Removed Edges ({len(removed_edges)}): {sorted(list(removed_edges))}")
        else: print("Removed Edges: None")
        print("----------------------------------------------------")

    # _train_single_node_model_job now takes specific hyperparameters for the node model
    def _train_single_node_model_job(node_name_job, features_job, X_train_df_job, y_train_series_job, X_val_df_job, y_val_series_job, 
                                     base_model_template_job, node_hps_job, metric_job, callbacks_job):
        if not features_job or X_train_df_job[features_job].empty: return node_name_job, None, None, False
        node_model_local = copy.deepcopy(base_model_template_job)
        node_model_local.set_params(**node_hps_job)
        node_model_local.fit(X_train_df_job[features_job], y_train_series_job, eval_set=[(X_val_df_job[features_job], y_val_series_job)], callbacks=callbacks_job)
        train_pred = node_model_local.predict(X_train_df_job[features_job]) if metric_job == 'accuracy' else node_model_local.predict_proba(X_train_df_job[features_job])[:, 1]
        val_pred = node_model_local.predict(X_val_df_job[features_job]) if metric_job == 'accuracy' else node_model_local.predict_proba(X_val_df_job[features_job])[:, 1]
        return node_name_job, train_pred, val_pred, True

    # _train_and_predict_for_final_graphs also takes specific HPs for the node model
    def _train_and_predict_for_final_graphs(node_name_job, features_job, X_train_df_job, y_train_series_job, X_val_df_job, y_val_series_job, X_test_df_job, 
                                            base_model_template_job, node_hps_job, metric_job, callbacks_job):
        if not features_job or X_train_df_job[features_job].empty or X_test_df_job[features_job].empty: return node_name_job, None, None, None, False
        node_model_local = copy.deepcopy(base_model_template_job)
        node_model_local.set_params(**node_hps_job)
        node_model_local.fit(X_train_df_job[features_job], y_train_series_job, eval_set=[(X_val_df_job[features_job], y_val_series_job)], callbacks=callbacks_job)
        train_pred = node_model_local.predict(X_train_df_job[features_job]) if metric_job == 'accuracy' else node_model_local.predict_proba(X_train_df_job[features_job])[:, 1]
        val_pred = node_model_local.predict(X_val_df_job[features_job]) if metric_job == 'accuracy' else node_model_local.predict_proba(X_val_df_job[features_job])[:, 1]
        test_pred = node_model_local.predict(X_test_df_job[features_job]) if metric_job == 'accuracy' else node_model_local.predict_proba(X_test_df_job[features_job])[:, 1]
        return node_name_job, train_pred, val_pred, test_pred, True

    # New core evaluation function
    def _perform_full_evaluation(current_node_groups_config, per_node_hp_map, final_ensemble_hp_config, 
                                 model_template, X_tr, y_tr, X_v, y_v, metric_eval, cb_eval):
        if not current_node_groups_config or all(not features for features in current_node_groups_config.values()):
            return 0.0
        
        X_train_graph = pd.DataFrame(index=X_tr.index)
        X_val_graph = pd.DataFrame(index=X_v.index)
        active_node_names = []
        
        job_args_eval = []
        for name, feats in current_node_groups_config.items():
            if feats: # only if node has features
                node_hps = per_node_hp_map.get(name, _extract_lgbm_params_from_model(model_template)) # Fallback to defaults
                job_args_eval.append((name, feats, X_tr, y_tr, X_v, y_v, model_template, node_hps, metric_eval, cb_eval))

        if not job_args_eval: return 0.0

        results = Parallel(n_jobs=-1)(delayed(_train_single_node_model_job)(*args) for args in job_args_eval)
        
        for name_res, train_res, val_res, success in results:
            if success and train_res is not None:
                X_train_graph[name_res] = train_res
                X_val_graph[name_res] = val_res
                active_node_names.append(name_res)
        
        if X_train_graph.empty: return 0.0
        
        X_train_graph = X_train_graph[active_node_names]
        X_val_graph = X_val_graph[active_node_names]
        
        final_model_eval = copy.deepcopy(model_template)
        final_model_eval.set_params(**final_ensemble_hp_config)
        final_model_eval.fit(X_train_graph, y_tr, eval_set=[(X_val_graph, y_v)], callbacks=cb_eval)
        
        val_preds = final_model_eval.predict(X_val_graph) if metric_eval == 'accuracy' else final_model_eval.predict_proba(X_val_graph)[:, 1]
        score = accuracy_score(y_v, val_preds) if metric_eval == 'accuracy' else roc_auc_score(y_v, val_preds)
        return score

    # --- Initial State Evaluation ---
    initial_edges, initial_unique_nodes = _calculate_graph_stats(initial_node_groups)
    print(f"Initial Graph: Edges={initial_edges}, Unique Nodes={initial_unique_nodes}")
    
    default_lgbm_hps = _extract_lgbm_params_from_model(base_model)
    initial_per_node_hps = {node_name: default_lgbm_hps for node_name in initial_node_groups}
    initial_final_hps = default_lgbm_hps
    
    initial_prediction_score = _perform_full_evaluation(
        initial_node_groups, initial_per_node_hps, initial_final_hps,
        base_model, X_train, y_train, X_val, y_val, METRIC, callbacks
    )
    initial_penalty = RELATIVE_PENALTY_STRENGTH if initial_unique_nodes > 0 else 0
    initial_combined_score = initial_prediction_score - initial_penalty
    print(f"Initial Pred Score (Val): {initial_prediction_score:.4f}")
    print(f"Initial Combined Score (Val): {initial_combined_score:.4f} (Penalty Str: {RELATIVE_PENALTY_STRENGTH}, Default HPs)")

    # --- Search Space Definition ---
    search_space = {}
    # Feature selection part
    for node_name, current_features_in_node in initial_node_groups.items():
        search_space[node_name] = [
            hp.pchoice(f'{node_name}_{feat}_decision', [
                (PROB_KEEP_EXISTING_FEATURE, feat), (prob_remove_existing, None)
            ]) if feat in current_features_in_node else hp.pchoice(f'{node_name}_{feat}_decision', [
                (PROB_ADD_NON_EXISTING_FEATURE, feat), (prob_no_add_non_existing, None)
            ]) for feat in all_features
        ]
    # Hyperparameters for each node model
    for node_name in initial_node_groups.keys():
        search_space[f'{node_name}_lgbm_max_depth'] = hp.choice(f'{node_name}_lgbm_max_depth', LGBM_MAX_DEPTH_CHOICES)
        search_space[f'{node_name}_lgbm_reg_lambda'] = hp.loguniform(f'{node_name}_lgbm_reg_lambda', np.log(LGBM_REG_LAMBDA_LOW), np.log(LGBM_REG_LAMBDA_HIGH))
        search_space[f'{node_name}_lgbm_reg_alpha'] = hp.loguniform(f'{node_name}_lgbm_reg_alpha', np.log(LGBM_REG_ALPHA_LOW), np.log(LGBM_REG_ALPHA_HIGH))
        search_space[f'{node_name}_lgbm_min_child_weight'] = hp.choice(f'{node_name}_lgbm_min_child_weight', LGBM_MIN_CHILD_WEIGHT_CHOICES)
    
    # Hyperparameters for the final ensemble model
    search_space['final_lgbm_max_depth'] = hp.choice('final_lgbm_max_depth', LGBM_MAX_DEPTH_CHOICES)
    search_space['final_lgbm_reg_lambda'] = hp.loguniform('final_lgbm_reg_lambda', np.log(LGBM_REG_LAMBDA_LOW), np.log(LGBM_REG_LAMBDA_HIGH))
    search_space['final_lgbm_reg_alpha'] = hp.loguniform('final_lgbm_reg_alpha', np.log(LGBM_REG_ALPHA_LOW), np.log(LGBM_REG_ALPHA_HIGH))
    search_space['final_lgbm_min_child_weight'] = hp.choice('final_lgbm_min_child_weight', LGBM_MIN_CHILD_WEIGHT_CHOICES)

    # --- Hyperopt Objective Function ---
    def hyperopt_objective(params):
        current_node_groups, current_unique_input_nodes_set, current_total_edges_count = {}, set(), 0
        
        # Parse feature selection params
        for node_name_key in initial_node_groups.keys(): # Iterate actual node names for safety
            if node_name_key in params:
                feature_decisions = params[node_name_key]
                selected_features = [f_name for f_name in feature_decisions if f_name is not None]
                if selected_features:
                    current_node_groups[node_name_key] = list(set(selected_features))
                    current_unique_input_nodes_set.update(selected_features)
                    current_total_edges_count += len(selected_features)
        
        num_unique_nodes_for_config = len(current_unique_input_nodes_set)

        # Parse per-node hyperparameters
        current_per_node_hps = {}
        for node_name_key in current_node_groups.keys(): # Only for active nodes
            current_per_node_hps[node_name_key] = {
                'max_depth': params[f'{node_name_key}_lgbm_max_depth'],
                'reg_lambda': params[f'{node_name_key}_lgbm_reg_lambda'],
                'reg_alpha': params[f'{node_name_key}_lgbm_reg_alpha'],
                'min_child_weight': params[f'{node_name_key}_lgbm_min_child_weight']
            }
            
        # Parse final ensemble model hyperparameters
        current_final_hps = {
            'max_depth': params['final_lgbm_max_depth'],
            'reg_lambda': params['final_lgbm_reg_lambda'],
            'reg_alpha': params['final_lgbm_reg_alpha'],
            'min_child_weight': params['final_lgbm_min_child_weight']
        }

        if not current_node_groups or current_total_edges_count == 0:
            pen_empty = RELATIVE_PENALTY_STRENGTH * (len(all_features) / initial_unique_nodes if initial_unique_nodes > 0 else len(all_features))
            return {'loss': float('inf') + pen_empty, 'status': STATUS_OK, 'node_groups': current_node_groups,
                    'prediction_score': -float('inf'), 'num_unique_nodes': num_unique_nodes_for_config,
                    'chosen_hyperparameters': params}

        prediction_score = _perform_full_evaluation(
            current_node_groups, current_per_node_hps, current_final_hps,
            base_model, X_train, y_train, X_val, y_val, METRIC, callbacks
        )
        
        penalty = RELATIVE_PENALTY_STRENGTH * (num_unique_nodes_for_config / initial_unique_nodes) if initial_unique_nodes > 0 else (RELATIVE_PENALTY_STRENGTH * num_unique_nodes_for_config if num_unique_nodes_for_config > 0 else 0)
        combined_loss = -prediction_score + penalty
        
        return {'loss': combined_loss, 'status': STATUS_OK, 'node_groups': current_node_groups,
                'prediction_score': prediction_score, 'num_unique_nodes': num_unique_nodes_for_config,
                'chosen_hyperparameters': params}

    # --- Run Optimization ---
    trials = Trials()
    fmin(fn=hyperopt_objective, space=search_space, algo=tpe.suggest, max_evals=MAX_HYPEROPT_EVALS, trials=trials)
    
    # --- Process Results ---
    best_trial_result = trials.best_trial['result']
    best_node_groups_config = best_trial_result['node_groups']
    best_pred_score_val = best_trial_result['prediction_score']
    best_num_unique_nodes = best_trial_result['num_unique_nodes']
    
    best_trial_penalty = RELATIVE_PENALTY_STRENGTH * (best_num_unique_nodes / initial_unique_nodes) if initial_unique_nodes > 0 else (RELATIVE_PENALTY_STRENGTH * best_num_unique_nodes if best_num_unique_nodes > 0 else 0)
    best_combined_score_val = best_pred_score_val - best_trial_penalty
    
    all_best_hps_from_trial = best_trial_result['chosen_hyperparameters']

    # Extract and structure the best HPs for reporting and final use
    best_optimized_node_hps = {}
    if best_node_groups_config: # Ensure there are nodes
        for node_name_key in best_node_groups_config.keys():
             # Check if HPs for this node exist in the trial params (might not if node was inactive in initial_node_groups)
            if f'{node_name_key}_lgbm_max_depth' in all_best_hps_from_trial:
                best_optimized_node_hps[node_name_key] = {
                    'max_depth': all_best_hps_from_trial[f'{node_name_key}_lgbm_max_depth'],
                    'reg_lambda': all_best_hps_from_trial[f'{node_name_key}_lgbm_reg_lambda'],
                    'reg_alpha': all_best_hps_from_trial[f'{node_name_key}_lgbm_reg_alpha'],
                    'min_child_weight': all_best_hps_from_trial[f'{node_name_key}_lgbm_min_child_weight']
                }
            else: # Fallback for nodes that might have been in best_node_groups_config but not in initial_node_groups for HP definition
                best_optimized_node_hps[node_name_key] = default_lgbm_hps


    best_optimized_final_hps = {
        'max_depth': all_best_hps_from_trial['final_lgbm_max_depth'],
        'reg_lambda': all_best_hps_from_trial['final_lgbm_reg_lambda'],
        'reg_alpha': all_best_hps_from_trial['final_lgbm_reg_alpha'],
        'min_child_weight': all_best_hps_from_trial['final_lgbm_min_child_weight']
    }

    print(f"\n--- Best Hyperparameters Found by Hyperopt ---")
    for node_name_key, hps in best_optimized_node_hps.items():
        print(f"Node '{node_name_key}': {hps}")
    print(f"Final Ensemble Model: {best_optimized_final_hps}")
    print("-----------------------------------------------")

    final_node_groups_to_use = initial_node_groups
    final_node_hps_to_use = initial_per_node_hps
    final_ensemble_hps_to_use = initial_final_hps
    final_pred_score_val_to_report = initial_prediction_score
    final_combined_score_val_to_report = initial_combined_score
    final_config_source = "Initial Config (Default HPs)"

    if best_node_groups_config and best_combined_score_val > initial_combined_score : # Ensure best config is not empty
        print(f"Optimized config is better (Combined Score: {best_combined_score_val:.4f} vs Initial: {initial_combined_score:.4f}). Using optimized settings.")
        final_node_groups_to_use = best_node_groups_config
        final_node_hps_to_use = best_optimized_node_hps
        final_ensemble_hps_to_use = best_optimized_final_hps
        final_pred_score_val_to_report = best_pred_score_val
        final_combined_score_val_to_report = best_combined_score_val
        final_config_source = "Optimized Best Trial (Individual HPs)"
    else:
        print(f"Optimized config (Combined Score: {best_combined_score_val:.4f}) not better than initial (Combined Score: {initial_combined_score:.4f}) or optimized graph is empty. Reverting to initial settings.")

    _report_edge_differences(initial_node_groups, final_node_groups_to_use, final_config_source)
    
    # Save the optimized graph structure (using final_node_groups_to_use)
    optimized_G = nx.DiGraph()
    for node, attrs in raw_graph_base.nodes(data=True):
        optimized_G.add_node(node, **attrs)
    if final_node_groups_to_use:
        for inter_node, features in final_node_groups_to_use.items():
            if features:
                for feature_node in features:
                    if optimized_G.has_node(feature_node) and optimized_G.has_node(inter_node):
                         optimized_G.add_edge(feature_node, inter_node, relationship="evidence_optimized")
    for u, v, attrs in raw_graph_base.edges(data=True):
        if raw_graph_base.nodes[u].get('entity_type') == 'INTERMEDIATE_NODE' and \
           raw_graph_base.nodes[v].get('entity_type') == 'TARGET_NODE':
            if optimized_G.has_node(u) and optimized_G.has_node(v): # Ensure nodes for final connections exist
                if inter_node in final_node_groups_to_use and final_node_groups_to_use[inter_node]: # Only add if inter_node is active
                     optimized_G.add_edge(u, v, **attrs)

    nodes_before_isolate_removal = optimized_G.number_of_nodes()
    isolates = list(nx.isolates(optimized_G))
    optimized_G.remove_nodes_from(isolates)
    if isolates:
        print(f"Removed {len(isolates)} isolated nodes (out of {nodes_before_isolate_removal}) from the optimized graph before saving: {sorted(isolates)}")
                
    kg_dir = "kg"
    os.makedirs(kg_dir, exist_ok=True)
    optimized_kg_path = os.path.join(kg_dir, f"{dataset_name_for_saving}_optimized.graphml")
    nx.write_graphml(optimized_G, optimized_kg_path)
    print(f"Optimized graph saved to: {optimized_kg_path}")

    final_test_edges, final_test_unique_nodes = _calculate_graph_stats(final_node_groups_to_use)
    print(f"\nUsing for Final Test ({final_config_source}): Edges={final_test_edges}, Unique Nodes={final_test_unique_nodes}")
    print(f"Associated Pred Score (Val): {final_pred_score_val_to_report:.4f}")
    print(f"Associated Combined Score (Val): {final_combined_score_val_to_report:.4f}")
    print("Building final model with chosen HPs for test evaluation...")

    X_train_graph_best, X_val_graph_best, X_test_graph_best = pd.DataFrame(index=X_train.index), pd.DataFrame(index=X_val.index), pd.DataFrame(index=X_test.index)
    final_active_node_names = []
    
    final_job_args = []
    if final_node_groups_to_use: # Check if there are any node groups to process
        for name, feats in final_node_groups_to_use.items():
            if feats: # Check if node has features assigned
                # Get HPs for this node, fallback to default if somehow missing (should not happen if logic is correct)
                node_hps = final_node_hps_to_use.get(name, default_lgbm_hps) 
                final_job_args.append((name, feats, X_train, y_train, X_val, y_val, X_test, base_model, node_hps, METRIC, callbacks))
    
    if final_job_args: # Only run Parallel if there are jobs
        final_graph_results = Parallel(n_jobs=-1)(delayed(_train_and_predict_for_final_graphs)(*args) for args in final_job_args)
        for name_res, train_res, val_res, test_res, success in final_graph_results:
            if success and train_res is not None: # Ensure predictions were successful
                X_train_graph_best[name_res], X_val_graph_best[name_res], X_test_graph_best[name_res] = train_res, val_res, test_res
                final_active_node_names.append(name_res)

    X_train_graph_best = X_train_graph_best[final_active_node_names]
    X_val_graph_best = X_val_graph_best[final_active_node_names]
    X_test_graph_best = X_test_graph_best[final_active_node_names]
    
    final_model_on_best_config = copy.deepcopy(base_model)
    final_model_on_best_config.set_params(**final_ensemble_hps_to_use)
    final_model_on_best_config.fit(X_train_graph_best, y_train, eval_set=[(X_val_graph_best, y_val)], callbacks=callbacks)
    
    test_preds_probas = final_model_on_best_config.predict_proba(X_test_graph_best)[:, 1]
    test_score_val = roc_auc_score(y_test, test_preds_probas) if METRIC != 'accuracy' else accuracy_score(y_test, final_model_on_best_config.predict(X_test_graph_best))
    
    print(f"Final Test Pred Score ({METRIC.upper()}): {test_score_val:.4f}")
    return final_node_groups_to_use

    
