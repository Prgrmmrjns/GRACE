import pandas as pd
from typing import Tuple, List, Any
from sklearn.metrics import roc_auc_score, accuracy_score
import lightgbm as lgb
from graph_utils import KnowledgeGraph, NodeType

def evaluate_model(y_true, y_pred, y_pred_proba=None, metric_name='accuracy'):
    """
    Evaluate model performance using the appropriate metric.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (used for AUC)
        metric_name: Name of the metric to use (default: 'accuracy')
        
    Returns:
        Score based on the specified metric
    """
    
    if metric_name == 'accuracy':
        return accuracy_score(y_true, y_pred)
    elif metric_name == 'auc':
        return roc_auc_score(y_true, y_pred_proba)

def optimize_feature_selection(
    X_train: pd.DataFrame, 
    X_val: pd.DataFrame, 
    y_train: pd.Series, 
    y_val: pd.Series, 
    feature_group: List[str], 
    metric: str = 'accuracy',
    model: lgb.LGBMClassifier = None,
    callbacks: Any = None
) -> Tuple:
    """
    Use forward selection to find effective feature combinations
    """
    best_score = float('-inf')
    best_features = []
    current_features = []
    callbacks = [lgb.early_stopping(stopping_rounds=10, verbose=False)]
    
    # First, evaluate each feature individually and sort by performance
    feature_scores = []
    for feature in feature_group:
        X_train_subset = X_train[[feature]]
        X_val_subset = X_val[[feature]]
        
        model.fit(X_train_subset, y_train, eval_set=[(X_val_subset, y_val)], callbacks=callbacks)
        y_pred = model.predict(X_val_subset)
        y_pred_proba = model.predict_proba(X_val_subset)[:, 1] if metric == 'auc' else None
        score = evaluate_model(y_val, y_pred, y_pred_proba, metric)
        feature_scores.append((feature, score))
    
    # Sort features by individual performance (best first)
    feature_scores.sort(key=lambda x: x[1], reverse=True)
    remaining_features = [f[0] for f in feature_scores]
    
    # Start with the best single feature
    current_features = [remaining_features.pop(0)]
    X_train_subset = X_train[current_features]
    X_val_subset = X_val[current_features]
    
    model.fit(X_train_subset, y_train, eval_set=[(X_val_subset, y_val)], callbacks=callbacks)
    y_pred = model.predict(X_val_subset)
    y_pred_proba = model.predict_proba(X_val_subset)[:, 1] if metric == 'auc' else None
    best_score = evaluate_model(y_val, y_pred, y_pred_proba, metric)
    best_features = current_features.copy()
    
    # Iteratively add features if they improve performance
    improvement = True
    while remaining_features and improvement:
        improvement = False
        best_new_feature = None
        best_new_score = best_score
        
        # Try adding each remaining feature
        for feature in remaining_features[:]:
            candidate_features = current_features + [feature]
            X_train_subset = X_train[candidate_features]
            X_val_subset = X_val[candidate_features]
            
            model.fit(X_train_subset, y_train, eval_set=[(X_val_subset, y_val)], callbacks=callbacks)
            y_pred = model.predict(X_val_subset)
            y_pred_proba = model.predict_proba(X_val_subset)[:, 1] if metric == 'auc' else None
            score = evaluate_model(y_val, y_pred, y_pred_proba, metric)
            
            if score > best_new_score:
                best_new_score = score
                best_new_feature = feature
                improvement = True
        
        # If found a feature that improves performance, add it
        if improvement:
            remaining_features.remove(best_new_feature)
            current_features.append(best_new_feature)
            best_score = best_new_score
            best_features = current_features.copy()
    
    return best_features, best_score

def optimize_intermediate_nodes(
    graph: KnowledgeGraph, 
    removed_nodes: set, 
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    X_val: pd.DataFrame, 
    y_val: pd.Series,
    X_test: pd.DataFrame,
    model: lgb.LGBMClassifier,
    callbacks: Any,
    metric: str,
    verbose: bool = True
) -> Tuple:
    """
    First round optimization: Select best features for each intermediate node.
    
    Returns:
        Tuple containing:
        - X_train_intermediate: DataFrame with intermediate node predictions for training set
        - X_val_intermediate: DataFrame with intermediate node predictions for validation set
        - X_test_intermediate: DataFrame with intermediate node predictions for test set
        - features_to_remove: Set of features that can be removed
        - intermediate_to_selected_features: Dict mapping intermediate nodes to their selected features
        - intermediate_models: Dict mapping intermediate nodes to their trained models
    """
    # Group input nodes by their connected intermediate nodes
    node_groups_by_intermediate = {}
    feature_to_intermediate = {}
    features_to_remove = set()
    
    # Extract node-to-edge relationships
    for node in graph.nodes:
        if node.node_type == NodeType.INPUT and node.name not in removed_nodes:
            for edge in node.edges:
                if edge.target not in removed_nodes:
                    # Check if target is an intermediate node
                    target_node = next((n for n in graph.nodes if n.name == edge.target), None)
                    if target_node and target_node.node_type == NodeType.INTERMEDIATE:
                        if edge.target not in node_groups_by_intermediate:
                            node_groups_by_intermediate[edge.target] = []
                        node_groups_by_intermediate[edge.target].append(node.name)
                        feature_to_intermediate[node.name] = edge.target
    
    # Create enhanced datasets
    X_train_intermediate = pd.DataFrame()
    X_val_intermediate = pd.DataFrame()
    X_test_intermediate = pd.DataFrame()
    
    # Store selected features for each intermediate node for later use
    intermediate_to_selected_features = {}
    intermediate_models = {}
    
    # FIRST ROUND: Find best feature combinations for each intermediate node
    if verbose:
        print("\n--- FIRST ROUND OPTIMIZATION ---")
    for intermediate_node, feature_group in node_groups_by_intermediate.items():
        # Find best feature combination
        selected_features, score = optimize_feature_selection(
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            feature_group=feature_group,
            metric=metric,
            model=model,
            callbacks=callbacks
        )
        
        # Store the selected features for this intermediate node
        intermediate_to_selected_features[intermediate_node] = selected_features
        
        # Identify features to remove
        features_rejected = [f for f in feature_group if f not in selected_features]
        if features_rejected:
            features_to_remove.update(features_rejected)
        
        # Print optimization results
        if verbose:
            print(f"{intermediate_node} - {metric.upper()}: {score:.4f}. Features: {len(feature_group)} total, {len(selected_features)} kept, {len(features_rejected)} removed")
            
        # Train model with selected features and save it
        X_train_selected = X_train[selected_features]
        X_val_selected = X_val[selected_features]
        intermediate_model = lgb.LGBMClassifier(**model.get_params())
        intermediate_model.fit(X_train_selected, y_train, eval_set=[(X_val_selected, y_val)], callbacks=callbacks)
        intermediate_models[intermediate_node] = intermediate_model
        
        # Get predictions for all datasets
        X_train_intermediate[intermediate_node] = intermediate_model.predict_proba(X_train[selected_features])[:, 1]
        X_val_intermediate[intermediate_node] = intermediate_model.predict_proba(X_val[selected_features])[:, 1]
        X_test_intermediate[intermediate_node] = intermediate_model.predict_proba(X_test[selected_features])[:, 1]
    
    return X_train_intermediate, X_val_intermediate, X_test_intermediate, features_to_remove, intermediate_to_selected_features, intermediate_models

def analyze_node_relationships(
    y_train, X_val, y_val,X_train_intermediate, X_val_intermediate, X_test_intermediate,
    intermediate_to_selected_features, intermediate_models, model, callbacks, metric, verbose):
    """
    Second round optimization: Prune weak intermediate nodes only.
    No interaction analysis or feature creation between intermediate nodes.
    """
    if verbose:
        print("\n--- SECOND ROUND: INTERMEDIATE NODE PRUNING ---")
    
    # First pass: Evaluate model with all intermediate nodes
    baseline_model = lgb.LGBMClassifier(**model.get_params())
    baseline_model.fit(X_train_intermediate, y_train, eval_set=[(X_val_intermediate, y_val)], callbacks=callbacks)
    baseline_preds = baseline_model.predict(X_val_intermediate)
    baseline_probs = baseline_model.predict_proba(X_val_intermediate)[:, 1] if metric == 'auc' else None
    baseline_score = evaluate_model(y_val, baseline_preds, baseline_probs, metric)
    
    if verbose:
        print(f"Baseline {metric.upper()} with all intermediate nodes: {baseline_score:.4f}")
    
    # Identify weak nodes for potential removal
    removed_intermediate_nodes = set()
    node_scores = {}
    
    for node_name in list(intermediate_to_selected_features.keys()):
        # Get direct performance of this node
        node_model = intermediate_models[node_name]
        features = intermediate_to_selected_features[node_name]
        
        node_preds = node_model.predict(X_val[features])
        node_probs = node_model.predict_proba(X_val[features])[:, 1] if metric == 'auc' else None
        node_score = evaluate_model(y_val, node_preds, node_probs, metric)
        node_scores[node_name] = node_score
        
        # Test removal if node is weak
        if node_score < 0.6:
            if verbose:
                print(f"\nTesting removal of {node_name} (score: {node_score:.4f})...")
            
            test_X_train = X_train_intermediate.drop(columns=[node_name])
            test_X_val = X_val_intermediate.drop(columns=[node_name])
            
            test_model = lgb.LGBMClassifier(**model.get_params())
            test_model.fit(test_X_train, y_train, eval_set=[(test_X_val, y_val)], callbacks=callbacks)
            
            test_preds = test_model.predict(test_X_val)
            test_probs = test_model.predict_proba(test_X_val)[:, 1] if metric == 'auc' else None
            test_score = evaluate_model(y_val, test_preds, test_probs, metric)
            
            if verbose:
                print(f"  Without {node_name}: {metric.upper()}: {test_score:.4f} (baseline: {baseline_score:.4f})")
            
            if test_score >= baseline_score:
                if verbose:
                    print(f"  ✓ Removing {node_name} - removal maintains performance")
                removed_intermediate_nodes.add(node_name)
            else:
                if verbose:
                    print(f"  ✗ Keeping {node_name} - removal degrades performance")
    
    # Update datasets if any nodes were removed
    if removed_intermediate_nodes:
        if verbose:
            print(f"\nRemoving intermediate nodes: {', '.join(removed_intermediate_nodes)}")
        X_train_intermediate = X_train_intermediate.drop(columns=list(removed_intermediate_nodes))
        X_val_intermediate = X_val_intermediate.drop(columns=list(removed_intermediate_nodes))
        X_test_intermediate = X_test_intermediate.drop(columns=list(removed_intermediate_nodes))
        
        # Clean up mappings
        for node in removed_intermediate_nodes:
            if node in intermediate_to_selected_features:
                del intermediate_to_selected_features[node]
            if node in intermediate_models:
                del intermediate_models[node]
            if node in node_scores:
                del node_scores[node]
    else:
        if verbose:
            print("\nNo intermediate nodes removed - all nodes contribute to model performance")
    
    # Calculate node contributions to final model directly
    final_model = lgb.LGBMClassifier(**model.get_params())
    final_model.fit(X_train_intermediate, y_train, eval_set=[(X_val_intermediate, y_val)], callbacks=callbacks)
    
    return X_train_intermediate, X_val_intermediate, X_test_intermediate, removed_intermediate_nodes

def evaluate_final_model(
    X_train_intermediate: pd.DataFrame,
    X_val_intermediate: pd.DataFrame,
    X_test_intermediate: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    model: lgb.LGBMClassifier,
    callbacks: Any,
    metric: str
) -> Tuple:
    """Evaluate the best model on the test set and save results."""
    model.fit(X_train_intermediate, y_train, eval_set=[(X_val_intermediate, y_val)], callbacks=callbacks)
    test_preds = model.predict(X_test_intermediate)
    test_preds_proba = model.predict_proba(X_test_intermediate)[:, 1] if metric == 'auc' else None
    test_score = evaluate_model(y_test, test_preds, test_preds_proba, metric)
    
    return test_score

def evaluate_final_model(
    X_train_intermediate: pd.DataFrame,
    X_val_intermediate: pd.DataFrame,
    X_test_intermediate: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    model: lgb.LGBMClassifier,
    callbacks: Any,
    metric: str
) -> Tuple:
    """Evaluate the best model on the test set and save results."""
    model.fit(X_train_intermediate, y_train, eval_set=[(X_val_intermediate, y_val)], callbacks=callbacks)
    test_preds = model.predict(X_test_intermediate)
    test_preds_proba = model.predict_proba(X_test_intermediate)[:, 1] if metric == 'auc' else None
    test_score = evaluate_model(y_test, test_preds, test_preds_proba, metric)
    return test_score