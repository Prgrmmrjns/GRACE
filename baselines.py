import time
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
import warnings
import os
import networkx as nx
from params import LIGHTGBM_PARAMS, TEST_SIZE, VAL_SIZE, ML_MODEL, METRIC, EARLY_STOPPING_CALLBACK, MODEL_TYPE, fit_model
from create_kg import run_kg_workflows
from graph_utils import nx_to_feature_interactions
from db import setup_lancedb_knowledge_base
from grace_shap import build_constraints_from_interactions, calculate_shap_values, shap_based_selection, optimize_thresholds, set_interaction_constraints

# Suppress all warnings
warnings.filterwarnings("ignore")

def evaluate_model(y_true, y_pred, y_pred_proba=None, metric_name='accuracy'):
    if metric_name == 'accuracy':
        return accuracy_score(y_true, y_pred)
    elif metric_name == 'auc':
        return roc_auc_score(y_true, y_pred_proba)

def run_grace_approach(X_train, X_val, X_test, y_train, y_val, y_test, feature_interactions, metric, unique_classes):
    start_time = time.time()
    
    constraints = build_constraints_from_interactions(feature_interactions, X_train.columns.tolist())
    model = clone(ML_MODEL)
    model = set_interaction_constraints(model, constraints, X_train.columns.tolist())
    model = fit_model(model, X_train, y_train, X_val, y_val, EARLY_STOPPING_CALLBACK)
    
    shap_values, _ = calculate_shap_values(model, X_train, X_val)
    best_params, _ = optimize_thresholds(
        X_train, X_val, y_train, y_val, shap_values, feature_interactions,
        X_train.columns.tolist(), ML_MODEL, metric, EARLY_STOPPING_CALLBACK
    )
    
    selected_features, _, _ = shap_based_selection(
        shap_values, feature_interactions, X_train.columns.tolist(), 
        best_params['min_shap_threshold'], best_params['min_interaction_threshold'], "temp"
    )
    
    X_train_filtered = X_train[selected_features]
    X_val_filtered = X_val[selected_features]
    X_test_filtered = X_test[selected_features]
    
    final_constraints = build_constraints_from_interactions(feature_interactions, selected_features)
    model = clone(ML_MODEL)
    model = set_interaction_constraints(model, final_constraints, selected_features)
    model = fit_model(model, X_train_filtered, y_train, X_val_filtered, y_val, EARLY_STOPPING_CALLBACK)
    
    y_pred = model.predict(X_test_filtered)
    y_pred_proba = model.predict_proba(X_test_filtered)[:, 1] if unique_classes == 2 else model.predict_proba(X_test_filtered)
    
    end_time = time.time()
    runtime = end_time - start_time
    
    if metric == 'auc':
        if unique_classes == 2:
            score = evaluate_model(y_test, y_pred, y_pred_proba, metric)
        else:
            score = roc_auc_score(pd.get_dummies(y_test), y_pred_proba, average='macro')
    else:
        score = evaluate_model(y_test, y_pred, None, metric)
    
    return score, runtime, len(selected_features)

def run_graph_no_constraints(X_train, X_val, X_test, y_train, y_val, y_test, feature_interactions, metric, unique_classes):
    start_time = time.time()
    
    model = clone(ML_MODEL)
    model = fit_model(model, X_train, y_train, X_val, y_val, EARLY_STOPPING_CALLBACK)
    
    shap_values, _ = calculate_shap_values(model, X_train, X_val)
    best_params, _ = optimize_thresholds(
        X_train, X_val, y_train, y_val, shap_values, feature_interactions,
        X_train.columns.tolist(), ML_MODEL, metric, EARLY_STOPPING_CALLBACK
    )
    
    selected_features, _, _ = shap_based_selection(
        shap_values, feature_interactions, X_train.columns.tolist(), 
        best_params['min_shap_threshold'], best_params['min_interaction_threshold'], "temp"
    )
    
    X_train_filtered = X_train[selected_features]
    X_val_filtered = X_val[selected_features]
    X_test_filtered = X_test[selected_features]
    
    model = clone(ML_MODEL)
    model = fit_model(model, X_train_filtered, y_train, X_val_filtered, y_val, EARLY_STOPPING_CALLBACK)
    
    y_pred = model.predict(X_test_filtered)
    y_pred_proba = model.predict_proba(X_test_filtered)[:, 1] if unique_classes == 2 else model.predict_proba(X_test_filtered)
    
    end_time = time.time()
    runtime = end_time - start_time
    
    if metric == 'auc':
        if unique_classes == 2:
            score = evaluate_model(y_test, y_pred, y_pred_proba, metric)
        else:
            score = roc_auc_score(pd.get_dummies(y_test), y_pred_proba, average='macro')
    else:
        score = evaluate_model(y_test, y_pred, None, metric)
    
    return score, runtime, len(selected_features)

def main(dataset_name, n_splits=5):
    # Setup based on dataset
    if dataset_name == 'adni':
        target_col = 'DIAGNOSIS'
        dataset_path = f'datasets/{dataset_name}.csv'
        metric = 'accuracy'
    elif dataset_name == 'mimic':
        target_col = 'mortality_flag'
        dataset_path = f'datasets/{dataset_name}.csv'
        metric = 'auc'
    
    # Load dataset
    df = pd.read_csv(dataset_path, encoding='utf-8')
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    if MODEL_TYPE == "xgboost" and len(y.unique()) > 2:
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        y = pd.Series(y_encoded, index=y.index)
    
    unique_classes = len(np.unique(y))
    
    # Load or create knowledge graph for GRACE approaches
    if os.path.exists(f"kg/{dataset_name}.graphml"):
        graph_nx = nx.read_graphml(f"kg/{dataset_name}.graphml")
        feature_interactions = nx_to_feature_interactions(graph_nx)
    else:
        arxiv_kb_instance = setup_lancedb_knowledge_base(queries=[], dataset_name=dataset_name, recreate_db=True) 
        graph_nx = run_kg_workflows(arxiv_kb=arxiv_kb_instance, recreate_search=True)
        feature_interactions = nx_to_feature_interactions(graph_nx)
    
    # Results dictionary
    results = {
        'No Processing': [],
        'PCA': [],
        'RFE': [],
        'GRACE': [],
        'Graph No Constraints': []
    }
    
    # Track runtime and feature counts
    runtimes = {
        'No Processing': [],
        'PCA': [],
        'RFE': [],
        'GRACE': [],
        'Graph No Constraints': []
    }
    
    feature_counts = {
        'No Processing': [],
        'PCA': [],
        'RFE': [],
        'GRACE': [],
        'Graph No Constraints': []
    }
    
    # Create stratified folds with 30% test size
    fold_indices = []
    
    # First, create stratified folds with test_size=0.3
    for i in range(n_splits):
        # For each fold, use a different random seed to ensure diversity
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=42+i
        )
        
        # Store indices for this fold
        train_val_indices = X_train_val.index.tolist()
        test_indices = X_test.index.tolist()
        fold_indices.append((train_val_indices, test_indices))
    
    # Approach 1: No processing - run for all folds
    for fold, (train_val_idx, test_idx) in enumerate(fold_indices):
        
        # Get data using the indices
        X_train_val, X_test = X.loc[train_val_idx], X.loc[test_idx]
        y_train_val, y_test = y.loc[train_val_idx], y.loc[test_idx]
        
        # Further split training data into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=VAL_SIZE, random_state=42
        )
        
        start_time = time.time()
        
        if unique_classes == 2:
            model_baseline = lgb.LGBMClassifier(objective='binary', **LIGHTGBM_PARAMS)
        else:
            model_baseline = lgb.LGBMClassifier(objective='multiclass', num_class=unique_classes, **LIGHTGBM_PARAMS)
            
        model_baseline.fit(
            X_train, y_train, 
            eval_set=(X_val, y_val),
            callbacks=[lgb.early_stopping(10, verbose=False)]
        )
        
        y_pred = model_baseline.predict(X_test)
        y_pred_proba = model_baseline.predict_proba(X_test)[:, 1] if unique_classes == 2 else model_baseline.predict_proba(X_test)
        
        end_time = time.time()
        runtime = end_time - start_time
        runtimes['No Processing'].append(runtime)
        feature_counts['No Processing'].append(X_train.shape[1])
        
        if metric == 'auc':
            # For binary classification
            if unique_classes == 2:
                score = evaluate_model(y_test, y_pred, y_pred_proba, metric)
            # For multiclass, use macro average AUC
            else:
                score = roc_auc_score(pd.get_dummies(y_test), y_pred_proba, average='macro')
        else:
            score = evaluate_model(y_test, y_pred, None, metric)
            
        results['No Processing'].append(score)
    
    # Approach 2: PCA - run for all folds
    for fold, (train_val_idx, test_idx) in enumerate(fold_indices):
        
        # Get data using the indices
        X_train_val, X_test = X.loc[train_val_idx], X.loc[test_idx]
        y_train_val, y_test = y.loc[train_val_idx], y.loc[test_idx]
        
        # Further split training data into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=VAL_SIZE, random_state=42
        )
        
        start_time = time.time()
        
        # Handle NaN values by replacing with column means
        X_train_for_pca = X_train.copy()
        X_val_for_pca = X_val.copy()
        X_test_for_pca = X_test.copy()
        
        # Fill NaN values with column means
        col_means = X_train_for_pca.mean()
        X_train_for_pca = X_train_for_pca.fillna(col_means)
        X_val_for_pca = X_val_for_pca.fillna(col_means)
        X_test_for_pca = X_test_for_pca.fillna(col_means)
        
        n_components = min(X_train.shape[1], X_train.shape[0]) // 2
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train_for_pca)
        X_val_pca = pca.transform(X_val_for_pca)
        X_test_pca = pca.transform(X_test_for_pca)
        
        if unique_classes == 2:
            model_pca = lgb.LGBMClassifier(objective='binary', **LIGHTGBM_PARAMS)
        else:
            model_pca = lgb.LGBMClassifier(objective='multiclass', num_class=unique_classes, **LIGHTGBM_PARAMS)

        model_pca.fit(
            X_train_pca, y_train, 
            eval_set=(X_val_pca, y_val),
            callbacks=[lgb.early_stopping(10, verbose=False)]
        )
        
        y_pred = model_pca.predict(X_test_pca)
        y_pred_proba = model_pca.predict_proba(X_test_pca)[:, 1] if unique_classes == 2 else model_pca.predict_proba(X_test_pca)
        
        end_time = time.time()
        runtime = end_time - start_time
        runtimes['PCA'].append(runtime)
        feature_counts['PCA'].append(X_train_pca.shape[1])
        
        if metric == 'auc':
            if unique_classes == 2:
                score = evaluate_model(y_test, y_pred, y_pred_proba, metric)
            else:
                score = roc_auc_score(pd.get_dummies(y_test), y_pred_proba, average='macro')
        else:
            score = evaluate_model(y_test, y_pred, None, metric)
            
        results['PCA'].append(score)
    
    # Approach 3: Recursive Feature Elimination - run for all folds
    for fold, (train_val_idx, test_idx) in enumerate(fold_indices):
        
        # Get data using the indices
        X_train_val, X_test = X.loc[train_val_idx], X.loc[test_idx]
        y_train_val, y_test = y.loc[train_val_idx], y.loc[test_idx]
        
        # Further split training data into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=VAL_SIZE, random_state=42
        )
        
        start_time = time.time()
        
        # Train a model on all features
        if unique_classes == 2:
            model_all = lgb.LGBMClassifier(objective='binary', **LIGHTGBM_PARAMS)
        else:
            model_all = lgb.LGBMClassifier(objective='multiclass', num_class=unique_classes, **LIGHTGBM_PARAMS)
            
        model_all.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            callbacks=[lgb.early_stopping(10, verbose=False)]
        )
        
        # Get feature importances and select top 50%
        importances = model_all.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_k = len(indices) // 2  # Select top 50% of features
        
        # Select only the top features
        selected_features = X_train.columns[indices[:top_k]]
        X_train_rfe = X_train[selected_features]
        X_val_rfe = X_val[selected_features]
        X_test_rfe = X_test[selected_features]
        
        # Train on reduced feature set
        if unique_classes == 2:
            model_rfe = lgb.LGBMClassifier(objective='binary', **LIGHTGBM_PARAMS)
        else:
            model_rfe = lgb.LGBMClassifier(objective='multiclass', num_class=unique_classes, **LIGHTGBM_PARAMS)
            
        model_rfe.fit(
            X_train_rfe, y_train, 
            eval_set=(X_val_rfe, y_val),
            callbacks=[lgb.early_stopping(10, verbose=False)]
        )
        
        y_pred = model_rfe.predict(X_test_rfe)
        y_pred_proba = model_rfe.predict_proba(X_test_rfe)[:, 1] if unique_classes == 2 else model_rfe.predict_proba(X_test_rfe)
        
        end_time = time.time()
        runtime = end_time - start_time
        runtimes['RFE'].append(runtime)
        feature_counts['RFE'].append(X_train_rfe.shape[1])
        
        if metric == 'auc':
            if unique_classes == 2:
                score = evaluate_model(y_test, y_pred, y_pred_proba, metric)
            else:
                score = roc_auc_score(pd.get_dummies(y_test), y_pred_proba, average='macro')
        else:
            score = evaluate_model(y_test, y_pred, None, metric)
            
        results['RFE'].append(score)

    # Approach 4: GRACE
    for fold, (train_val_idx, test_idx) in enumerate(fold_indices):
        X_train_val, X_test = X.loc[train_val_idx], X.loc[test_idx]
        y_train_val, y_test = y.loc[train_val_idx], y.loc[test_idx]
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=VAL_SIZE, random_state=42
        )
        
        score, runtime, n_features = run_grace_approach(
            X_train, X_val, X_test, y_train, y_val, y_test, 
            feature_interactions, metric, unique_classes
        )
        
        results['GRACE'].append(score)
        runtimes['GRACE'].append(runtime)
        feature_counts['GRACE'].append(n_features)

    # Approach 5: Graph optimization without constraints
    for fold, (train_val_idx, test_idx) in enumerate(fold_indices):
        X_train_val, X_test = X.loc[train_val_idx], X.loc[test_idx]
        y_train_val, y_test = y.loc[train_val_idx], y.loc[test_idx]
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=VAL_SIZE, random_state=42
        )
        
        score, runtime, n_features = run_graph_no_constraints(
            X_train, X_val, X_test, y_train, y_val, y_test, 
            feature_interactions, metric, unique_classes
        )
        
        results['Graph No Constraints'].append(score)
        runtimes['Graph No Constraints'].append(runtime)
        feature_counts['Graph No Constraints'].append(n_features)

    # Calculate and print average results
    print("\n=== Summary ===")
    summary_data = []
    for method in results:
        avg_score = np.mean(results[method])
        std_score = np.std(results[method])
        avg_runtime = np.mean(runtimes[method])
        avg_features = np.mean(feature_counts[method])
        print(f"{method} - Avg {metric}: {avg_score:.4f}±{std_score:.4f}, Avg Runtime: {avg_runtime:.2f}s, Avg Features: {avg_features:.1f}")
        
        summary_data.append({
            'Method': method,
            f'Avg_{metric}': avg_score,
            f'Std_{metric}': std_score,
            'Avg_Runtime': avg_runtime,
            'Avg_Features': avg_features
        })
    
    # Save detailed results to CSV
    os.makedirs("results", exist_ok=True)
    detailed_results = []
    for method in results:
        for fold in range(n_splits):
            detailed_results.append({
                'Method': method,
                'Fold': fold + 1,
                f'{metric}': results[method][fold],
                'Runtime': runtimes[method][fold],
                'Features': feature_counts[method][fold]
            })
    
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv(f"results/{dataset_name}_detailed_comparison.csv", index=False)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"results/{dataset_name}_summary_comparison.csv", index=False)
    
    print(f"\nResults saved to results/{dataset_name}_detailed_comparison.csv and results/{dataset_name}_summary_comparison.csv")

def run_both_datasets():
    """Run comparison for both datasets and create comprehensive results."""
    datasets = ['mimic', 'adni']
    all_results = []
    
    for dataset in datasets:
        print(f"\n{'='*50}")
        print(f"Running comprehensive baseline comparison for {dataset}")
        print(f"{'='*50}")
        main(dataset)
        
        # Read the summary results
        summary_df = pd.read_csv(f"results/{dataset}_summary_comparison.csv")
        summary_df['Dataset'] = dataset
        all_results.append(summary_df)
    
    # Combine results from both datasets
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv("results/combined_comparison.csv", index=False)
    
    print(f"\n{'='*50}")
    print("COMPREHENSIVE COMPARISON RESULTS")
    print(f"{'='*50}")
    
    for dataset in datasets:
        dataset_results = combined_df[combined_df['Dataset'] == dataset]
        metric_col = 'Avg_auc' if dataset == 'mimic' else 'Avg_accuracy'
        std_col = 'Std_auc' if dataset == 'mimic' else 'Std_accuracy'
        metric_name = 'AUC' if dataset == 'mimic' else 'Accuracy'
        
        print(f"\n{dataset.upper()} Dataset ({metric_name}):")
        print("-" * 40)
        for _, row in dataset_results.iterrows():
            print(f"{row['Method']:20} - {metric_name}: {row[metric_col]:.4f}±{row[std_col]:.4f}, "
                  f"Runtime: {row['Avg_Runtime']:.2f}s, Features: {row['Avg_Features']:.1f}")
    
    print(f"\nCombined results saved to results/combined_comparison.csv")

if __name__ == "__main__":
    # Run both datasets
    run_both_datasets()