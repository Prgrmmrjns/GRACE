import time
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
from params import CATBOOST_PARAMS

VAL_SIZE = 0.5

# Suppress all warnings
warnings.filterwarnings("ignore")

def evaluate_model(y_true, y_pred, y_pred_proba=None, metric_name='accuracy'):
    if metric_name == 'accuracy':
        return accuracy_score(y_true, y_pred)
    elif metric_name == 'auc':
        return roc_auc_score(y_true, y_pred_proba)

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
    
    unique_classes = len(np.unique(y))
    
    # Results dictionary
    results = {
        'No Processing': [],
        'PCA': [],
        'RFE': [],
    }
    
    # Track runtime and feature counts
    runtimes = {
        'No Processing': [],
        'PCA': [],
        'RFE': [],
    }
    
    feature_counts = {
        'No Processing': [],
        'PCA': [],
        'RFE': [],
    }
    
    # Create stratified folds with 30% test size
    fold_indices = []
    
    # First, create stratified folds with test_size=0.3
    for i in range(n_splits):
        # For each fold, use a different random seed to ensure diversity
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=VAL_SIZE, stratify=y, random_state=42+i
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
        
        model_baseline = CatBoostClassifier(**CATBOOST_PARAMS)
        model_baseline.fit(
            X_train, y_train, 
            eval_set=(X_val, y_val),
            verbose=False
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
        
        model_pca = CatBoostClassifier(**CATBOOST_PARAMS)
        model_pca.fit(
            X_train_pca, y_train, 
            eval_set=(X_val_pca, y_val),
            verbose=False
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
        model_all = CatBoostClassifier(**CATBOOST_PARAMS)
        model_all.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False
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
        model_rfe = CatBoostClassifier(**CATBOOST_PARAMS)
        model_rfe.fit(
            X_train_rfe, y_train, 
            eval_set=(X_val_rfe, y_val),
            verbose=False
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

    # Calculate and print average results
    print("\n=== Summary ===")
    for method in results:
        avg_score = np.mean(results[method])
        avg_runtime = np.mean(runtimes[method])
        avg_features = np.mean(feature_counts[method])
        print(f"{method} - Avg {metric}: {avg_score:.4f}, Avg Runtime: {avg_runtime:.2f}s, Avg Features: {avg_features:.1f}")

if __name__ == "__main__":
    dataset_name = 'mimic'
    print(f"Running {dataset_name} baseline")
    main(dataset_name)