import time
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings

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
    obj = 'binary' if unique_classes == 2 else 'multiclass'
    
    # Model parameters
    model_params = {
        'learning_rate': 0.1,
        'max_depth': 3,
        'min_child_samples': 0,
        'reg_alpha': 1,
        'reg_lambda': 20,
        'path_smooth': 1,
        'objective': 'binary' if unique_classes == 2 else 'multiclass',
        'n_estimators': 1000,
        'num_threads': 10,
        'random_state': 42,
        'data_sample_strategy': 'goss',
        'verbosity': -1
    }
    
    # Create stratified folds with 30% test size
    fold_indices = []
    
    # First, create stratified folds with test_size=0.3
    for i in range(n_splits):
        # For each fold, use a different random seed to ensure diversity
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42+i
        )
        
        # Store indices for this fold
        train_val_indices = X_train_val.index.tolist()
        test_indices = X_test.index.tolist()
        fold_indices.append((train_val_indices, test_indices))
    
    # Results dictionary
    results = {
        'No Processing': [],
        'PCA': [],
        'RFE': [],
        'KG-Based': []
    }
    
    # Track runtime and feature counts
    runtimes = {
        'No Processing': [],
        'PCA': [],
        'RFE': [],
        'KG-Based': []
    }
    
    feature_counts = {
        'No Processing': [],
        'PCA': [],
        'RFE': [],
        'KG-Based': []
    }
    
    callbacks = [lgb.early_stopping(stopping_rounds=10, verbose=False)]
    
    # Approach 1: No processing - run for all folds
    print("\n=== No Processing ===")
    for fold, (train_val_idx, test_idx) in enumerate(fold_indices):
        print(f"Fold {fold+1}/{n_splits}")
        
        # Get data using the indices
        X_train_val, X_test = X.loc[train_val_idx], X.loc[test_idx]
        y_train_val, y_test = y.loc[train_val_idx], y.loc[test_idx]
        
        # Further split training data into train and validation (val is 10% of train+val)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=42
        )
        
        start_time = time.time()
        
        model_baseline = lgb.LGBMClassifier(**model_params)
        model_baseline.fit(
            X_train, y_train, 
            eval_set=[(X_val, y_val)],
            callbacks=callbacks
        )
        
        y_pred = model_baseline.predict(X_test)
        y_pred_proba = model_baseline.predict_proba(X_test)
        
        end_time = time.time()
        runtime = end_time - start_time
        runtimes['No Processing'].append(runtime)
        feature_counts['No Processing'].append(X_train.shape[1])
        
        if metric == 'auc':
            # For binary classification
            if obj == 'binary':
                score = evaluate_model(y_test, y_pred, y_pred_proba[:, 1], metric)
            # For multiclass, use macro average AUC
            else:
                score = roc_auc_score(pd.get_dummies(y_test), y_pred_proba, average='macro')
        else:
            score = evaluate_model(y_test, y_pred, None, metric)
            
        results['No Processing'].append(score)
        print(f"No Processing {metric}: {score:.4f}, Runtime: {runtime:.2f}s, Features: {X_train.shape[1]}")
    
    # Approach 2: PCA - run for all folds
    print("\n=== PCA ===")
    for fold, (train_val_idx, test_idx) in enumerate(fold_indices):
        print(f"Fold {fold+1}/{n_splits}")
        
        # Get data using the indices
        X_train_val, X_test = X.loc[train_val_idx], X.loc[test_idx]
        y_train_val, y_test = y.loc[train_val_idx], y.loc[test_idx]
        
        # Further split training data into train and validation (val is 10% of train+val)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=42
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
        
        model_pca = lgb.LGBMClassifier(**model_params)
        model_pca.fit(
            X_train_pca, y_train, 
            eval_set=[(X_val_pca, y_val)],
            callbacks=callbacks
        )
        
        y_pred = model_pca.predict(X_test_pca)
        y_pred_proba = model_pca.predict_proba(X_test_pca)
        
        end_time = time.time()
        runtime = end_time - start_time
        runtimes['PCA'].append(runtime)
        feature_counts['PCA'].append(X_train_pca.shape[1])
        
        if metric == 'auc':
            if obj == 'binary':
                score = evaluate_model(y_test, y_pred, y_pred_proba[:, 1], metric)
            else:
                score = roc_auc_score(pd.get_dummies(y_test), y_pred_proba, average='macro')
        else:
            score = evaluate_model(y_test, y_pred, None, metric)
            
        results['PCA'].append(score)
        print(f"PCA {metric}: {score:.4f}, Runtime: {runtime:.2f}s, Features: {X_train_pca.shape[1]}")
    
    # Approach 3: Recursive Feature Elimination - run for all folds
    print("\n=== RFE ===")
    for fold, (train_val_idx, test_idx) in enumerate(fold_indices):
        print(f"Fold {fold+1}/{n_splits}")
        
        # Get data using the indices
        X_train_val, X_test = X.loc[train_val_idx], X.loc[test_idx]
        y_train_val, y_test = y.loc[train_val_idx], y.loc[test_idx]
        
        # Further split training data into train and validation (val is 10% of train+val)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=42
        )
        
        start_time = time.time()
        
        # Skip RFECV as it's causing issues with sklearn tags
        # Instead, use a simpler approach for feature selection
        X_train_rfe = X_train.copy()
        X_val_rfe = X_val.copy()
        X_test_rfe = X_test.copy()
        
        # Train a model on all features
        model_all = lgb.LGBMClassifier(**model_params)
        model_all.fit(
            X_train_rfe, y_train,
            eval_set=[(X_val_rfe, y_val)],
            callbacks=callbacks
        )
        
        # Get feature importances and select top 50%
        importances = model_all.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_k = len(indices) // 2  # Select top 50% of features
        
        # Select only the top features
        selected_features = indices[:top_k]
        X_train_rfe = X_train.iloc[:, selected_features]
        X_val_rfe = X_val.iloc[:, selected_features]
        X_test_rfe = X_test.iloc[:, selected_features]
        
        # Train on reduced feature set
        model_rfe = lgb.LGBMClassifier(**model_params)
        model_rfe.fit(
            X_train_rfe, y_train, 
            eval_set=[(X_val_rfe, y_val)],
            callbacks=callbacks
        )
        
        y_pred = model_rfe.predict(X_test_rfe)
        y_pred_proba = model_rfe.predict_proba(X_test_rfe)
        
        end_time = time.time()
        runtime = end_time - start_time
        runtimes['RFE'].append(runtime)
        feature_counts['RFE'].append(X_train_rfe.shape[1])
        
        if metric == 'auc':
            if obj == 'binary':
                score = evaluate_model(y_test, y_pred, y_pred_proba[:, 1], metric)
            else:
                score = roc_auc_score(pd.get_dummies(y_test), y_pred_proba, average='macro')
        else:
            score = evaluate_model(y_test, y_pred, None, metric)
            
        results['RFE'].append(score)
        print(f"RFE {metric}: {score:.4f}, Runtime: {runtime:.2f}s, Features: {X_train_rfe.shape[1]}")

if __name__ == "__main__":
    main('mimic')