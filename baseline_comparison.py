from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time
import copy
from main import grace_feature_selection, apply_grace_to_test
import params

def run_no_processing(X_train, y_train, X_val, y_val, X_test, y_test):
    """Run baseline without any preprocessing."""
    
    start_time = time.time()
    model = copy.deepcopy(params.ML_MODEL)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    test_preds = params.PREDICT_FN(model, X_test)
    test_score = roc_auc_score(y_test, test_preds) if params.METRIC == 'roc_auc' else accuracy_score(y_test, test_preds)
    runtime = time.time() - start_time
    
    return {
        'test_score': test_score,
        'num_features': X_train.shape[1],
        'runtime': runtime
    }

def run_pca(X_train, y_train, X_val, y_val, X_test, y_test):
    """Run PCA with smart component selection using explained variance and validation performance."""
    
    start_time = time.time()
    
    # Handle missing values with imputation
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    X_test_imputed = imputer.transform(X_test)
    
    # First, find components that explain 80%, 90%, 95%, 99% of variance
    pca_full = PCA(random_state=42)
    pca_full.fit(X_train_imputed)
    cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
    
    # Get candidate component counts based on variance thresholds
    variance_thresholds = [0.8, 0.9, 0.95, 0.99]
    candidates = []
    for threshold in variance_thresholds:
        n_comp = np.argmax(cumsum_var >= threshold) + 1
        candidates.append(min(n_comp, len(cumsum_var)))
    
    # Add some additional strategic points
    n_features = X_train_imputed.shape[1]
    candidates.extend([
        max(2, n_features // 10),  # 10% of features
        max(5, n_features // 5),   # 20% of features
        max(10, n_features // 3),  # 33% of features
        max(15, n_features // 2),  # 50% of features
    ])
    
    # Remove duplicates and sort
    candidates = sorted(list(set(candidates)))
    candidates = [c for c in candidates if c >= 2 and c <= min(50, n_features, X_train_imputed.shape[0] - 1)]
    
    best_val_score = -1
    best_n_components = candidates[0] if candidates else 2
    
    # Use a faster model for optimization
    model = copy.deepcopy(params.ML_MODEL)
    
    for n_comp in candidates:
        pca_temp = PCA(n_components=n_comp, random_state=42)
        X_train_temp = pca_temp.fit_transform(X_train_imputed)
        X_val_temp = pca_temp.transform(X_val_imputed)
        
        # Use fast model for optimization
        model.fit(X_train_temp, y_train, eval_set=[(X_val_temp, y_val)])
        if params.METRIC == 'roc_auc':
            val_preds = model.predict_proba(X_val_temp)[:, 1]
            val_score = roc_auc_score(y_val, val_preds)
        else:
            val_preds = model.predict(X_val_temp)
            val_score = accuracy_score(y_val, val_preds)
        
        if val_score > best_val_score:
            best_val_score = val_score
            best_n_components = n_comp
    
    # Use best number of components for final model with original ML model
    pca = PCA(n_components=best_n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_imputed)
    X_val_pca = pca.transform(X_val_imputed)
    X_test_pca = pca.transform(X_test_imputed)
    
    model = copy.deepcopy(params.ML_MODEL)
    model.fit(X_train_pca, y_train, eval_set=[(X_val_pca, y_val)])
    test_preds = params.PREDICT_FN(model, X_test_pca)
    test_score = roc_auc_score(y_test, test_preds) if params.METRIC == 'roc_auc' else accuracy_score(y_test, test_preds)
    runtime = time.time() - start_time
    
    return {
        'test_score': test_score,
        'num_features': X_train_pca.shape[1],
        'runtime': runtime
    }

def run_rfe(X_train, y_train, X_val, y_val, X_test, y_test):
    """Run Recursive Feature Elimination."""
    
    start_time = time.time()
    
    # Handle missing values with imputation
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    X_test_imputed = imputer.transform(X_test)
    
    # Use a simpler model for RFE estimation without early stopping
    estimator = copy.deepcopy(params.ML_MODEL)
    estimator.set_params(n_estimators=50, early_stopping_rounds=None)
    
    # Determine number of features for RFE (similar to what we get with GRACE)
    n_features_select = max(5, min(20, X_train.shape[1] // 3))
    
    rfe = RFE(estimator=estimator, n_features_to_select=n_features_select, verbose=0)
    X_train_rfe = rfe.fit_transform(X_train_imputed, y_train)
    X_val_rfe = rfe.transform(X_val_imputed)
    X_test_rfe = rfe.transform(X_test_imputed)
    
    model = copy.deepcopy(params.ML_MODEL)
    model.fit(X_train_rfe, y_train, eval_set=[(X_val_rfe, y_val)])
    test_preds = params.PREDICT_FN(model, X_test_rfe)
    test_score = roc_auc_score(y_test, test_preds) if params.METRIC == 'roc_auc' else accuracy_score(y_test, test_preds)
    runtime = time.time() - start_time
    
    return {
        'test_score': test_score,
        'num_features': X_train_rfe.shape[1],
        'runtime': runtime
    }

def run_grace(X_train, y_train, X_val, y_val, X_test, y_test, dataset_name):
    """Run GRACE feature selection with LightGBM interaction constraints."""
    
    start_time = time.time()
    
    # Get trained model with interaction constraints
    model = grace_feature_selection(X_train, y_train, X_val, y_val, params.ML_MODEL, dataset_name)
    
    # Apply feature selection to test set
    X_test_grace = apply_grace_to_test(model, X_test)
    
    # Make predictions on test set
    test_preds = params.PREDICT_FN(model, X_test_grace)
    test_score = roc_auc_score(y_test, test_preds) if params.METRIC == 'roc_auc' else accuracy_score(y_test, test_preds)
    
    runtime = time.time() - start_time
    
    # Count the number of REMAINING features (not removed features)
    original_features = X_train.shape[1]
    removed_features = len(model.removed_features_) if hasattr(model, 'removed_features_') else 0
    remaining_features = original_features - removed_features
    
    return {
        'test_score': test_score,
        'num_features': remaining_features,
        'runtime': runtime
    }

def run_baseline_comparison(dataset_name=None):
    """
    Compare GRACE with baseline methods using 5-fold cross-validation.
    
    Args:
        dataset_name: Optional dataset name (defaults to DATASET_NAME from params)
    """
    
    if dataset_name is None:
        dataset_name = DATASET_NAME
    
    # Load data
    df = pd.read_csv(f'datasets/{dataset_name}.csv')
    X = df.drop(params.TARGET_COL, axis=1)
    y = df[params.TARGET_COL]
    
    # Initialize results storage
    results = {
        'No Processing': {'test_scores': [], 'num_features': [], 'runtimes': []},
        'PCA': {'test_scores': [], 'num_features': [], 'runtimes': []},
        'RFE': {'test_scores': [], 'num_features': [], 'runtimes': []},
        'GRACE': {'test_scores': [], 'num_features': [], 'runtimes': []},
    }
    
    # 5-fold cross-validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y), 1):
        print(f"\n=== Fold {fold} ===")
        
        # Split data
        X_train_full, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train_full, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Further split training into train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=params.VAL_SIZE, random_state=42, stratify=y_train_full
        )
        
        # Run each method
        methods = {
            'No Processing': lambda: run_no_processing(X_train, y_train, X_val, y_val, X_test, y_test),
            'PCA': lambda: run_pca(X_train, y_train, X_val, y_val, X_test, y_test),
            'RFE': lambda: run_rfe(X_train, y_train, X_val, y_val, X_test, y_test),
            'GRACE': lambda: run_grace(X_train, y_train, X_val, y_val, X_test, y_test, dataset_name)
        }
        
        for method_name, method_func in methods.items():
            print(f"Running {method_name}...")
            result = method_func()
            
            results[method_name]['test_scores'].append(result['test_score'])
            results[method_name]['num_features'].append(result['num_features'])
            results[method_name]['runtimes'].append(result['runtime'])
            
            if not np.isnan(result['test_score']):
                print(f"{method_name} - Test Score: {result['test_score']:.4f}, Features: {result['num_features']}, Runtime: {result['runtime']:.2f}s")
            else:
                print(f"{method_name} - FAILED")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    summary_table = []
    for method in ['No Processing', 'PCA', 'RFE', 'GRACE']:
        test_scores = [s for s in results[method]['test_scores'] if not np.isnan(s)]
        num_features = [f for f in results[method]['num_features'] if not np.isnan(f)]
        runtimes = [r for r in results[method]['runtimes'] if not np.isnan(r)]
        
        if test_scores:
            summary_table.append([
                method,
                f"{np.mean(test_scores):.4f}",
                f"{np.std(test_scores):.4f}",
                f"{np.mean(num_features):.1f}",
                f"{np.mean(runtimes):.1f}",
                len(test_scores)
            ])
    
    print(f"{'Method':<15} {'Mean':<8} {'Std':<8} {'Features':<8} {'Runtime':<8} {'Folds':<5}")
    print("-" * 60)
    for row in summary_table:
        print(f"{row[0]:<15} {row[1]:<8} {row[2]:<8} {row[3]:<8} {row[4]:<8} {row[5]:<5}")
    
    # Print detailed fold-by-fold results
    print(f"\n{'Method':<15} {'Fold 1':<8} {'Fold 2':<8} {'Fold 3':<8} {'Fold 4':<8} {'Fold 5':<8}")
    print("-" * 80)
    for method in ['No Processing', 'PCA', 'RFE', 'GRACE']:
        scores = [f"{s:.4f}" if not np.isnan(s) else "FAILED" for s in results[method]['test_scores']]
        print(f"{method:<15} {scores[0]:<8} {scores[1]:<8} {scores[2]:<8} {scores[3]:<8} {scores[4]:<8}")
    
    return results

if __name__ == "__main__":
    from params import DATASET_NAME
    print(f"Running baseline comparison for {DATASET_NAME} dataset...")
    results = run_baseline_comparison() 