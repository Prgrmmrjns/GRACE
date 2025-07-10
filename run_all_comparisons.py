import pandas as pd
import numpy as np
import json
from sklearn.model_selection import StratifiedKFold, train_test_split
import lightgbm as lgb
from baseline_comparison import run_no_processing, run_pca, run_rfe, run_grace
import warnings
import params
from params import VAL_SIZE, PARAMS

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def save_results_to_csv(results, dataset_name):
    """Save results to CSV files for easy analysis."""
    
    # Create detailed results table
    detailed_results = []
    
    for method in ['No Processing', 'PCA', 'RFE', 'GRACE']:
        for fold in range(5):
            fold_idx = fold + 1
            if fold < len(results[method]['test_scores']):
                detailed_results.append({
                    'Dataset': dataset_name,
                    'Method': method,
                    'Fold': fold_idx,
                    'Test_Score': results[method]['test_scores'][fold],
                    'Num_Features': results[method]['num_features'][fold],
                    'Runtime': results[method]['runtimes'][fold]
                })
    
    # Save detailed results
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv(f'results/results_{dataset_name}_detailed.csv', index=False)
    
    # Create summary statistics table
    summary_results = []
    
    for method in ['No Processing', 'PCA', 'RFE', 'GRACE']:
        test_scores = [s for s in results[method]['test_scores'] if not np.isnan(s)]
        num_features = [f for f in results[method]['num_features'] if not np.isnan(f)]
        runtimes = [r for r in results[method]['runtimes'] if not np.isnan(r)]
        
        if test_scores:
            summary_results.append({
                'Dataset': dataset_name,
                'Method': method,
                'Mean_Test_Score': np.mean(test_scores),
                'Std_Test_Score': np.std(test_scores),
                'Mean_Num_Features': np.mean(num_features),
                'Mean_Runtime': np.mean(runtimes)
            })
    
    # Save summary results
    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv(f'results/{dataset_name}_summary.csv', index=False)

def generate_latex_table(all_results):
    """Generate LaTeX table from all results."""
    
    # Prepare data for LaTeX table
    methods = ['No Processing', 'PCA', 'RFE', 'GRACE']
    datasets = ['adni', 'mimic']
    
    table_data = {}
    for method in methods:
        table_data[method] = {}
        for dataset in datasets:
            if dataset in all_results:
                test_scores = [s for s in all_results[dataset][method]['test_scores'] if not np.isnan(s)]
                if test_scores:
                    table_data[method][dataset] = {
                        'mean': np.mean(test_scores),
                        'std': np.std(test_scores)
                    }
                else:
                    table_data[method][dataset] = {'mean': np.nan, 'std': np.nan}
            else:
                table_data[method][dataset] = {'mean': np.nan, 'std': np.nan}
    
    # Generate LaTeX table
    latex_content = r"""\begin{table}[h]
\centering
\begin{tabular}{l|cc|cc}
\hline
& \multicolumn{2}{c|}{ADNI (Accuracy)} & \multicolumn{2}{c}{MIMIC (AUC)} \\
Method & Mean & Std & Mean & Std \\
\hline
"""
    
    for method in methods:
        adni_data = table_data[method]['adni']
        mimic_data = table_data[method]['mimic']
        
        # Format values
        if not np.isnan(adni_data['mean']):
            adni_mean = f"{adni_data['mean']:.4f}"
            adni_std = f"{adni_data['std']:.4f}"
        else:
            adni_mean = "N/A"
            adni_std = "N/A"
            
        if not np.isnan(mimic_data['mean']):
            mimic_mean = f"{mimic_data['mean']:.4f}"
            mimic_std = f"{mimic_data['std']:.4f}"
        else:
            mimic_mean = "N/A"
            mimic_std = "N/A"
        
        latex_content += f"{method} & {adni_mean} & {adni_std} & {mimic_mean} & {mimic_std} \\\\\n"
    
    latex_content += r"""\hline
\end{tabular}
\caption{Comparison of dimensionality reduction techniques - Performance metrics}
\label{tab:performance_comparison}
\end{table}"""
    
    # Save to file
    with open('manuscript_files/performance_results.tex', 'w') as f:
        f.write(latex_content)

def run_single_dataset_comparison(dataset_name, target_col, val_size):
    """Run baseline comparison for a single dataset."""
    
    # Load data
    df = pd.read_csv(f'datasets/{dataset_name}.csv')
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
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
            X_train_full, y_train_full, test_size=val_size, random_state=42, stratify=y_train_full
        )
        
        # Method functions mapping
        methods = {
            'No Processing': lambda: run_no_processing(X_train, y_train, X_val, y_val, X_test, y_test),
            'PCA': lambda: run_pca(X_train, y_train, X_val, y_val, X_test, y_test),
            'RFE': lambda: run_rfe(X_train, y_train, X_val, y_val, X_test, y_test),
            'GRACE': lambda: run_grace(X_train, y_train, X_val, y_val, X_test, y_test, dataset_name)
        }
        
        # Run each method
        for method_name, method_func in methods.items():
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
                f"{np.mean(runtimes):.1f}"
            ])
    
    print(f"{'Method':<15} {'Mean':<8} {'Std':<8} {'Features':<8} {'Runtime':<8}")
    print("-" * 55)
    for row in summary_table:
        print(f"{row[0]:<15} {row[1]:<8} {row[2]:<8} {row[3]:<8} {row[4]:<8}")
    
    # Print detailed fold-by-fold results
    print(f"\n{'Method':<15} {'Fold 1':<8} {'Fold 2':<8} {'Fold 3':<8} {'Fold 4':<8} {'Fold 5':<8}")
    print("-" * 80)
    for method in ['No Processing', 'PCA', 'RFE', 'GRACE']:
        scores = [f"{s:.4f}" if not np.isnan(s) else "FAILED" for s in results[method]['test_scores']]
        print(f"{method:<15} {scores[0]:<8} {scores[1]:<8} {scores[2]:<8} {scores[3]:<8} {scores[4]:<8}")
    
    return results

def run_all_comparisons():
    """Run baseline comparisons for both datasets."""

    base_model_params = PARAMS.copy()
    for key in ['objective', 'num_class']:
        base_model_params.pop(key, None)

    DATASET_CONFIGS = {
        "adni": {
            "target_col": "DIAGNOSIS",
            "metric": "accuracy",
            "predict_fn": lambda m, d: m.predict(d),
            "model_params_override": {
                'objective': 'multiclass', 'num_class': 3
            }
        },
        "mimic": {
            "target_col": "mortality_flag",
            "metric": "roc_auc",
            "predict_fn": lambda model, X: model.predict_proba(X)[:, 1],
            "model_params_override": {
                'objective': 'binary'
            }
        }
    }
    
    datasets = ['adni', 'mimic']
    all_results = {}

    # Backup original params
    original_params = {
        'DATASET_NAME': params.DATASET_NAME,
        'TARGET_COL': params.TARGET_COL,
        'METRIC': params.METRIC,
        'PREDICT_FN': params.PREDICT_FN,
        'ML_MODEL': params.ML_MODEL,
    }
    
    for dataset in datasets:
        print(f"\n{'='*100}")
        print(f"RUNNING COMPARISON FOR {dataset.upper()} DATASET")
        print(f"{'='*100}")
        
        config = DATASET_CONFIGS[dataset]
        
        model_params = base_model_params.copy()
        model_params.update(config["model_params_override"])
        
        # Update params for current dataset
        params.DATASET_NAME = dataset
        params.TARGET_COL = config["target_col"]
        params.METRIC = config["metric"]
        params.PREDICT_FN = config["predict_fn"]
        
        model_class = type(params.ML_MODEL)
        params.ML_MODEL = model_class(**model_params)
        
        results = run_single_dataset_comparison(dataset, config["target_col"], VAL_SIZE)
        all_results[dataset] = results
        
        # Save results to CSV
        save_results_to_csv(results, dataset)
    
        # Restore original parameters
    for key, value in original_params.items():
        setattr(params, key, value)
    
    # Save combined results
    with open('results/all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Generate LaTeX table
    generate_latex_table(all_results)
    
    return all_results

if __name__ == "__main__":
    results = run_all_comparisons() 