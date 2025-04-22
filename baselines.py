import os
import pickle
import time
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import subprocess
import warnings
import sys

# Ultimate warning suppression - modify the warning showwarning function
def suppress_all_warnings(*args, **kwargs):
    pass
warnings.showwarning = suppress_all_warnings
warnings.filterwarnings('ignore')

# Also completely block stderr
os.environ['PYTHONWARNINGS'] = 'ignore'

# Capture the actual stderr file descriptor and redirect it

devnull = open(os.devnull, 'w')
old_stderr = sys.stderr
sys.stderr = devnull

def evaluate_model(y_true, y_pred, y_pred_proba=None, metric_name='accuracy'):
    if metric_name == 'accuracy':
        return accuracy_score(y_true, y_pred)
    elif metric_name == 'auc':
        if y_pred_proba is None:
            raise ValueError("Predicted probabilities required for AUC calculation")
        return roc_auc_score(y_true, y_pred_proba)

def run_comparison(dataset_name='mimic', n_splits=5):
    # Setup based on dataset
    if dataset_name == 'adni':
        target_col = 'DIAGNOSIS'
        dataset_path = f'example_datasets/{dataset_name}.csv'
        metric = 'accuracy'
    elif dataset_name == 'mimic':
        target_col = 'mortality_flag'
        dataset_path = f'example_datasets/{dataset_name}.csv'
        metric = 'auc'
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Load dataset
    df = pd.read_csv(dataset_path, encoding='utf-8')
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    unique_classes = len(np.unique(y))
    obj = 'binary' if unique_classes == 2 else 'multiclass'
    
    # Model parameters
    model_params = {
        'objective': obj,
        'boosting_type': 'gbdt',
        'learning_rate': 0.1,
        'n_estimators': 1000,
        'min_split_gain': 20,
        'random_state': 42,
        'num_threads': 6,
        'data_sample_strategy': 'goss',
        'use_quantized_grad': True,
        'verbosity': -1
    }
    
    if obj == 'multiclass':
        model_params['num_class'] = unique_classes
    
    if metric == 'accuracy':
        model_params['metric'] = 'multi_error' if obj == 'multiclass' else 'binary_error'
    elif metric == 'auc':
        model_params['metric'] = 'auc' if obj == 'binary' else 'multi_logloss'
    
    # Initialize cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
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
    
    # Cross-validation loop
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold+1}/{n_splits}")
        
        # Split data
        X_train_val, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train_val, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Further split training data into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=42
        )
        
        callbacks = [lgb.early_stopping(stopping_rounds=5, verbose=False)]
        
        # Approach 1: No processing
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
        
        # Approach 2: PCA
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
        
        # Approach 3: Recursive Feature Elimination
        start_time = time.time()
        
        base_model = lgb.LGBMClassifier(**model_params)
        rfe = RFECV(
            estimator=base_model,
            step=5,
            cv=3,
            scoring='accuracy' if metric == 'accuracy' else 'roc_auc',
            n_jobs=-1
        )
        
        rfe.fit(X_train, y_train)
        X_train_rfe = X_train.iloc[:, rfe.support_]
        X_val_rfe = X_val.iloc[:, rfe.support_]
        X_test_rfe = X_test.iloc[:, rfe.support_]
        
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
        
        # Approach 4: KG-Based (load from saved model)
        start_time = time.time()
        # Run main.py with interpretation and verbosity disabled
        
        
        # Pass environment variables to control main.py behavior
        env = os.environ.copy()
        env['SKIP_INTERPRETATION'] = 'true'
        
        # Call main.py with command line arguments for verbosity and interpretation
        subprocess.run([
            "python", "main.py", 
        ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Load model dir
        model_dir = f"./models/{dataset_name}"
        
        # Load results
        with open(f"{model_dir}/data.pkl", 'rb') as f:
            kg_data = pickle.load(f)
        with open(f"{model_dir}/final_model.pkl", 'rb') as f:
            kg_model = pickle.load(f)
            
        # Use test indices to identify common test samples
        common_indices = np.intersect1d(kg_data['X_test'].index, X_test.index)
        kg_test_idx = np.where(np.isin(kg_data['X_test'].index, common_indices))[0]
        
        # Get predictions for test set
        kg_y_test = kg_data['y_test'].iloc[kg_test_idx]
        kg_X_test_intermediate = kg_data['X_test_intermediate'].iloc[kg_test_idx]
        
        y_pred = kg_model.predict(kg_X_test_intermediate)
        y_pred_proba = kg_model.predict_proba(kg_X_test_intermediate)
        
        end_time = time.time()
        runtime = end_time - start_time
        runtimes['KG-Based'].append(runtime)
        
        # Count selected features (intermediate nodes)
        kg_feature_count = kg_X_test_intermediate.shape[1]
        feature_counts['KG-Based'].append(kg_feature_count)
        
        if metric == 'auc':
            if obj == 'binary':
                score = evaluate_model(kg_y_test, y_pred, y_pred_proba[:, 1], metric)
            else:
                score = roc_auc_score(pd.get_dummies(kg_y_test), y_pred_proba, average='macro')
        else:
            score = evaluate_model(kg_y_test, y_pred, None, metric)
            
        results['KG-Based'].append(score)
        print(f"KG-Based {metric}: {score:.4f}, Runtime: {runtime:.2f}s, Features: {kg_feature_count}")
    
    # Calculate mean and std for each method
    summary = {}
    for method, scores in results.items():
        # Process performance scores
        scores_array = np.array(scores)
        scores_array = scores_array[~np.isnan(scores_array)]  # Remove NaN values
        
        # Process runtime
        runtime_array = np.array(runtimes[method])
        runtime_array = runtime_array[~np.isnan(runtime_array)]
        
        # Process feature counts
        feature_array = np.array(feature_counts[method])
        feature_array = feature_array[~np.isnan(feature_array)]

        mean_score = np.mean(scores_array)
        std_score = np.std(scores_array)
        mean_runtime = np.mean(runtime_array) if len(runtime_array) > 0 else np.nan
        std_runtime = np.std(runtime_array) if len(runtime_array) > 0 else np.nan
        mean_features = np.mean(feature_array) if len(feature_array) > 0 else np.nan
        std_features = np.std(feature_array) if len(feature_array) > 0 else np.nan
        
        summary[method] = {
            'mean': mean_score, 
            'std': std_score,
            'runtime_mean': mean_runtime,
            'runtime_std': std_runtime,
            'features_mean': mean_features,
            'features_std': std_features
        }
        
        print(f"{method}: {mean_score:.4f} ± {std_score:.4f}, "
                f"Runtime: {mean_runtime:.2f}s ± {std_runtime:.2f}s, "
                f"Features: {mean_features:.1f} ± {std_features:.1f}")
    
    return results, summary, metric, runtimes, feature_counts

def create_visualizations(datasets=['adni', 'mimic']):
    all_results = {}
    all_summaries = {}
    metrics = {}
    all_runtimes = {}
    all_feature_counts = {}
    
    for dataset in datasets:
        results, summary, metric, runtimes, feature_counts = run_comparison(dataset_name=dataset)
        all_results[dataset] = results
        all_summaries[dataset] = summary
        metrics[dataset] = metric
        all_runtimes[dataset] = runtimes
        all_feature_counts[dataset] = feature_counts
    
    # Create LaTeX tables
    performance_table = create_performance_latex_table(all_summaries, metrics)
    runtime_table = create_runtime_latex_table(all_summaries)
    feature_table = create_feature_latex_table(all_summaries)
    
    with open(f"results/performance_results.tex", "w") as f:
        f.write(performance_table)
    with open(f"results/runtime_results.tex", "w") as f:
        f.write(runtime_table)
    with open(f"results/feature_count_results.tex", "w") as f:
        f.write(feature_table)
    
    # Create visualizations
    create_performance_plot(all_summaries, metrics)
    create_runtime_plot(all_summaries)
    create_feature_count_plot(all_summaries)
    
def create_performance_latex_table(summaries, metrics):
    latex = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{l|cc|cc}\n\\hline\n"
    latex += "& \\multicolumn{2}{c|}{ADNI (Accuracy)} & \\multicolumn{2}{c}{MIMIC (AUC)} \\\\\n"
    latex += "Method & Mean & Std & Mean & Std \\\\\n\\hline\n"
    
    methods = ["No Processing", "PCA", "RFE", "KG-Based"]
    for method in methods:
        latex += f"{method} & "
        
        # ADNI results
        mean = summaries['adni'][method]['mean']
        std = summaries['adni'][method]['std']
        latex += f"{mean:.4f} & {std:.4f} & "
            
        # MIMIC results
        mean = summaries['mimic'][method]['mean']
        std = summaries['mimic'][method]['std']
        latex += f"{mean:.4f} & {std:.4f}"
            
        latex += " \\\\\n"
    
    latex += "\\hline\n\\end{tabular}\n"
    latex += "\\caption{Comparison of dimensionality reduction techniques - Performance metrics}\n"
    latex += "\\label{tab:performance_comparison}\n\\end{table}"
    
    return latex

def create_runtime_latex_table(summaries):
    latex = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{l|cc|cc}\n\\hline\n"
    latex += "& \\multicolumn{2}{c|}{ADNI (Runtime in seconds)} & \\multicolumn{2}{c}{MIMIC (Runtime in seconds)} \\\\\n"
    latex += "Method & Mean & Std & Mean & Std \\\\\n\\hline\n"
    
    methods = ["No Processing", "PCA", "RFE", "KG-Based"]
    for method in methods:
        latex += f"{method} & "
        
        # ADNI results
        mean = summaries['adni'][method]['runtime_mean']
        std = summaries['adni'][method]['runtime_std']
        latex += f"{mean:.2f} & {std:.2f} & "
            
        # MIMIC results
        mean = summaries['mimic'][method]['runtime_mean']
        std = summaries['mimic'][method]['runtime_std']
        latex += f"{mean:.2f} & {std:.2f}"
            
        latex += " \\\\\n"
    
    latex += "\\hline\n\\end{tabular}\n"
    latex += "\\caption{Comparison of dimensionality reduction techniques - Runtime (seconds)}\n"
    latex += "\\label{tab:runtime_comparison}\n\\end{table}"
    
    return latex

def create_feature_latex_table(summaries):
    latex = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{l|cc|cc}\n\\hline\n"
    latex += "& \\multicolumn{2}{c|}{ADNI (Feature Count)} & \\multicolumn{2}{c}{MIMIC (Feature Count)} \\\\\n"
    latex += "Method & Mean & Std & Mean & Std \\\\\n\\hline\n"
    
    methods = ["No Processing", "PCA", "RFE", "KG-Based"]
    for method in methods:
        latex += f"{method} & "
        
        # ADNI results
        mean = summaries['adni'][method]['features_mean']
        std = summaries['adni'][method]['features_std']
        latex += f"{mean:.1f} & {std:.1f} & "

        # MIMIC results
        mean = summaries['mimic'][method]['features_mean']
        std = summaries['mimic'][method]['features_std']
        latex += f"{mean:.1f} & {std:.1f}"
            
        latex += " \\\\\n"
    
    latex += "\\hline\n\\end{tabular}\n"
    latex += "\\caption{Comparison of dimensionality reduction techniques - Feature Count}\n"
    latex += "\\label{tab:feature_count_comparison}\n\\end{table}"
    
    return latex

def create_performance_plot(summaries, metrics):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot for ADNI
    methods = []
    means = []
    stds = []
        
    for method, stats in summaries['adni'].items():
        methods.append(method)
        means.append(stats['mean'])
        stds.append(stats['std'])

    # Fixed: Use matplotlib's bar plot with error bars instead of seaborn
    x_pos = np.arange(len(methods))
    ax1.bar(x_pos, means, yerr=stds, align='center', alpha=0.7, ecolor='black', capsize=10)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods)
    ax1.set_title('ADNI Dataset (Accuracy)')
    ax1.set_ylim([max(0, min(means) - max(stds) - 0.05), min(1.0, max(means) + max(stds) + 0.05)])
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Method')
    
    # Plot for MIMIC
    methods = []
    means = []
    stds = []
        
    for method, stats in summaries['mimic'].items():
        methods.append(method)
        means.append(stats['mean'])
        stds.append(stats['std'])

    # Fixed: Use matplotlib's bar plot with error bars instead of seaborn
    x_pos = np.arange(len(methods))
    ax2.bar(x_pos, means, yerr=stds, align='center', alpha=0.7, ecolor='black', capsize=10)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(methods)
    ax2.set_title('MIMIC Dataset (AUC)')
    ax2.set_ylim([max(0, min(means) - max(stds) - 0.05), min(1.0, max(means) + max(stds) + 0.05)])
    ax2.set_ylabel('AUC')
    ax2.set_xlabel('Method')
    
    plt.tight_layout()

def create_runtime_plot(summaries):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot for ADNI
    methods = []
    means = []
    stds = []
        
    for method, stats in summaries['adni'].items():
        if not np.isnan(stats['runtime_mean']):
            methods.append(method)
            means.append(stats['runtime_mean'])
            stds.append(stats['runtime_std'])
    
    if methods:
        # Fixed: Use matplotlib's bar plot with error bars
        x_pos = np.arange(len(methods))
        ax1.bar(x_pos, means, yerr=stds, align='center', alpha=0.7, ecolor='black', capsize=10)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(methods)
        ax1.set_title('ADNI Dataset (Runtime)')
        ax1.set_ylabel('Runtime (seconds)')
        ax1.set_xlabel('Method')
    
    # Plot for MIMIC
    methods = []
    means = []
    stds = []
    
    for method, stats in summaries['mimic'].items():
        if not np.isnan(stats['runtime_mean']):
            methods.append(method)
            means.append(stats['runtime_mean'])
            stds.append(stats['runtime_std'])
    
    if methods:
        # Fixed: Use matplotlib's bar plot with error bars
        x_pos = np.arange(len(methods))
        ax2.bar(x_pos, means, yerr=stds, align='center', alpha=0.7, ecolor='black', capsize=10)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(methods)
        ax2.set_title('MIMIC Dataset (Runtime)')
        ax2.set_ylabel('Runtime (seconds)')
        ax2.set_xlabel('Method')
    
    plt.tight_layout()

def create_feature_count_plot(summaries):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot for ADNI
    methods = []
    means = []
    stds = []
    
    for method, stats in summaries['adni'].items():
        if not np.isnan(stats['features_mean']):
            methods.append(method)
            means.append(stats['features_mean'])
            stds.append(stats['features_std'])

    # Fixed: Use matplotlib's bar plot with error bars
    x_pos = np.arange(len(methods))
    ax1.bar(x_pos, means, yerr=stds, align='center', alpha=0.7, ecolor='black', capsize=10)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods)
    ax1.set_title('ADNI Dataset (Feature Count)')
    ax1.set_ylabel('Number of Features')
    ax1.set_xlabel('Method')
    
    # Plot for MIMIC
    methods = []
    means = []
    stds = []
    
    for method, stats in summaries['mimic'].items():
        methods.append(method)
        means.append(stats['features_mean'])
        stds.append(stats['features_std'])

    # Fixed: Use matplotlib's bar plot with error bars
    x_pos = np.arange(len(methods))
    ax2.bar(x_pos, means, yerr=stds, align='center', alpha=0.7, ecolor='black', capsize=10)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(methods)
    ax2.set_title('MIMIC Dataset (Feature Count)')
    ax2.set_ylabel('Number of Features')
    ax2.set_xlabel('Method')
    
    plt.tight_layout()

if __name__ == "__main__":
    np.random.seed(42)
    create_visualizations()