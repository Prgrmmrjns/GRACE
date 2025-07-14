from sklearn.metrics import accuracy_score, roc_auc_score
import params
import optuna
import numpy as np
from sklearn.base import clone
from visualizations import visualize_pareto_front
import networkx as nx
import itertools

optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial, X_train, y_train, X_val, y_val, col_to_idx, initial_constraints):
    group_sizes = [len(group) for group in initial_constraints]
    mask = [np.zeros(sz, dtype=bool) for sz in group_sizes]
    for i, group in enumerate(initial_constraints):
        for j, feature in enumerate(group):
            mask[i][j] = trial.suggest_categorical(f'{i}_{feature}', [True, False])
    selected_constraints = [ [group[j] for j in np.where(mask[i])[0]] for i, group in enumerate(initial_constraints) ]
    selected_constraints = [group for group in selected_constraints if len(group) > 1]
    if not selected_constraints:
        raise optuna.TrialPruned()

    hparams = {}
    for param_name, bounds in params.HPARAMS.items():
        if isinstance(bounds[0], int):
            hparams[param_name] = trial.suggest_int(param_name, bounds[0], bounds[1])
        else:
            hparams[param_name] = trial.suggest_float(param_name, bounds[0], bounds[1])
    
    model = clone(params.ML_MODEL)
    model.set_params(interaction_constraints=[[col_to_idx[feat] for feat in group] for group in selected_constraints],
                     **hparams)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    val_preds = model.predict_proba(X_val, pred_early_stop=True)[:, 1] if params.METRIC == 'roc_auc' else model.predict(X_val, pred_early_stop=True)
    val_score = roc_auc_score(y_val, val_preds) if params.METRIC == 'roc_auc' else accuracy_score(y_val, val_preds)
    
    all_edges = set()
    for group in selected_constraints:
        for edge in itertools.combinations(group, 2):
            all_edges.add(tuple(sorted(edge)))
    num_edges = len(all_edges)

    return val_score, num_edges

def optimize_graph(X_train, y_train, X_val, y_val, n_trials, initial_constraints, initial_labels):
    if params.VERBOSE:
        print(f"Dataset: {params.DATASET_NAME}")
        print(f"Total features: {len(X_train.columns)}")
        print(f"Initial constraint groups: {len(initial_constraints)}")

    col_to_idx = {col: idx for idx, col in enumerate(X_train.columns)}
    study = optuna.create_study(directions=['maximize', 'minimize'])
    study.optimize(lambda t: objective(t, X_train, y_train, X_val, y_val, col_to_idx, initial_constraints), n_trials=n_trials, show_progress_bar=True, n_jobs=-1)

    selected_trial = max(study.best_trials, key=lambda t: t.values[0] - t.values[1] * params.EDGE_PENALTY)
    if params.VERBOSE:
        print(f"\nSelected trial with: Score={selected_trial.values[0]:.4f}, Edges={int(selected_trial.values[1])}")
    if params.PLOT_IMAGES:
        visualize_pareto_front(study, params.DATASET_NAME, selected_trial)

    # Reconstruct selected constraints
    selected_constraints = []
    selected_labels = []
    for i, group in enumerate(initial_constraints):
        selected_features = []
        for feature in group:
            include_feature = selected_trial.params.get(f'{i}_{feature}', 0.5)
            if include_feature:
                selected_features.append(feature)
        
        # Only add groups with more than 1 feature
        if len(selected_features) > 1:
            selected_constraints.append(selected_features)
            selected_labels.append(initial_labels[i])

    if params.VERBOSE:
        print(f"Selected {len(selected_constraints)} out of {len(initial_constraints)} constraint groups")
        print("Selected constraint group sizes:")
        for i, group in enumerate(selected_constraints):
            print(f"  Group {i+1}: {len(group)} features")

    optimized_kg = nx.Graph()
    for feature in X_train.columns:
        optimized_kg.add_node(feature)
    for group in selected_constraints:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                optimized_kg.add_edge(group[i], group[j])
    return selected_constraints, selected_labels, selected_trial.params