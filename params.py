DATASET_NAME = "adni"
DATASET_PATH = f"datasets/{DATASET_NAME}.csv"
TARGET_COL = "DIAGNOSIS" if DATASET_NAME == "adni" else "mortality_flag"
KEYWORDS = ['Alzheimers disease', 'Cognitive impairment', 'Mild cognitive impairment', 'Depression', 'Anxiety', 'Risk factors Alzheimer', 'Medication Alzheimer'] if DATASET_NAME == 'adni' else ['intensive_care mortality', 'risk factors intensive_care', 'sepsis', 'mechanical_ventilation', 'infection'] 
METRIC = "accuracy" if DATASET_NAME == "adni" else "auc"
LLM_PROVIDER = 'ollama'  # Set to 'ollama' or 'openai'
MODEL = "qwen3:30b" if LLM_PROVIDER == 'ollama' else "gpt-4.1-mini"
EMBEDDING_MODEL = "nomic-embed-text:latest"
VECTOR_DB = 'PGVECTOR' 
LOAD_AGENT_RESPONSES = False
LOAD_KG = True
VISUALIZE_KG = True

# Import CatBoost
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# --- Model Selection --- 
# Set this to 'catboost', 'xgboost', 'glm', 'svm', 'naive_bayes', or 'sgd' to choose the model
MODEL_TYPE = 'catboost' 
# ----------------------

# CatBoost Parameters 
CATBOOST_PARAMS = {
    # 'loss_function': 'Logloss' if METRIC == 'auc' else 'MultiClass', # Often inferred
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 3, 
    #'l2_leaf_reg': 10, 
    'early_stopping_rounds': 20, 
    'verbose': 0, 
    'random_state': 42,
}

# GLM (Logistic Regression) Parameters
GLM_PARAMS = {
    'penalty': 'l2', # Regularization type ('l1', 'l2', 'elasticnet', None)
    'C': 1.0, # Inverse of regularization strength (smaller values mean stronger regularization)
    'solver': 'liblinear', # Restore solver ('liblinear' is good for small datasets and l1/l2)
    'max_iter': 1000, # Increase max iterations for convergence
    'random_state': 42,
    # Note: multi_class='auto' is default and usually works.
}

# SVM (Support Vector Classifier) Parameters
SVM_PARAMS = {
    'C': 1.0, # Regularization parameter
    'kernel': 'rbf', # Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
    'gamma': 'scale', # Kernel coefficient ('scale', 'auto' or float)
    'probability': True, # MUST be True to use predict_proba
    'random_state': 42,
    # Note: SVMs can be computationally expensive, especially with non-linear kernels.
    # They also don't use early_stopping_rounds, learning_rate, depth etc.
}

# Gaussian Naive Bayes Parameters (usually none needed)
NAIVE_BAYES_PARAMS = {}

# SGD Classifier Parameters
SGD_PARAMS = {
    'loss': 'log_loss', # 'log_loss' for logistic regression, 'hinge' for linear SVM
    'penalty': 'l2', # Regularization type ('l1', 'l2', 'elasticnet')
    'alpha': 0.0001, # Regularization strength
    'max_iter': 1000, # Max passes over the training data
    'tol': 1e-3, # Stopping criterion tolerance
    'n_jobs': -1, # Use all available CPU cores for some solvers (if applicable)
    'random_state': 42,
    # Needs predict_proba, which SGDClassifier provides if loss is log_loss or modified_huber
    # Note: SGD is sensitive to feature scaling (already handled in main.py)
}

def get_base_model():
    if MODEL_TYPE == 'catboost':
        print(f"Using CatBoost model with params: {CATBOOST_PARAMS}")
        return CatBoostClassifier(**CATBOOST_PARAMS)
    elif MODEL_TYPE == 'glm':
        print(f"Using GLM (Logistic Regression) model with params: {GLM_PARAMS}")
        return LogisticRegression(**GLM_PARAMS)
    elif MODEL_TYPE == 'svm':
        print(f"Using SVM (SVC) model with params: {SVM_PARAMS}")
        return SVC(**SVM_PARAMS)
    elif MODEL_TYPE == 'naive_bayes':
        print(f"Using GaussianNB model with params: {NAIVE_BAYES_PARAMS}")
        return GaussianNB(**NAIVE_BAYES_PARAMS)
    elif MODEL_TYPE == 'sgd':
        print(f"Using SGDClassifier model with params: {SGD_PARAMS}")
        return SGDClassifier(**SGD_PARAMS)
    else:
        raise ValueError(f"Unsupported MODEL_TYPE: {MODEL_TYPE}. Choose 'catboost', 'xgboost', 'glm', 'svm', 'naive_bayes', or 'sgd'.")

# Callbacks are generally not needed for basic early stopping with CatBoost/XGBoost
# as it's handled via parameters or fit arguments.

# --- Model Fitting Function --- 
def model_fit(model, X_train, y_train, X_val=None, y_val=None):
    """Fits the model using appropriate arguments based on MODEL_TYPE."""
    if MODEL_TYPE == 'catboost':
        # These models support eval_set for early stopping (configured in params)
        if X_val is not None and y_val is not None:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else: # Should not happen in current usage, but handle just in case
            model.fit(X_train, y_train, verbose=False)
            
    elif MODEL_TYPE in ['glm', 'svm', 'naive_bayes', 'sgd']:
        # LogisticRegression & SVC don't use eval_set or verbose in fit
        model.fit(X_train, y_train)
    else:
        # Fallback or raise error for other types if added later
        print(f"Warning: model_fit not specifically implemented for MODEL_TYPE '{MODEL_TYPE}'. Using basic fit.")
        model.fit(X_train, y_train)
# ---------------------------
