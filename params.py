from agno.models.openai import OpenAIChat
from agno.models.ollama import Ollama
from agno.models.azure import AzureOpenAI
from agno.embedder.openai import OpenAIEmbedder
from agno.embedder.azure_openai import AzureOpenAIEmbedder
from agno.embedder.ollama import OllamaEmbedder
import os
from dotenv import load_dotenv

import lightgbm as lgb
import xgboost as xgb

## Dataset configuration
DATASET_NAME = "mimic"
DATASET_PATH = f"datasets/{DATASET_NAME}.csv"
TARGET_COL = "DIAGNOSIS" if DATASET_NAME == "adni" else "mortality_flag"
TARGET_COL_DICT = {1: "Normal Cognitive Function", 2: "Mild Cognitive Impairment", 3: "Alzheimer's Disease"} if DATASET_NAME == "adni" else {0: "Survived", 1: "Died"}

## KG creation configuration
LLM_PROVIDER = 'azureopenai'  # Set to 'ollama', 'openai', or 'azureopenai'
VECTOR_DB = 'LanceDB' 
LOAD_KG = True

## ML configuration
if LLM_PROVIDER == "azureopenai":
    load_dotenv()
    os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
    os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT") 
    os.environ["OPENAI_API_VERSION"] = os.getenv("OPENAI_API_VERSION")
    LLM_MODEL = AzureOpenAI(id="gpt-4.1-nano", temperature=0)
    EMBEDDING_MODEL = AzureOpenAIEmbedder()
elif LLM_PROVIDER == "openai":
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    LLM_MODEL = OpenAIChat(id="gpt-4.1-nano", temperature=0)
    EMBEDDING_MODEL = OpenAIEmbedder(id="text-embedding-3-large")
else:
    LLM_MODEL = Ollama(id="devstral")
    EMBEDDING_MODEL = OllamaEmbedder(id="nomic-embed-text")

METRIC = "accuracy" if DATASET_NAME == "adni" else "auc"

LOAD_OPTIMIZATION_RESULTS = True

## Data split configuration
TEST_SIZE = 0.3  # Proportion of data for test set
VAL_SIZE = 0.2   # Proportion of remaining data for validation set (after test split)

## Model configuration
MODEL_TYPE = "xgboost"  # Set to 'lightgbm' or 'xgboost'

# LightGBM parameters
LIGHTGBM_PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.1,
    'max_depth': 3,
    'reg_lambda': 20,
    'random_state': 42,
    'use_quantized_grad': True,
    'data_sample_strategy': 'goss',
    'n_jobs': -1,
    'verbose': -1
}

# XGBoost parameters (similar to LightGBM)
XGBOOST_PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'subsample': 0.2,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0,  # XGBoost uses verbosity instead of verbose
}

# Set objective and eval_metric based on dataset
if DATASET_NAME == "adni":  # multiclass
    XGBOOST_PARAMS.update({
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss'
    })
else:  # binary classification
    XGBOOST_PARAMS.update({
        'objective': 'binary:logistic', 
        'eval_metric': 'auc'
    })

# Select model based on MODEL_TYPE
if MODEL_TYPE == "lightgbm":
    ML_MODEL = lgb.LGBMClassifier(**LIGHTGBM_PARAMS)
    EARLY_STOPPING_CALLBACK = lgb.early_stopping(20, verbose=False)
elif MODEL_TYPE == "xgboost":
    ML_MODEL = xgb.XGBClassifier(**XGBOOST_PARAMS)
    EARLY_STOPPING_CALLBACK = xgb.callback.EarlyStopping(rounds=20, save_best=True)

def fit_model(model, X_train, y_train, X_val, y_val, early_stopping_callback):
    """Fit model with early stopping for both LightGBM and XGBoost."""
    if MODEL_TYPE == "lightgbm":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[early_stopping_callback])
    elif MODEL_TYPE == "xgboost":
        # XGBoost: Set early stopping via model parameters and fit
        model.set_params(early_stopping_rounds=20)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model

## Visualization configuration
VISUALIZE_KG = True
EXPLAIN_WITH_LLM = False

# Optimization ranges
MIN_SHAP_THRESHOLD_RANGE = (1e-8, 0.1)  # Broader range for SHAP threshold
MIN_INTERACTION_THRESHOLD_RANGE = (1e-10, 0.05)  # Broader range for interaction threshold

# Optimization penalty coefficients
FEATURE_PENALTY_COEFF = 0.0005  # Penalty coefficient for number of features
EDGE_PENALTY_COEFF = 0.0001  # Penalty coefficient for number of edges

N_TRIALS = 100