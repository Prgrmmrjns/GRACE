from agno.models.openai import OpenAIChat
from agno.models.ollama import Ollama
from agno.models.azure import AzureOpenAI
from agno.embedder.openai import OpenAIEmbedder
from agno.embedder.ollama import OllamaEmbedder
import os
import lightgbm as lgb
from dotenv import load_dotenv

## Dataset configuration
DATASET_NAME = "mimic"
DATASET_PATH = f"datasets/{DATASET_NAME}.csv"
TARGET_COL = "DIAGNOSIS" if DATASET_NAME == "adni" else "mortality_flag"
TARGET_COL_DICT = {1: "Normal Cognitive Function", 2: "Mild Cognitive Impairment", 3: "Alzheimer's Disease"} if DATASET_NAME == "adni" else {0: "Survived", 1: "Died"}

## KG creation configuration
LLM_PROVIDER = 'azureopenai'  # Set to 'ollama', 'openai', or 'azureopenai'
LOAD_CACHE = False # Set to True to load all cached data (LLM, Optuna, SHAP)
LLM_CACHE_DIR = "llm_cache"
OPTUNA_CACHE_DIR = "optuna_cache"
SHAP_CACHE_DIR = "shap_cache"

## ML configuration
if LLM_PROVIDER == "azureopenai":
    load_dotenv()
    os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
    os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT") 
    os.environ["OPENAI_API_VERSION"] = os.getenv("OPENAI_API_VERSION")
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    LLM_MODEL = AzureOpenAI(id="gpt-4.1-mini", temperature=0)
    EMBEDDING_MODEL = OpenAIEmbedder()
elif LLM_PROVIDER == "openai":
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    LLM_MODEL = OpenAIChat(id="gpt-4.1-mini", temperature=0)
    EMBEDDING_MODEL = OpenAIEmbedder(id="text-embedding-3-small")
else:
    LLM_MODEL = Ollama(id="magistral")
    EMBEDDING_MODEL = OllamaEmbedder(id="nomic-embed-text")

METRIC = 'auc' # or 'accuracy'

## Data split configuration
TEST_SIZE = 0.3  # Proportion of data for test set
VAL_SIZE = 0.2   # Proportion of remaining data for validation set (after test split)

N_TRIALS = 100 # Reduced for faster testing

LOAD_LLM_CACHE = False # Set to True to load cached LLM responses
LOAD_OPT_CACHE = False # Set to True to load all cached data (Optuna, SHAP)


# Lightgbm parameters
PARAMS = {
    'n_estimators': 2000,
    'max_depth': 3,
    'learning_rate': 0.1,
    'reg_lambda': 20,
    'random_seed': 42,
    'early_stopping_round': 100,
    'data_sample_strategy': 'goss',
    'use_quantized_grad': True,
    'verbose': -1
}

if DATASET_NAME == "adni":
    PARAMS['objective'] = 'multiclass'
    PARAMS['num_class'] = 3
    METRIC = 'accuracy'
    PREDICT_FN = lambda m, d: m.predict(d)
else:
    METRIC = 'auc'
    PREDICT_FN = lambda m, d: m.predict_proba(d)[:, 1]

ML_MODEL = lgb.LGBMClassifier(**PARAMS)
