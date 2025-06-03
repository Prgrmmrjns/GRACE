from agno.models.openai import OpenAIChat
from agno.models.ollama import Ollama
from agno.models.azure import AzureOpenAI
from agno.embedder.openai import OpenAIEmbedder
from agno.embedder.ollama import OllamaEmbedder
import os
from dotenv import load_dotenv
import xgboost as xgb

## Dataset configuration
DATASET_NAME = "mimic"
DATASET_PATH = f"datasets/{DATASET_NAME}.csv"
TARGET_COL = "DIAGNOSIS" if DATASET_NAME == "adni" else "mortality_flag"
TARGET_COL_DICT = {1: "Normal Cognitive Function", 2: "Mild Cognitive Impairment", 3: "Alzheimer's Disease"} if DATASET_NAME == "adni" else {0: "Survived", 1: "Died"}

## KG creation configuration
LLM_PROVIDER = 'azureopenai'  # Set to 'ollama', 'openai', or 'azureopenai'
VECTOR_DB = 'LanceDB' 

## ML configuration
if LLM_PROVIDER == "azureopenai":
    load_dotenv()
    os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
    os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT") 
    os.environ["OPENAI_API_VERSION"] = os.getenv("OPENAI_API_VERSION")
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    LLM_MODEL = AzureOpenAI(id="gpt-4.1", temperature=0)
    EMBEDDING_MODEL = OpenAIEmbedder()
elif LLM_PROVIDER == "openai":
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    LLM_MODEL = OpenAIChat(id="gpt-4.1-nano", temperature=0)
    EMBEDDING_MODEL = OpenAIEmbedder(id="text-embedding-3-small")
else:
    LLM_MODEL = Ollama(id="devstral")
    EMBEDDING_MODEL = OllamaEmbedder(id="nomic-embed-text")

METRIC = "accuracy" if DATASET_NAME == "adni" else "auc"

## Data split configuration
TEST_SIZE = 0.3  # Proportion of data for test set
VAL_SIZE = 0.2   # Proportion of remaining data for validation set (after test split)

# XGBoost parameters (similar to LightGBM)
XGBOOST_PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': 3,
    'reg_lambda': 1,
    'subsample': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0,  
}

EARLY_STOPPING_ROUNDS = 20

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

ML_MODEL = xgb.XGBClassifier(**XGBOOST_PARAMS)

## Visualization configuration
VISUALIZE_KG = True
LOAD_KG = True
LOAD_OPTIMIZATION_RESULTS = False

# Optimization ranges
MIN_SHAP_THRESHOLD_RANGE = (0, 0.05)  # Broader range for SHAP threshold
MIN_INTERACTION_THRESHOLD_RANGE = (0, 0.2)  # Broader range for interaction threshold

# Optimization penalty coefficients
FEATURE_PENALTY_COEFF = 0.00003  # Penalty coefficient for number of features
EDGE_PENALTY_COEFF = 0.000003  # Penalty coefficient for number of edges

N_TRIALS = 100  # Reduced for faster testing