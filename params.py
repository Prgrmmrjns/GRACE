from agno.models.openai import OpenAIChat
from agno.models.ollama import Ollama
from agno.models.azure import AzureOpenAI
from agno.embedder.openai import OpenAIEmbedder
from agno.embedder.ollama import OllamaEmbedder
import os
import lightgbm as lgb
from dotenv import load_dotenv

## Dataset configuration
# DATASET_NAME = "adni"
# DATASET_NAME = "adni"
# DATASET_NAME = 'mimic'
# DATASET_NAME = 'adni'
DATASET_NAME = 'adni'
DATASET_PATH = f"datasets/{DATASET_NAME}.csv"
TARGET_COL = "DIAGNOSIS" if DATASET_NAME == "adni" else "mortality_flag"
TARGET_COL_DICT = {1: "Normal Cognitive Function", 2: "Mild Cognitive Impairment", 3: "Alzheimer's Disease"} if DATASET_NAME == "adni" else {0: "Survived", 1: "Died"}
DATASET_INFO = f'dataset_info/{DATASET_NAME}_info.txt'

## KG creation configuration
LLM_PROVIDER = 'openai'  # Set to 'ollama', 'openai', or 'azureopenai'

## ML configuration
if LLM_PROVIDER == "azureopenai":
    load_dotenv()
    os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
    os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT") 
    #os.environ["OPENAI_API_VERSION"] = os.getenv("OPENAI_API_VERSION")
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    LLM_MODEL = AzureOpenAI(id="gpt-4.1")
    EMBEDDING_MODEL = OpenAIEmbedder()
elif LLM_PROVIDER == "openai":
    load_dotenv()
    os.environ['OPENAI_API_VERSION']
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    LLM_MODEL = OpenAIChat(id="gpt-4.1")
    EMBEDDING_MODEL = OpenAIEmbedder(id="text-embedding-3-small")
else:
    LLM_MODEL = Ollama(id="mistral-small3.2")
    EMBEDDING_MODEL = OllamaEmbedder(id="nomic-embed-text")

## Data split configuration
TEST_SIZE = 0.3  # Proportion of data for test set
VAL_SIZE = 0.2   # Proportion of remaining data for validation set (after test split)

# LightGBM parameters
PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.1,
    'max_depth': 2,
    'n_jobs': 1,
    'random_state': 42,
    'data_sample_strategy': 'goss',
    'use_quantized_grad': True,
    'force_col_wise': True,
    'early_stopping_rounds': 20,
    'verbosity': -1,
}

if DATASET_NAME == "adni":
    KEYWORDS = ["Alzheimer's Disease", "MCI", "Cognitive Impairment", "Neurodegenerative Disease", "Neuroimaging"]
    PARAMS['objective'] = 'multiclass'
    PARAMS['num_class'] = 3
    METRIC = 'accuracy'
    PREDICT_FN = lambda m, d: m.predict(d)
else:
    KEYWORDS = ["Intensive Care", "Mortality", "ICU", "Critical Care"]  
    PARAMS['objective'] = 'binary'
    METRIC = 'roc_auc'
    PREDICT_FN = lambda model, X: model.predict_proba(X)[:, 1]

ML_MODEL = lgb.LGBMClassifier(**PARAMS)

N_TRIALS = 100
EDGE_PENALTY = 0.00002

# Feature inclusion probability coefficients
INCLUSION_BASE_PROB = 0.1
INCLUSION_IMPORTANCE_SCALE = 0.8

# KG loading configuration
LOAD_AGENT_KG = True
PLOT_IMAGES = False
VERBOSE = True
AGENT_KG_PATH = f"kg/{DATASET_NAME}_initial_agent_kg.graphml"