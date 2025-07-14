from agno.models.openai import OpenAIChat
from agno.models.ollama import Ollama
from agno.models.azure import AzureOpenAI
from agno.models.lmstudio import LMStudio
import os
import lightgbm as lgb
from dotenv import load_dotenv

## Dataset configuration
DATASET_NAME = 'adni'
DATASET_PATH = f"datasets/{DATASET_NAME}.csv"
TARGET_COL = "DIAGNOSIS" if DATASET_NAME == "adni" else "mortality_flag"
TARGET_COL_DICT = {1: "Normal Cognitive Function", 2: "Mild Cognitive Impairment", 3: "Alzheimer's Disease"} if DATASET_NAME == "adni" else {0: "Survived", 1: "Died"}
DATASET_INFO = f'dataset_info/{DATASET_NAME}_info.txt'

## KG creation configuration
LLM_PROVIDER = 'lmstudio'  # Set to 'ollama', 'openai', or 'azureopenai'

## LLM configuration
if LLM_PROVIDER == "azureopenai":
    load_dotenv()
    os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
    os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT") 
    #os.environ["OPENAI_API_VERSION"] = os.getenv("OPENAI_API_VERSION")
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    LLM_MODEL = AzureOpenAI(id="gpt-4.1")
elif LLM_PROVIDER == "openai":
    load_dotenv()
    os.environ['OPENAI_API_VERSION']
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    LLM_MODEL = OpenAIChat(id="o4-mini")
elif LLM_PROVIDER == "lmstudio":
    LLM_MODEL = LMStudio(id="medgemma-27b-it")
else:
    LLM_MODEL = Ollama(id="devstral")

## Data split configuration
TEST_SIZE = 0.3  
VAL_SIZE = 0.5   

# LightGBM parameters
HPARAMS = {
    "max_depth": (2, 10),
    "reg_alpha": (0, 30),
    "reg_lambda": (0, 30),
    #"subsample": (0.5, 1),
    #"colsample_bytree": (0.5, 1),
    #"min_split_gain": (0, 1),
    #"min_child_weight": (1, 20),
    #"num_leaves": (5, 40),
    #"learning_rate": (0.01, 0.1),
}
PARAMS = {
    "max_depth": 3,
    "subsample": 0.7,
    'n_estimators': 1000,
    'n_jobs': 1,
    'random_state': 42,
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

# Optimization configuration
N_TRIALS = 1000
EDGE_PENALTY = 0.00002

# Loading configuration
USE_KNOWLEDGE_BASE = False
LOAD_AGENT_KG = False
PLOT_IMAGES = True
VERBOSE = True
EXPLAIN_RESULTS = False

# Explanation configuration
USER_AIM = """I am a biomedical researcher trying to understand the feature interactions leading to mortality during ICU stay. Please explain me which feature interactions are most important for mortality prediction."""