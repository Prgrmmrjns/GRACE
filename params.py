from agno.models.openai import OpenAIChat
from agno.models.ollama import Ollama
from agno.models.azure import AzureOpenAI
from agno.embedder.openai import OpenAIEmbedder
from agno.embedder.ollama import OllamaEmbedder
import os
import lightgbm as lgb
from dotenv import load_dotenv

## Dataset configuration
DATASET_NAME = "adni"
DATASET_PATH = f"datasets/{DATASET_NAME}.csv"
TARGET_COL = "DIAGNOSIS" if DATASET_NAME == "adni" else "mortality_flag"
TARGET_COL_DICT = {1: "Normal Cognitive Function", 2: "Mild Cognitive Impairment", 3: "Alzheimer's Disease"} if DATASET_NAME == "adni" else {0: "Survived", 1: "Died"}

## KG creation configuration
LLM_PROVIDER = 'openai'  # Set to 'ollama', 'openai', or 'azureopenai'

## ML configuration
if LLM_PROVIDER == "azureopenai":
    load_dotenv()
    os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
    os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT") 
    #os.environ["OPENAI_API_VERSION"] = os.getenv("OPENAI_API_VERSION")
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    LLM_MODEL = AzureOpenAI(id="gpt-4.1-mini", temperature=0)
    EMBEDDING_MODEL = OpenAIEmbedder()
elif LLM_PROVIDER == "openai":
    load_dotenv()
    os.environ['OPENAI_API_VERSION']
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    LLM_MODEL = OpenAIChat(id="gpt-4.1-mini", temperature=0)
    EMBEDDING_MODEL = OpenAIEmbedder(id="text-embedding-3-small")
else:
    LLM_MODEL = Ollama(id="qwen3:32b")
    EMBEDDING_MODEL = OllamaEmbedder(id="nomic-embed-text")

## Data split configuration
TEST_SIZE = 0.3  # Proportion of data for test set
VAL_SIZE = 0.1   # Proportion of remaining data for validation set (after test split)


# LGBM parameters
PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.1,
    'max_depth': 3,
    'reg_lambda': 10,
    'subsample': 0.8,
    'use_quantized_grad': True,
    'verbose': -1
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
    METRIC = 'auc'
    PREDICT_FN = lambda model, X: model.predict_proba(X)[:, 1]

ML_MODEL = lgb.LGBMClassifier(**PARAMS)
CALLBACKS = [lgb.early_stopping(stopping_rounds=100, verbose=False)]

# KG loading configuration
LOAD_AGENT_KG = True
AGENT_KG_PATH = f"kg/{DATASET_NAME}_initial_agent_kg.graphml"
