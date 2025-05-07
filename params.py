# Parameters for GRACE project configuration

# Dataset configuration
DATASET_NAME = "mimic"  # Options: adni or mimic
DATASET_PATH = f"datasets/{DATASET_NAME}.csv"
TARGET_COL = "DIAGNOSIS" if DATASET_NAME == "adni" else "mortality_flag"

# Keywords for knowledge extraction
KEYWORDS = ['Alzheimers disease', 'Cognitive impairment', 'Mild cognitive impairment', 'Depression', 'Anxiety', 'Risk factors Alzheimer', 'Medication Alzheimer'] if DATASET_NAME == 'adni' else ['intensive_care mortality', 'risk factors intensive_care', 'sepsis', 'mechanical_ventilation', 'infection'] 

# ML model configuration
METRIC = "accuracy" if DATASET_NAME == "adni" else "auc"

# LLM provider: 'ollama' or 'openai'
LLM_PROVIDER = 'ollama'  # Set to 'ollama' or 'openai'
MODEL = "qwen3:30b" if LLM_PROVIDER == 'ollama' else "gpt-4.1-mini"
EMBEDDING_MODEL = "nomic-embed-text:latest"
# suggested openai models that work with tool calling: gpt-4.1-mini, gpt-4.1-nano, gpt-4.1
# suggested ollama models that work with tool calling: cogito:14b, mistral-small3.1

VERBOSE = False

VECTOR_DB = 'PGVECTOR' # 'PGVECTOR' or 'MILVUS' PGVector needs setup with Docker but offers better performance

# JSON agent response loading: when True, reuse stored agent responses via JsonStorage
LOAD_AGENT_RESPONSES = False  # if True load agent responses from JSON storage
LOAD_KG = True
LOAD_OPTIMIZED_MODELS = True # When True, load optimized models from step 1 instead of re-running optimization
VISUALIZE_KG = True

# ML base model definition
import lightgbm as lgb

# Default model params; users may modify here
MODEL_PARAMS = {
    'objective': 'binary' if METRIC == 'auc' else 'multiclass',
    'n_estimators': 1000,
    'learning_rate': 0.1,
    'reg_lambda': 10,
    'path_smooth': 1.0,
    'min_child_samples': 20,
    'data_sample_strategy': 'goss',
    'use_quantized_grad': True,
    'n_workers': -1,
    'verbosity': -1,
    'random_state': 42,
}

def get_base_model():
    return lgb.LGBMClassifier(**MODEL_PARAMS)

def get_callbacks():
    return [lgb.early_stopping(stopping_rounds=10, verbose=False)]
