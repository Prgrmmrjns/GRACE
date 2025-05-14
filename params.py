DATASET_NAME = "mimic"
DATASET_PATH = f"datasets/{DATASET_NAME}.csv"
TARGET_COL = "DIAGNOSIS" if DATASET_NAME == "adni" else "mortality_flag"
STUDY_TYPE = "Retrospective"
TARGET_COL_DICT = {1: "Normal Cognitive Function", 2: "Mild Cognitive Impairment", 3: "Alzheimer's Disease"} if DATASET_NAME == "adni" else {0: "Survived", 1: "Died"}
KEYWORDS = ['Alzheimers disease', 'Cognitive impairment', 'Mild cognitive impairment', 'Depression', 'Anxiety', 'Risk factors Alzheimer', 'Medication Alzheimer'] if DATASET_NAME == 'adni' else ['intensive_care mortality', 'risk factors intensive_care', 'sepsis', 'mechanical_ventilation', 'infection'] 
METRIC = "accuracy" if DATASET_NAME == "adni" else "auc"
LLM_PROVIDER = 'openai'  # Set to 'ollama' or 'openai'
MODEL = "qwen3:30b" if LLM_PROVIDER == 'ollama' else "gpt-4.1-mini"
VECTOR_DB = 'PGVECTOR' 
LOAD_AGENT_RESPONSES = False
LOAD_KG = True
VISUALIZE_KG = True
LOAD_OPTIMIZATION_RESULTS = True

# Define your own model or use different parameters
from catboost import CatBoostClassifier

CATBOOST_PARAMS = {
    'iterations': 1000,
    'learning_rate': 0.1,
    'depth': 3, 
    'l2_leaf_reg': 10, 
    'early_stopping_rounds': 20, 
    'verbose': 0, 
    'random_state': 42,
    'allow_writing_files': False,
}

def get_model():
    return CatBoostClassifier(**CATBOOST_PARAMS)

# Define fit function - may differ for different models
def model_fit(model, X_train, y_train, X_val=None, y_val=None):
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)