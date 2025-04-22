# Parameters for GRACE project configuration

# Path to the dataset CSV file
DATASET_PATH = 'datasets/mimic.csv'  # Change to your dataset path
DATASET_NAME = 'mimic'
# Name of the target column in the dataset
TARGET_COL = 'mortality_flag'  # Change to your target column

# Name of the prediction task (for context and explainability)
PREDICTION_TASK = 'Mortality Risk'  # e.g., 'Mortality Risk', 'Diagnosis', etc.
METRIC = 'auc'

# LLM provider: 'ollama' or 'openai'
LLM_PROVIDER = 'openai'  # Set to 'ollama' or 'openai'

MODEL = "gemma3:12b" if LLM_PROVIDER == 'ollama' else "gpt-4.1-nano"
# suggested openai models that work with tool calling: gpt-4.1-mini, gpt-4.1-nano, gpt-4.1
# suggested ollama models that work with tool calling: cogito:14b, mistral-small3.1, gemma3:12b
VERBOSE = False
LOAD_KG = True  # if True load KG from existing file instead of rebuilding

# Default settings for automatic literature retrieval used by find_articles.py
ARTICLES_MAX = 200  # maximum number of open‑access articles to fetch automatically
MANUAL_PDF_DIR = None  # optional path to a folder with user‑provided PDFs