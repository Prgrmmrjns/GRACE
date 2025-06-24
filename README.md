# GRACE: Graph-based Complexity Reduction and Context-enhanced Explainability

## 🎯 Project Goal

The GRACE project aims to process a given dataset into a graph structure where nodes represent dataset features and edges represent possible feature interactions. We use this graph to constrain an XGBoost model, with two primary objectives:

1.  **Improve ML Performance**: By providing the model with domain-informed or empirically discovered feature interactions, we can guide it towards better performance.
2.  **Enhance Explainability & Reduce Complexity**: By simplifying the graph structure to a minimal set of nodes and edges, we create a more interpretable and less complex model.

## 🛠️ How It Works

The workflow is as follows:

1.  **Initial Graph Creation**: An initial knowledge graph is created. This can be done manually, through an automated agent (`create_kg.py`), or by loading a pre-existing graph. The initial graph is based on feature importance (SHAP-IQ) and known biological/domain mechanisms.
2.  **Graph Optimization**: The core of the project is in `graph_reduction.py`. We use a multi-objective optimization process with [Optuna](https://optuna.org/) to iteratively refine the graph. The optimization seeks to find a Pareto front of graphs that are optimal in terms of both predictive performance (e.g., AUC or Accuracy) and simplicity (number of nodes and edges).
3.  **Constrained Model Training**: The optimized graph structure is used to generate `interaction_constraints` for an `XGBoost` classifier. This forces the model to only consider interactions between features connected by an edge in the graph.
4.  **Evaluation**: The final constrained model is evaluated on a test set to measure its performance.

## 🚀 Getting Started

### 1. Prerequisites

-   Python 3.10+
-   A virtual environment (e.g., `venv` or `conda`) is highly recommended.

### 2. Installation

Clone the repository to your local machine:
```bash
git clone <repository-url>
cd GRACE
```

Create and activate a virtual environment. For example, with `venv`:
```bash
python -m venv venv
source venv/bin/activate
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Configuration

The project requires API keys for an LLM provider (like OpenAI) for the agent-based graph creation.

1.  Create a `.env` file in the root of the project directory:
    ```
    OPENAI_API_KEY="your-api-key-here"
    ```
2.  Edit the `params.py` file to configure the project:
    *   Set `DATASET_NAME` to either `"mimic"` or `"adni"`.
    *   Set `LLM_PROVIDER` to your desired provider (e.g., `"openai"`).

### 4. Running the Project

Execute the main script from the root directory:
```bash
python main.py
```

The script will load the data, run the graph optimization process, train the final model, and save the results and visualizations in the `images/` and `models/` directories.

## 📁 Project Structure

```
GRACE/
├── datasets/         # Raw CSV datasets
├── kg/               # Knowledge Graphs (GraphML) and agent outputs
├── models/           # Saved trained model files
├── images/           # Saved plots and visualizations
├── main.py           # Main script to run the full pipeline
├── graph_reduction.py# Core logic for graph optimization using Optuna
├── create_kg.py      # Script for agent-based initial KG creation
├── visualizations.py # Functions for plotting results
├── utils.py          # Utility functions for graph manipulation
├── params.py         # All user-configurable parameters
└── README.md         # This file
```
