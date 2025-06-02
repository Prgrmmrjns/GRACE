# 🧠 GRACE: Graph-based Dimensionality Reduction & Context-Enhanced Explainability

Welcome to **GRACE**! This project leverages knowledge graphs and large language models to both reduce dataset dimensionality and provide context-rich, clinically meaningful explanations for machine learning predictions.

## 🎯 Aim of the Project

- **Dimensionality Reduction:** Remove irrelevant features by leveraging knowledge graphs derived from scientific articles, ensuring only the most causally relevant features are used for prediction.
- **Enhanced Explainability:** Use domain-specific knowledge graphs and LLMs to generate personalized, context-aware explanations for each prediction, making results more interpretable for clinicians, researchers, and patients.
- **Automated Optimization:** Use Optuna to automatically optimize SHAP thresholds for optimal feature selection and model performance.

## 🚀 How to Use

### Basic Setup

1. **Configure your experiment:**
   - Edit `params.py` to set:
     - `DATASET_PATH`: Path to your dataset CSV file
     - `TARGET_COL`: Name of the target column
     - `LLM_PROVIDER`: Determine whether to use OPENAI LLMs or Ollama
     - `MODEL`: Set the model that performs graph creation and interpretation
     - `TEST_SIZE`: Proportion of data for test set (default: 0.3)
     - `VAL_SIZE`: Proportion of remaining data for validation set (default: 0.2)
     - `N_TRIALS`: Number of Optuna optimization trials (default: 500)
     - `MIN_SHAP_THRESHOLD_RANGE`: Range for SHAP threshold optimization
     - `MIN_INTERACTION_THRESHOLD_RANGE`: Range for interaction threshold optimization
     - `FEATURE_PENALTY_COEFF`: Penalty coefficient for number of features (default: 0.1)
     - `EDGE_PENALTY_COEFF`: Penalty coefficient for number of edges (default: 0.05)

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Set up vector database:**
   ```bash
   docker run -d \
     -e POSTGRES_DB=ai \
     -e POSTGRES_USER=ai \
     -e POSTGRES_PASSWORD=ai \
     -e PGDATA=/var/lib/postgresql/data/pgdata \
     -v pgvolume:/var/lib/postgresql/data \
     -p 5532:5432 \
     --name pgvector \
     agnohq/pgvector:16
   ```

### Running the Analysis

4. **Run the full pipeline:**
   ```bash
   python main.py
   ```

## 🔧 SHAP-Based Feature Selection & Optimization

GRACE uses SHAP (SHapley Additive exPlanations) values to intelligently select features and filter knowledge graph edges based on their actual contribution to model predictions.

### Key Features

- **Automated Threshold Optimization:** Uses Optuna to find optimal SHAP and interaction thresholds
- **Multi-Objective Optimization:** Balances validation performance with feature/edge reduction
- **SHAP Interaction Filtering:** Filters knowledge graph edges based on actual SHAP interaction strength
- **Parallel Optimization:** Multi-core optimization for faster hyperparameter tuning
- **Configurable Penalties:** Adjustable coefficients for feature and edge complexity penalties

### How It Works

1. **Initial Model Training:** Train a model with knowledge graph constraints to get SHAP values
2. **Threshold Optimization:** Use Optuna to optimize:
   - `min_shap_threshold`: Minimum SHAP importance for feature selection
   - `min_interaction_threshold`: Minimum SHAP interaction strength for edge filtering
3. **Feature Selection:** Keep only features with SHAP importance above threshold
4. **Edge Filtering:** Keep only edges where connected features have strong SHAP interactions
5. **Final Model:** Train final model with optimized feature set and filtered knowledge graph

### Configuration

Customize optimization in `params.py`:
```python
# Optimization ranges
MIN_SHAP_THRESHOLD_RANGE = (1e-8, 0.5)  # Broader range for SHAP threshold
MIN_INTERACTION_THRESHOLD_RANGE = (1e-10, 0.1)  # Broader range for interaction threshold

# Optimization penalty coefficients
FEATURE_PENALTY_COEFF = 0.1  # Penalty coefficient for number of features
EDGE_PENALTY_COEFF = 0.05    # Penalty coefficient for number of edges

N_TRIALS = 500  # Number of Optuna optimization trials
```

### Output Files

#### Optimization Results (`results/`)
- **Selected features:** `{DATASET_NAME}_selected_features.csv`
- **Optimization results:** `{DATASET_NAME}_results.csv`
  - Contains test score, number of features used, and optimized thresholds

#### Knowledge Graphs (`kg/`)
- **Original graph:** `{DATASET_NAME}.graphml`
- **Filtered graph:** `{DATASET_NAME}_filtered.graphml`

#### Visualizations (`images/`)
- **Original knowledge graph:** `feature_interactions.png`
- **Filtered knowledge graph:** `{DATASET_NAME}_filtered_kg_shap.png`

## 🗂️ Project Structure

### Core Files
- **`main.py`** — Entry point, runs the full pipeline from KG creation to SHAP optimization
- **`params.py`** — Main configuration for dataset, target, models, and optimization parameters
- **`grace_shap.py`** — SHAP calculations, feature selection, and Optuna optimization

### Knowledge Graph & Training
- **`create_kg.py`** — Knowledge graph generation from scientific articles
- **`graph_utils.py`** — Graph structure utilities and feature interaction management

### Visualization & Analysis
- **`visualizations.py`** — Knowledge graph visualizations and plotting functions
- **`baselines.py`** — Baseline model comparisons

### Supporting Files
- **`db.py`** — Database setup and knowledge base management

## 📊 Performance Considerations

### SHAP Optimization
- **Parallel Processing:** Optuna uses `n_jobs=-1` for multi-core optimization
- **Trial Count:** 500 trials typically provide good optimization results
- **Computational Intensity:** SHAP calculation scales with dataset size and model complexity
- **Memory Usage:** SHAP values are cached for efficient optimization

### Optimization Speed
- **Fast Objective Function:** Optimized for speed with pre-computed SHAP values
- **Efficient Edge Counting:** Direct edge counting without full graph creation during optimization
- **Parallel Trials:** Multiple optimization trials run simultaneously
- **Early Stopping:** Median pruner stops unpromising trials early

## 🎯 Multi-Objective Optimization

GRACE optimizes multiple objectives simultaneously:

### Primary Objective
- **Validation Performance:** Maximize accuracy (classification) or AUC (binary classification)

### Secondary Objectives (Penalties)
- **Feature Count:** Minimize number of selected features
- **Edge Count:** Minimize number of knowledge graph edges
- **Configurable Weights:** Adjust penalty coefficients to balance objectives

### Optimization Formula
```
objective = val_score - FEATURE_PENALTY_COEFF * (n_features/100) - EDGE_PENALTY_COEFF * (n_edges/1000)
```

## 🔬 Advanced Configuration

### Custom Optimization Ranges
```python
# Narrow ranges for fine-tuning
MIN_SHAP_THRESHOLD_RANGE = (0.001, 0.1)
MIN_INTERACTION_THRESHOLD_RANGE = (1e-6, 1e-3)

# Broad ranges for exploration
MIN_SHAP_THRESHOLD_RANGE = (1e-8, 0.5)
MIN_INTERACTION_THRESHOLD_RANGE = (1e-10, 0.1)
```

### Penalty Tuning
```python
# Prioritize performance over simplicity
FEATURE_PENALTY_COEFF = 0.01
EDGE_PENALTY_COEFF = 0.005

# Prioritize simplicity over performance
FEATURE_PENALTY_COEFF = 0.5
EDGE_PENALTY_COEFF = 0.2
```

## 📋 To Dos

### High Priority
- [ ] 🔗 **Improved graph creation:** Allow input nodes to connect to multiple intermediate nodes
- [ ] ✂️ **Revise pruning method:** Prevent removal of predictive features, especially for MIMIC dataset
- [ ] 📊 **Multi-objective visualization:** Add Pareto front plots for optimization results

### Medium Priority
- [ ] 🎯 **Bayesian optimization:** Experiment with different Optuna samplers
- [ ] 📈 **Optimization history:** Save and visualize optimization progress
- [ ] 🔄 **Cross-validation:** Add CV-based optimization for more robust results

### Completed
- [x] 🤖 **Support more LLM providers:** Both OpenAI and Ollama supported
- [x] 🎨 **Enhanced visualizations:** Improved knowledge graph plots with consistent styling
- [x] ⚡ **SHAP optimization:** Automated threshold optimization with Optuna
- [x] 🔍 **Interaction filtering:** SHAP-based edge filtering for meaningful connections
- [x] 🎛️ **Configurable parameters:** All optimization parameters in params.py

## ⚡ Performance Optimization

GRACE features intelligent optimization and efficient processing:

### SHAP Optimization (Fast)
- **Pre-computed SHAP values:** Calculate once, use for all optimization trials
- **Efficient edge counting:** Direct counting without full graph creation
- **Parallel trials:** Multi-core optimization with Optuna
- **Early stopping:** Median pruner for faster convergence

### Knowledge Graph Processing
- **Cached graphs:** Save and load processed knowledge graphs
- **Filtered visualizations:** Generate plots only for selected features/edges
- **Optimized constraints:** Build interaction constraints only for filtered features

## 💡 Get Involved

Contributions, suggestions, and feedback are welcome! Feel free to open issues or submit pull requests.

### Contributing Areas
- **Optimization algorithms:** Improve multi-objective optimization strategies
- **SHAP enhancements:** Add new SHAP-based feature selection methods
- **Performance optimization:** Improve optimization speed and memory usage
- **Visualization enhancements:** Create new plot types for optimization results

## 📚 Citation

If you use GRACE in your research, please cite:
```bibtex
@software{grace2024,
  title={GRACE: Graph-based Dimensionality Reduction and Context-Enhanced Explainability},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/grace}
}
```