# 🧠 GRACE: Graph-based Dimensionality Reduction & Context-Enhanced Explainability

Welcome to **GRACE**! This project leverages knowledge graphs and large language models to both reduce dataset dimensionality and provide context-rich, clinically meaningful explanations for machine learning predictions.

## 🎯 Aim of the Project

- **Dimensionality Reduction:** Remove irrelevant features by leveraging knowledge graphs derived from scientific articles, ensuring only the most causally relevant features are used for prediction.
- **Enhanced Explainability:** Use domain-specific knowledge graphs and LLMs to generate personalized, context-aware explanations for each prediction, making results more interpretable for clinicians, researchers, and patients.

## 🚀 How to Use

1. **Configure your experiment:**
   - Edit `params.py` to set:
     - `DATASET_PATH`: Path to your dataset CSV file
     - `TARGET_COL`: Name of the target column
     - `PREDICTION_TASK`: Name of the prediction task (e.g., 'Mortality Risk', 'Diagnosis')
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the main script:**
   ```bash
   python main.py
   ```
4. **Interpret results:**
   - Patient-specific interpretations and edge annotations will be saved in the `patient_interpretations/` directory.
   - Visualizations and further analysis can be generated using the provided scripts.

## 🗂️ Project Structure

- `params.py` — Main configuration for dataset, target, and prediction task
- `main.py` — Entry point, calls all major functions
- `visualizations.py` — Image generation and plotting
- `train.py` — Dimensionality reduction logic
- `interpretability.py` — Explainability logic (now includes LLM models)
- `build_kg.py` — Knowledge graph generation from scientific articles
- `create_graph_structure.py` — Validated graph structure creation
- `graph_utils.py` — Graph structure utilities
- `create_prompt.py` — Custom prompt engineering for Lightrag
- `baselines.py` — Baseline model comparisons

## 📋 To Dos

- [ ] 🔗 **Improved graph creation:** Allow input nodes to connect to multiple intermediate nodes (e.g., CRP → cardiovascular & respiratory systems)
- [ ] ✂️ **Revise pruning method:** Prevent removal of predictive features, especially for the MIMIC dataset
- [ ] 🤖 **Support more LLM providers:** Add Ollama and other model providers (currently only OpenAI is supported)
- [ ] 🩺 **Refined interpretability:** Make explanations more structured and useful for clinicians, researchers, and patients

## 💡 Get Involved

Contributions, suggestions, and feedback are welcome! Feel free to open issues or submit pull requests.

---

Made with ❤️ for explainable AI in healthcare.
