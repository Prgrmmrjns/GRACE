import os
import json
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Any
from agno.agent import Agent, RunResponse
from agno.knowledge.arxiv import ArxivKnowledgeBase
from agno.storage.json import JsonStorage
from agno.workflow import Workflow
from params import DATASET_NAME, TARGET_COL, LLM_MODEL, DATASET_PATH
from pydantic import BaseModel, Field

class ExplainabilityReport(BaseModel):
    summary: str = Field(description="High-level summary of the model's decision-making process")
    feature_importance_analysis: str = Field(description="Analysis of the most important features and their biological significance")
    interaction_analysis: str = Field(description="Analysis of feature interactions and their biological mechanisms")
    clinical_insights: str = Field(description="Clinical insights and practical implications")
    limitations: str = Field(description="Model limitations and considerations")
    recommendations: str = Field(description="Recommendations for clinical use or further research")

class NoMemoryWorkflow(Workflow):
    def get_workflow_session(self):
        ws = super().get_workflow_session()
        ws.memory = None
        return ws

class ExplainabilityWorkflow(NoMemoryWorkflow):
    description: str = "Generates comprehensive explainability report for the trained model."

    def run(self, model_params: Dict, filtered_kg_info: Dict, shap_analysis: Dict, 
            dataset_info: Dict, arxiv_kb: ArxivKnowledgeBase, recreate_analysis: bool = False):
        
        cache_key = f"explainability_{DATASET_NAME}"
        if not recreate_analysis and self.session_state.get(cache_key):
            cached_report = self.session_state[cache_key]
            yield RunResponse(run_id=self.run_id, content={"step": "explainability", "report": cached_report})
            return
        
        agent = Agent(model=LLM_MODEL, knowledge=arxiv_kb, search_knowledge=True, response_model=ExplainabilityReport)
        
        prompt = f"""As a medical AI expert, provide a comprehensive explainability analysis for a machine learning model trained to predict {TARGET_COL}.

## Model Information:
- Dataset: {DATASET_NAME}
- Target: {TARGET_COL}
- Model Performance: {model_params.get('accuracy', 'N/A')}% accuracy
- Selected Features: {len(shap_analysis['selected_features'])} out of {dataset_info['total_features']} original features
- Knowledge Graph Edges: {filtered_kg_info['edge_count']} interaction constraints

## Selected Features and SHAP Importance:
{self._format_feature_importance(shap_analysis['feature_importance'])}

## Feature Interactions (Knowledge Graph):
{self._format_interactions(filtered_kg_info['interactions'])}

## Dataset Context:
- Total Samples: {dataset_info['total_samples']}
- Feature Types: {dataset_info.get('feature_description', 'Medical/clinical features')}

## Instructions:
Using the scientific literature in the knowledge base, provide a comprehensive analysis that:

1. **Summary**: Explain how the model makes predictions and what biological/medical mechanisms it captures
2. **Feature Importance Analysis**: Interpret the most important features from a medical/biological perspective
3. **Interaction Analysis**: Explain the biological significance of the feature interactions the model uses
4. **Clinical Insights**: Discuss practical implications for clinical decision-making
5. **Limitations**: Address model limitations and potential biases
6. **Recommendations**: Suggest how this model could be used in practice or improved

Focus on:
- Biological plausibility of the selected features and interactions
- Clinical relevance and interpretability
- Potential mechanisms underlying the predictions
- Practical considerations for deployment

Base your analysis on scientific evidence from the literature and ensure medical accuracy."""

        resp = agent.run(prompt)
        report = resp.content
        
        self.session_state[cache_key] = report
        yield RunResponse(run_id=self.run_id, content={"step": "explainability", "report": report})

    def _format_feature_importance(self, feature_importance: Dict) -> str:
        formatted = []
        for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            formatted.append(f"- {feature}: {importance:.4f}")
        return "\n".join(formatted)

    def _format_interactions(self, interactions: List[Dict]) -> str:
        formatted = []
        for interaction in interactions[:10]:  # Show top 10 interactions
            feature = interaction['feature']
            targets = interaction['interactions'][:5]  # Show top 5 targets per feature
            formatted.append(f"- {feature} interacts with: {', '.join(targets)}")
        return "\n".join(formatted)

def load_model_results(results_path: str = "results") -> Dict:
    """Load model results from saved files."""
    try:
        with open(f"{results_path}/model_performance.json", 'r') as f:
            performance = json.load(f)
        return performance
    except FileNotFoundError:
        return {"accuracy": "N/A", "features_used": "N/A"}

def load_filtered_kg(kg_path: str = None) -> Dict:
    """Load filtered knowledge graph information."""
    if kg_path is None:
        kg_path = f"kg/{DATASET_NAME}_filtered.graphml"
    
    try:
        G = nx.read_graphml(kg_path)
        
        # Extract interactions
        interactions = []
        processed_nodes = set()
        
        for node in G.nodes():
            if node not in processed_nodes:
                neighbors = list(G.neighbors(node))
                if neighbors:
                    interactions.append({
                        'feature': node,
                        'interactions': neighbors
                    })
                processed_nodes.add(node)
        
        return {
            'edge_count': G.number_of_edges(),
            'node_count': G.number_of_nodes(),
            'interactions': interactions
        }
    except FileNotFoundError:
        return {'edge_count': 0, 'node_count': 0, 'interactions': []}

def calculate_feature_importance_from_shap(shap_values: np.ndarray, feature_names: List[str]) -> Dict:
    """Calculate feature importance from SHAP values."""
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    return dict(zip(feature_names, mean_abs_shap))

def get_dataset_info() -> Dict:
    """Get basic dataset information."""
    try:
        df = pd.read_csv(DATASET_PATH)
        return {
            'total_samples': len(df),
            'total_features': len(df.columns) - 1,  # Exclude target
            'feature_description': f"Clinical/medical features for {DATASET_NAME} dataset"
        }
    except FileNotFoundError:
        return {
            'total_samples': 'N/A',
            'total_features': 'N/A',
            'feature_description': 'Dataset not found'
        }

def run_explainability_analysis(shap_values: np.ndarray, selected_features: List[str], 
                               model_performance: Dict, arxiv_kb: ArxivKnowledgeBase,
                               recreate_analysis: bool = False) -> ExplainabilityReport:
    """
    Run comprehensive explainability analysis using the trained model results.
    
    Args:
        shap_values: SHAP values for selected features
        selected_features: List of selected feature names
        model_performance: Dictionary with model performance metrics
        arxiv_kb: Knowledge base for literature search
        recreate_analysis: Whether to recreate the analysis
    
    Returns:
        ExplainabilityReport with comprehensive analysis
    """
    
    # Prepare data for analysis
    feature_importance = calculate_feature_importance_from_shap(shap_values, selected_features)
    filtered_kg_info = load_filtered_kg()
    dataset_info = get_dataset_info()
    
    shap_analysis = {
        'selected_features': selected_features,
        'feature_importance': feature_importance
    }
    
    # Run explainability workflow
    storage_dir = os.path.join("storage", f"{DATASET_NAME}_explainability")
    os.makedirs(storage_dir, exist_ok=True)
    workflow_storage = JsonStorage(dir_path=storage_dir)
    
    wf_explainability = ExplainabilityWorkflow(
        session_id=f"explainability-{DATASET_NAME}", 
        storage=workflow_storage
    )
    
    report = None
    for resp in wf_explainability.run(
        model_params=model_performance,
        filtered_kg_info=filtered_kg_info,
        shap_analysis=shap_analysis,
        dataset_info=dataset_info,
        arxiv_kb=arxiv_kb,
        recreate_analysis=recreate_analysis
    ):
        if hasattr(resp, 'content') and isinstance(resp.content, dict) and 'report' in resp.content:
            report = resp.content['report']
    
    return report

def save_explainability_report(report: ExplainabilityReport, output_path: str = "results"):
    """Save explainability report to file."""
    os.makedirs(output_path, exist_ok=True)
    
    # Save as JSON
    with open(f"{output_path}/explainability_report.json", 'w') as f:
        json.dump(report.dict(), f, indent=2)
    
    # Save as readable text
    with open(f"{output_path}/explainability_report.txt", 'w') as f:
        f.write(f"EXPLAINABILITY REPORT - {DATASET_NAME.upper()}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"{report.summary}\n\n")
        
        f.write("FEATURE IMPORTANCE ANALYSIS\n")
        f.write("-" * 30 + "\n")
        f.write(f"{report.feature_importance_analysis}\n\n")
        
        f.write("INTERACTION ANALYSIS\n")
        f.write("-" * 25 + "\n")
        f.write(f"{report.interaction_analysis}\n\n")
        
        f.write("CLINICAL INSIGHTS\n")
        f.write("-" * 20 + "\n")
        f.write(f"{report.clinical_insights}\n\n")
        
        f.write("LIMITATIONS\n")
        f.write("-" * 15 + "\n")
        f.write(f"{report.limitations}\n\n")
        
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 18 + "\n")
        f.write(f"{report.recommendations}\n")
    
    print(f"Explainability report saved to {output_path}/") 