import os
import json
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List
from agno.agent import Agent, RunResponse
from agno.knowledge.arxiv import ArxivKnowledgeBase
from agno.storage.json import JsonStorage
from agno.workflow import Workflow
from params import DATASET_NAME, TARGET_COL, LLM_MODEL, DATASET_PATH
from pydantic import BaseModel, Field

class FeatureExplanation(BaseModel):
    feature_name: str = Field(description="Name of the feature")
    importance_score: float = Field(description="SHAPIQ importance score")
    biological_mechanism: str = Field(description="Biological/medical mechanism explaining the feature's importance")
    clinical_relevance: str = Field(description="Clinical relevance and interpretation")
    supporting_evidence: List[str] = Field(description="Citations from literature supporting this explanation")

class InteractionExplanation(BaseModel):
    feature1: str = Field(description="First feature in the interaction")
    feature2: str = Field(description="Second feature in the interaction")
    interaction_mechanism: str = Field(description="Biological mechanism of the interaction")
    clinical_significance: str = Field(description="Clinical significance of this interaction")
    supporting_evidence: List[str] = Field(description="Citations supporting this interaction")

class ExplainabilityReport(BaseModel):
    executive_summary: str = Field(description="Executive summary for clinicians")
    top_features: List[FeatureExplanation] = Field(description="Detailed explanations of top 10 most important features")
    key_interactions: List[InteractionExplanation] = Field(description="Explanations of top 5 most important feature interactions")
    clinical_decision_support: str = Field(description="Specific guidance for clinical decision-making")
    risk_stratification: str = Field(description="How to use the model for patient risk stratification")
    monitoring_recommendations: str = Field(description="Recommendations for patient monitoring based on important features")
    limitations_and_caveats: str = Field(description="Important limitations and caveats for clinical use")
    future_directions: str = Field(description="Suggestions for clinical validation and improvement")

class NoMemoryWorkflow(Workflow):
    def get_workflow_session(self):
        ws = super().get_workflow_session()
        ws.memory = None
        return ws

class FeatureAnalysisResponse(BaseModel):
    features: List[FeatureExplanation] = Field(description="List of feature explanations")

class InteractionAnalysisResponse(BaseModel):
    interactions: List[InteractionExplanation] = Field(description="List of interaction explanations")

class DetailedFeatureAnalysisWorkflow(NoMemoryWorkflow):
    description: str = "Analyzes individual features with detailed literature support."

    def run(self, top_features_with_scores: List[tuple], arxiv_kb: ArxivKnowledgeBase, recreate_analysis: bool = False):
        cache_key = f"feature_analysis_{DATASET_NAME}"
        if not recreate_analysis and self.session_state.get(cache_key):
            cached_analysis = self.session_state[cache_key]
            yield RunResponse(run_id=self.run_id, content={"step": "feature_analysis", "analysis": cached_analysis})
            return
        
        # Try with knowledge base first, fallback to without if it fails
        try:
            agent = Agent(model=LLM_MODEL, knowledge=arxiv_kb, search_knowledge=True, response_model=FeatureAnalysisResponse)
        except Exception as e:
            print(f"Warning: Knowledge base search failed ({e}), using agent without search")
            agent = Agent(model=LLM_MODEL, response_model=FeatureAnalysisResponse)
        
        feature_list = "\n".join([f"- {feat}: SHAPIQ importance = {score:.4f}" for feat, score in top_features_with_scores])
        
        prompt = f"""Analyze the following top features for predicting {TARGET_COL} in the {DATASET_NAME} dataset.

Top Features (by SHAPIQ importance):
{feature_list}

For EACH feature listed above, provide:
1. The exact biological/medical mechanism explaining why this feature is important for predicting {TARGET_COL}
2. Clinical relevance - how clinicians should interpret high/low values
3. Supporting evidence from medical literature (use your knowledge if literature search is unavailable)

Focus on:
- Pathophysiological mechanisms
- Clinical studies showing associations
- Biological pathways
- Risk factors and protective factors

Be specific and evidence-based. Each explanation should be actionable for clinicians."""

        try:
            resp = agent.run(prompt)
            feature_explanations = resp.content.features if hasattr(resp.content, 'features') else resp.content
        except Exception as e:
            print(f"Error in feature analysis: {e}")
            # Create fallback explanations
            feature_explanations = []
            for feat, score in top_features_with_scores:
                feature_explanations.append(FeatureExplanation(
                    feature_name=feat,
                    importance_score=score,
                    biological_mechanism=f"Feature {feat} shows high predictive importance for {TARGET_COL}",
                    clinical_relevance=f"Monitor {feat} values for clinical decision making",
                    supporting_evidence=["Analysis based on SHAPIQ importance scores"]
                ))
        
        self.session_state[cache_key] = feature_explanations
        yield RunResponse(run_id=self.run_id, content={"step": "feature_analysis", "analysis": feature_explanations})

class InteractionAnalysisWorkflow(NoMemoryWorkflow):
    description: str = "Analyzes feature interactions with literature support."

    def run(self, top_interactions: List[tuple], arxiv_kb: ArxivKnowledgeBase, recreate_analysis: bool = False):
        cache_key = f"interaction_analysis_{DATASET_NAME}"
        if not recreate_analysis and self.session_state.get(cache_key):
            cached_analysis = self.session_state[cache_key]
            yield RunResponse(run_id=self.run_id, content={"step": "interaction_analysis", "analysis": cached_analysis})
            return
        
        # Try with knowledge base first, fallback to without if it fails
        try:
            agent = Agent(model=LLM_MODEL, knowledge=arxiv_kb, search_knowledge=True, response_model=InteractionAnalysisResponse)
        except Exception as e:
            print(f"Warning: Knowledge base search failed ({e}), using agent without search")
            agent = Agent(model=LLM_MODEL, response_model=InteractionAnalysisResponse)
        
        interaction_list = "\n".join([f"- {feat1} <-> {feat2}" for feat1, feat2 in top_interactions])
        
        prompt = f"""Analyze the following feature interactions for predicting {TARGET_COL} in the {DATASET_NAME} dataset.

Top Feature Interactions:
{interaction_list}

For EACH interaction listed above, provide:
1. The biological/medical mechanism explaining how these features interact
2. Clinical significance - why this interaction matters for patient outcomes
3. Supporting evidence from medical literature (use your knowledge if literature search is unavailable)

Focus on:
- Synergistic effects between these features
- Compensatory mechanisms
- Cascade effects
- Feedback loops
- Common pathways

Be specific about the nature of the interaction and its clinical implications."""

        try:
            resp = agent.run(prompt)
            interaction_explanations = resp.content.interactions if hasattr(resp.content, 'interactions') else resp.content
        except Exception as e:
            print(f"Error in interaction analysis: {e}")
            # Create fallback explanations
            interaction_explanations = []
            for feat1, feat2 in top_interactions:
                interaction_explanations.append(InteractionExplanation(
                    feature1=feat1,
                    feature2=feat2,
                    interaction_mechanism=f"Features {feat1} and {feat2} show important interaction patterns",
                    clinical_significance=f"Combined monitoring of {feat1} and {feat2} may improve clinical outcomes",
                    supporting_evidence=["Analysis based on knowledge graph structure"]
                ))
        
        self.session_state[cache_key] = interaction_explanations
        yield RunResponse(run_id=self.run_id, content={"step": "interaction_analysis", "analysis": interaction_explanations})

class ClinicalInsightsWorkflow(NoMemoryWorkflow):
    description: str = "Generates comprehensive clinical insights and recommendations."

    def run(self, feature_explanations: List[FeatureExplanation], interaction_explanations: List[InteractionExplanation], 
            model_performance: Dict, arxiv_kb: ArxivKnowledgeBase, recreate_analysis: bool = False):
        
        cache_key = f"clinical_insights_{DATASET_NAME}"
        if not recreate_analysis and self.session_state.get(cache_key):
            cached_report = self.session_state[cache_key]
            yield RunResponse(run_id=self.run_id, content={"step": "clinical_insights", "report": cached_report})
            return
        
        # Try with knowledge base first, fallback to without if it fails
        try:
            agent = Agent(model=LLM_MODEL, knowledge=arxiv_kb, search_knowledge=True, response_model=ExplainabilityReport)
        except Exception as e:
            print(f"Warning: Knowledge base search failed ({e}), using agent without search")
            agent = Agent(model=LLM_MODEL, response_model=ExplainabilityReport)
        
        # Format feature and interaction summaries
        feature_summary = "\n".join([f"- {fe.feature_name}: {fe.biological_mechanism}" for fe in feature_explanations[:5]])
        interaction_summary = "\n".join([f"- {ie.feature1} & {ie.feature2}: {ie.interaction_mechanism}" for ie in interaction_explanations[:3]])
        
        prompt = f"""Based on the detailed feature and interaction analyses, generate a comprehensive clinical explainability report for the {TARGET_COL} prediction model.

Model Performance:
- Accuracy: {model_performance.get('accuracy', 'N/A')}%
- Number of features used: {model_performance.get('features_used', 'N/A')}

Key Features (summary):
{feature_summary}

Key Interactions (summary):
{interaction_summary}

Generate a comprehensive report that includes:

1. **Executive Summary**: A concise summary for busy clinicians explaining what the model does and its key insights

2. **Clinical Decision Support**: Specific, actionable guidance on how to use the model outputs in clinical practice, including:
   - When to be most concerned about a patient
   - Which features to monitor most closely
   - How to interpret combinations of features

3. **Risk Stratification**: Clear guidelines on how to stratify patients into risk categories based on the model's predictions and important features

4. **Monitoring Recommendations**: Specific recommendations for monitoring patients based on the important features, including:
   - Frequency of monitoring
   - Which parameters to track
   - Early warning signs

5. **Limitations and Caveats**: Important limitations that clinicians must understand, including:
   - Patient populations where the model may be less accurate
   - Scenarios where clinical judgment should override the model
   - Data quality requirements

6. **Future Directions**: Suggestions for clinical validation and improvement

Be practical and clinically oriented."""

        try:
            resp = agent.run(prompt)
            report = resp.content
        except Exception as e:
            print(f"Error in clinical insights generation: {e}")
            # Create fallback report
            report = ExplainabilityReport(
                executive_summary=f"Machine learning model for {TARGET_COL} prediction achieved {model_performance.get('accuracy', 'N/A')}% accuracy using {model_performance.get('features_used', 'N/A')} key features.",
                clinical_decision_support="Use model predictions as supplementary information alongside clinical judgment.",
                risk_stratification="Higher prediction scores indicate increased risk and may warrant closer monitoring.",
                monitoring_recommendations="Monitor the key features identified by the model for changes over time.",
                limitations_and_caveats="Model should be used as a decision support tool, not as a replacement for clinical expertise.",
                future_directions="Validate model performance in prospective clinical studies."
            )
        
        # Merge the detailed feature and interaction analyses into the report
        report.top_features = feature_explanations
        report.key_interactions = interaction_explanations
        
        self.session_state[cache_key] = report
        yield RunResponse(run_id=self.run_id, content={"step": "clinical_insights", "report": report})

def get_top_features_and_interactions(shap_dict: dict, feature_names: List[str], 
                                     filtered_graph: nx.Graph, n_features: int = 10, n_interactions: int = 5):
    """Extract top features by SHAP importance and top interactions from the graph."""
    # Extract individual feature SHAP values and take absolute values
    feature_shap_values = {}
    for key, value in shap_dict.items():
        if len(key) == 1:  # Single feature
            feature_idx = key[0]
            if feature_idx < len(feature_names):
                feature_shap_values[feature_names[feature_idx]] = abs(value)
    
    # Sort features by absolute SHAP importance and get top features
    sorted_features = sorted(feature_shap_values.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:n_features]
    
    # Extract pairwise interaction SHAP values and take absolute values
    interaction_shap_values = {}
    for key, value in shap_dict.items():
        if len(key) == 2:  # Pairwise interaction
            f1_idx, f2_idx = key
            if f1_idx < len(feature_names) and f2_idx < len(feature_names):
                f1_name = feature_names[f1_idx]
                f2_name = feature_names[f2_idx]
                edge = tuple(sorted([f1_name, f2_name]))
                interaction_shap_values[edge] = abs(value)
    
    # Get top interactions from the filtered graph with SHAP scores
    interactions_with_scores = []
    for edge in filtered_graph.edges():
        edge_sorted = tuple(sorted([edge[0], edge[1]]))
        if edge_sorted in interaction_shap_values:
            interactions_with_scores.append((edge[0], edge[1], interaction_shap_values[edge_sorted]))
    
    # Sort by SHAP interaction strength and get top interactions
    interactions_with_scores.sort(key=lambda x: x[2], reverse=True)
    top_interactions = [(i[0], i[1]) for i in interactions_with_scores[:n_interactions]]
    
    return top_features, top_interactions

def run_explainability_analysis(shap_dict: dict, selected_features: List[str], 
                               model_performance: Dict, arxiv_kb: ArxivKnowledgeBase,
                               recreate_analysis: bool = False) -> ExplainabilityReport:
    """Run comprehensive explainability analysis with detailed literature support."""
    
    # Load filtered knowledge graph
    filtered_kg_path = f"kg/{DATASET_NAME}_filtered.graphml"
    filtered_graph = nx.read_graphml(filtered_kg_path) if os.path.exists(filtered_kg_path) else nx.Graph()
    
    # Get top features and interactions
    top_features, top_interactions = get_top_features_and_interactions(
        shap_dict, selected_features, filtered_graph
    )
    
    # Step 1: Detailed feature analysis
    print("Step 1: Analyzing top features with literature support...")
    storage_dir = os.path.join("storage", f"{DATASET_NAME}_explainability")
    os.makedirs(storage_dir, exist_ok=True)
    
    workflow_storage = JsonStorage(dir_path=storage_dir)
    wf_features = DetailedFeatureAnalysisWorkflow(
        session_id=f"feature-analysis-{DATASET_NAME}", 
        storage=workflow_storage
    )
    
    feature_explanations = None
    for resp in wf_features.run(
        top_features_with_scores=top_features,
        arxiv_kb=arxiv_kb,
        recreate_analysis=recreate_analysis
    ):
        if hasattr(resp, 'content') and isinstance(resp.content, dict) and 'analysis' in resp.content:
            feature_explanations = resp.content['analysis']
    
    # Step 2: Interaction analysis
    print("Step 2: Analyzing feature interactions with literature support...")
    wf_interactions = InteractionAnalysisWorkflow(
        session_id=f"interaction-analysis-{DATASET_NAME}", 
        storage=workflow_storage
    )
    
    interaction_explanations = None
    for resp in wf_interactions.run(
        top_interactions=top_interactions,
        arxiv_kb=arxiv_kb,
        recreate_analysis=recreate_analysis
    ):
        if hasattr(resp, 'content') and isinstance(resp.content, dict) and 'analysis' in resp.content:
            interaction_explanations = resp.content['analysis']
    
    # Step 3: Generate comprehensive clinical insights
    print("Step 3: Generating comprehensive clinical insights...")
    wf_insights = ClinicalInsightsWorkflow(
        session_id=f"clinical-insights-{DATASET_NAME}", 
        storage=workflow_storage
    )
    
    report = None
    for resp in wf_insights.run(
        feature_explanations=feature_explanations or [],
        interaction_explanations=interaction_explanations or [],
        model_performance=model_performance,
        arxiv_kb=arxiv_kb,
        recreate_analysis=recreate_analysis
    ):
        if hasattr(resp, 'content') and isinstance(resp.content, dict) and 'report' in resp.content:
            report = resp.content['report']
    
    return report

def save_explainability_report(report: ExplainabilityReport, output_path: str = "results"):
    """Save comprehensive explainability report."""
    os.makedirs(output_path, exist_ok=True)
    
    # Save as JSON
    with open(f"{output_path}/explainability_report.json", 'w') as f:
        json.dump(report.dict(), f, indent=2)
    
    # Save as readable text
    with open(f"{output_path}/explainability_report.txt", 'w') as f:
        f.write(f"CLINICAL EXPLAINABILITY REPORT - {DATASET_NAME.upper()}\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 30 + "\n")
        f.write(f"{report.executive_summary}\n\n")
        
        f.write("TOP PREDICTIVE FEATURES\n")
        f.write("-" * 30 + "\n")
        for i, feat in enumerate(report.top_features, 1):
            f.write(f"\n{i}. {feat.feature_name} (Importance: {feat.importance_score:.4f})\n")
            f.write(f"   Mechanism: {feat.biological_mechanism}\n")
            f.write(f"   Clinical Relevance: {feat.clinical_relevance}\n")
            f.write(f"   Evidence: {'; '.join(feat.supporting_evidence)}\n")
        
        f.write("\n\nKEY FEATURE INTERACTIONS\n")
        f.write("-" * 30 + "\n")
        for i, inter in enumerate(report.key_interactions, 1):
            f.write(f"\n{i}. {inter.feature1} <-> {inter.feature2}\n")
            f.write(f"   Mechanism: {inter.interaction_mechanism}\n")
            f.write(f"   Clinical Significance: {inter.clinical_significance}\n")
            f.write(f"   Evidence: {'; '.join(inter.supporting_evidence)}\n")
        
        f.write("\n\nCLINICAL DECISION SUPPORT\n")
        f.write("-" * 30 + "\n")
        f.write(f"{report.clinical_decision_support}\n\n")
        
        f.write("RISK STRATIFICATION GUIDELINES\n")
        f.write("-" * 30 + "\n")
        f.write(f"{report.risk_stratification}\n\n")
        
        f.write("MONITORING RECOMMENDATIONS\n")
        f.write("-" * 30 + "\n")
        f.write(f"{report.monitoring_recommendations}\n\n")
        
        f.write("LIMITATIONS AND CAVEATS\n")
        f.write("-" * 30 + "\n")
        f.write(f"{report.limitations_and_caveats}\n\n")
        
        f.write("FUTURE DIRECTIONS\n")
        f.write("-" * 30 + "\n")
        f.write(f"{report.future_directions}\n")
    
    print(f"Comprehensive clinical explainability report saved to {output_path}/") 