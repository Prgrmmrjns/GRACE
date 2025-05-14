import os
import pandas as pd
import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
import lime
import lime.lime_tabular

# Imports from params
from params import (
    DATASET_PATH, DATASET_NAME, TARGET_COL, METRIC, get_model as get_ml_model,
    STUDY_TYPE, KEYWORDS, LLM_PROVIDER, MODEL as LLM_MODEL_NAME,
    TARGET_COL_DICT # Added import
)
from graph_utils import nx_to_node_groups
from visualizations import visualize_lime_knowledge_graph

# Agno imports (similar to create_kg.py)
from typing import List, Iterator, Dict, Any
from agno.agent import Agent, RunResponse
from pydantic import BaseModel, Field
from agno.models.openai import OpenAIChat
from agno.models.ollama import Ollama
from agno.vectordb.pgvector import PgVector, SearchType
from agno.knowledge.arxiv import ArxivKnowledgeBase
from agno.workflow import Workflow
from agno.embedder.openai import OpenAIEmbedder
import csv

# PDF Generation imports
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import HexColor


# --- Helper functions for Agno (adapted from create_kg.py) ---
def get_llm_agno():
    if LLM_PROVIDER == "openai":
        return OpenAIChat(id=LLM_MODEL_NAME, temperature=0.1)
    else:
        return Ollama(id=LLM_MODEL_NAME)

def get_feature_names_and_description():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        feature_names = [col for col in header if col.lower() != TARGET_COL.lower()]
    
    dataset_info_path = f"dataset_info/{DATASET_NAME.lower()}_info.txt"

    with open(dataset_info_path, 'r') as f:
        dataset_description = f.read().strip()
    return feature_names, dataset_description

def get_knowledge_base_agno():
    db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"
    vector_db = PgVector(
        table_name="articles", 
        db_url=db_url,
        search_type=SearchType.hybrid,
        embedder=OpenAIEmbedder() # Explicitly OpenAIEmbedder() with defaults
    )
    return ArxivKnowledgeBase(queries=KEYWORDS, vector_db=vector_db, create_if_empty=False)


# --- Pydantic Model for Structured Report ---
class PatientReport(BaseModel):
    # patient_summary: str = Field(..., description="Summary of LIME findings, including the most important contributing features (and their LIME values) for each intermediate node and the ensemble model. State the predicted outcome for the patient (using descriptive labels like 'Alzheimer\'s Disease' or 'Survived') and its probability if applicable.") # REMOVED
    contextual_explanation: str = Field(..., description="Explain the findings using background information from the provided dataset description, the patient's specific feature values, and prediction task context. Connect the LIME findings and actual patient feature values to plausible mechanisms or reasons based on this context.")
    recommendations_insights: str = Field(..., description=f"Based on the study type ('{STUDY_TYPE}'), provide recommendations. If retrospective, suggest general policies or research directions. If prospective, suggest potential (non-medical, general well-being) interventions or lifestyle adjustments. Also, provide additional insights into potential causal disease or pathological mechanisms suggested by the feature contributions, their actual values for this patient, and graph structure.")


# --- Agno Workflow for Patient Report Generation ---
class PatientReportWorkflow(Workflow):
    description: str = "Generates a patient-specific structured report based on LIME findings, graph structure, and knowledge base."

    def run(self,
            instance_features: pd.Series,
            node_lime_weights: Dict[str, Dict[str, float]],
            ensemble_lime_weights: Dict[str, float],
            predicted_class_label_str: str, 
            predicted_proba: float,
            target_col: str,
            dataset_name: str,
            dataset_description: str,
            study_type: str,
            raw_graph_nx: nx.DiGraph, 
            optimized_node_groups: Dict[str, List[str]]
           ) -> Iterator[RunResponse]:

        model = get_llm_agno()
        agent = Agent(model=model, response_model=PatientReport) 

        lime_summary_str = "LIME Explanations (for context, not for patient summary section):\n" # Clarified purpose
        lime_summary_str += "Node Model Contributions (feature: LIME_weight):\n"
        for node, weights in node_lime_weights.items():
            if weights: 
                lime_summary_str += f"  Intermediate Node '{node}':\n"
                for feature, weight in sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
                    lime_summary_str += f"    - {feature}: {weight:.4f}\n"
        
        lime_summary_str += "\nEnsemble Model Contributions (intermediate_node: LIME_weight):\n"
        for node, weight in sorted(ensemble_lime_weights.items(), key=lambda x: abs(x[1]), reverse=True):
            lime_summary_str += f"  - {node}: {weight:.4f}\n"

        graph_structure_str = "\nKnowledge Graph Structure (Intermediate Node: [Input Features]):\n"
        for node, features in optimized_node_groups.items():
            graph_structure_str += f"  - {node}: {features}\n"

        instance_values_str = "\nPatient's Feature Values (for features with notable LIME contributions):\n"
        relevant_features_for_instance = set()
        for node, weights in node_lime_weights.items():
            for feat in weights.keys():
                relevant_features_for_instance.add(feat)
        
        for ens_feat_name in ensemble_lime_weights.keys():
            is_original_feature = ens_feat_name in instance_features.index
            is_complex_node = ens_feat_name in optimized_node_groups and len(optimized_node_groups.get(ens_feat_name, [])) > 1
            if is_original_feature and not is_complex_node: 
                 relevant_features_for_instance.add(ens_feat_name)

        for feat_name in sorted(list(relevant_features_for_instance)):
            if feat_name in instance_features:
                 instance_values_str += f"  - {feat_name}: {instance_features[feat_name]}\n"

        # The Patient Summary section will be generated statically in the PDF.
        # The LLM should start with the Contextual Explanation.
        prompt = f"""
You are an AI assistant helping to interpret machine learning model predictions for a patient.
Contextual Information (provided for your understanding):
- Dataset: '{dataset_name}', Target Variable: '{target_col}'.
- Study Type: '{study_type}'.
- Dataset Description: {dataset_description}
- Model Prediction for this patient: '{target_col}' is '{predicted_class_label_str}' (Probability: {predicted_proba:.4f}).
- LIME contributions and patient feature values are summarized below for your reference to build the explanation.

LIME Context:
{lime_summary_str}

Graph Structure Context:
{graph_structure_str}

Patient's Feature Values Context:
{instance_values_str}

Please generate the following sections for the patient report, starting directly with section 1 (Contextual Explanation):

1.  **Contextual Explanation**: Based on the provided dataset information, the graph structure, *the patient's specific feature values (listed above)*, and your general knowledge about topics related to '{KEYWORDS}', explain *why these features, with their given values for this patient*, might be contributing in this way. Connect findings to plausible underlying mechanisms or pathways.
2.  **Personalized Insights and Considerations for an Individual with this Profile**:
    *   Considering this patient's specific feature values (provided in the 'Patient\'s Feature Values' section above) and their LIME contributions (detailed in the 'LIME Explanations' section), what are the most striking observations or potential areas of focus that emerge for an individual exhibiting this particular data profile?
    *   Even acknowledging the overall '{study_type}' nature of the data, if one were advising an individual with a similar profile, what general well-being factors or lifestyle considerations might be relevant for them to discuss with their healthcare provider? (Focus on general well-being and informational points, not direct medical advice).
    *   Based *specifically* on the interplay of this patient's feature values, their LIME scores, and the known roles of the features/nodes in the graph structure, what potential underlying pathological mechanisms or individual risk factors seem most highlighted or exacerbated in this specific case? Be as specific as the data allows.

Focus on clarity, conciseness, and relevance to the patient's specific prediction and context.
"""
        
        self.session_id = f"patient_report_{dataset_name}_instance_{instance_features.name}"
        resp = agent.run(prompt)
        yield RunResponse(run_id=self.run_id, content={"step": "patient_report_generated", "report_data": resp.content})


# --- PDF Generation Function ---
def generate_pdf_report(output_path: str,
                        report_title: str,
                        graph_image_path: str,
                        report_data: PatientReport,
                        instance_index: Any,
                        predicted_class_label: str,
                        predicted_proba: float,
                        node_lime_weights: Dict[str, Dict[str, float]],
                        ensemble_lime_weights: Dict[str, float]):
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    lime_contribution_style = ParagraphStyle(
        'LimeContribution',
        parent=styles['Normal'],
        leftIndent=20,
    )
    # Define the inner style here
    lime_node_inner_style = ParagraphStyle(
        'LimeNodeInner',
        parent=lime_contribution_style, # It should be based on the parent's indentation
        leftIndent=40, # Further indent from the parent style's indent is not cumulative by default
                      # So, this 40 will be from the page margin, not from lime_contribution_style's 20.
                      # If cumulative indent is desired, it needs more complex handling or direct calculation.
                      # For now, this sets an absolute indent of 40.
    )

    story.append(Paragraph(report_title, styles['h1']))
    story.append(Paragraph(f"Interpretability Report for Patient Instance: {instance_index}", styles['h2']))
    story.append(Spacer(1, 0.2 * inch))

    # --- Static Patient Summary Section ---
    story.append(Paragraph("1. Patient Summary", styles['h2']))
    story.append(Spacer(1, 0.1 * inch))
    
    # Prediction and Confidence
    story.append(Paragraph(f"<b>Prediction:</b> {predicted_class_label}", styles['Normal']))
    story.append(Paragraph(f"<b>Confidence:</b> {predicted_proba:.2%}", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # LIME Contributions Heading
    story.append(Paragraph("<b>Key Feature Contributions according to LIME:</b>", styles['h3']))
    story.append(Paragraph("LIME values are between -1 and 1, where positive values support the prediction, and negative values oppose it.", styles['Normal']))
    story.append(Paragraph("Only contributions with absolute value > 0.01 are shown.", styles['Normal']))
    story.append(Spacer(1, 0.1 * inch))

    # Ensemble Model Contributions
    if ensemble_lime_weights:
        story.append(Paragraph("<u>Ensemble Model:</u>", styles['Normal']))
        # Sort by absolute value, descending
        sorted_ensemble_weights = sorted(ensemble_lime_weights.items(), key=lambda item: abs(item[1]), reverse=True)
        # Filter for contributions > 0.01 in absolute value
        filtered_ensemble_weights = [(feature, weight) for feature, weight in sorted_ensemble_weights if abs(weight) > 0.01]
        
        for feature, weight in filtered_ensemble_weights:
            story.append(Paragraph(f"- {feature}: {weight:.4f}", lime_contribution_style))
        
        story.append(Spacer(1, 0.15 * inch))
    
    # Node Model Contributions
    if node_lime_weights:
        story.append(Paragraph("<u>Node Models (Contributions to Intermediate Nodes):</u>", styles['Normal']))
        for node_name, weights in node_lime_weights.items():
            if weights:
                # Filter for features with contributions > 0.01 in absolute value
                filtered_weights = {feat: weight for feat, weight in weights.items() if abs(weight) > 0.01}
                if filtered_weights:
                    story.append(Paragraph(f"<i>Node - {node_name}:</i>", lime_contribution_style))
                    sorted_node_weights = sorted(filtered_weights.items(), key=lambda item: abs(item[1]), reverse=True)
                    for feature, weight in sorted_node_weights:
                        # Use the pre-defined inner style
                        story.append(Paragraph(f"-- {feature}: {weight:.4f}", style=lime_node_inner_style))
                story.append(Spacer(1, 0.1 * inch))
    story.append(Spacer(1, 0.2 * inch))
    # --- End Static Patient Summary Section ---

    # Add LIME graph image (can be before or after LLM content)
    story.append(Paragraph("LIME-Informed Knowledge Graph Visualization", styles['h2']))
    story.append(Spacer(1, 0.1 * inch))
    img = Image(graph_image_path, width=6.5*inch, height=6.5*inch, hAlign='CENTER')
    story.append(img)
    story.append(PageBreak()) 

    # --- LLM Generated Sections ---
    # Renumbering to follow the static summary
    story.append(Paragraph("2. Contextual Explanation", styles['h2']))
    story.append(Paragraph(report_data.contextual_explanation.replace("\n", "<br/>"), styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("3. Personalized Insights and Considerations", styles['h2']))
    story.append(Paragraph(report_data.recommendations_insights.replace("\n", "<br/>"), styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))
    
    doc.build(story)


def run_lime_interpretability():

    df = pd.read_csv(DATASET_PATH, encoding='utf-8')
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

    raw_graph_path_kg = f"kg/{DATASET_NAME}.graphml" 
    optimized_graph_path = f"kg/{DATASET_NAME}_optimized.graphml"
        
    raw_graph_nx_kg = nx.read_graphml(raw_graph_path_kg)
    optimized_graph_nx = nx.read_graphml(optimized_graph_path)
    optimized_node_groups = nx_to_node_groups(optimized_graph_nx)

    X_train_graph = pd.DataFrame(index=X_train.index)
    X_val_graph = pd.DataFrame(index=X_val.index)
    X_test_graph = pd.DataFrame(index=X_test.index)
    node_models = {}

    for node, features in optimized_node_groups.items():

        if len(features) == 1 and features[0] in X_train.columns:
            X_train_graph[node] = X_train[features[0]]
            X_val_graph[node] = X_val[features[0]]
            X_test_graph[node] = X_test[features[0]]
        elif features: 
            node_model = get_ml_model()
            eval_set_node = [(X_val[features], y_val)]
            node_model.fit(X_train[features], y_train, eval_set=eval_set_node, verbose=False)
            node_models[node] = node_model
            
            if METRIC != 'accuracy': 
                X_train_graph[node] = node_model.predict_proba(X_train[features])[:, 1]
                X_val_graph[node] = node_model.predict_proba(X_val[features])[:, 1]
                X_test_graph[node] = node_model.predict_proba(X_test[features])[:, 1]
            else: 
                X_train_graph[node] = node_model.predict(X_train[features])
                X_val_graph[node] = node_model.predict(X_val[features])
                X_test_graph[node] = node_model.predict(X_test[features])

    for node in optimized_node_groups.keys():
        if node not in X_train_graph.columns:
            X_train_graph[node] = 0.0
            X_val_graph[node] = 0.0
            X_test_graph[node] = 0.0

    ensemble_model = get_ml_model()
    eval_set_ensemble = [(X_val_graph, y_val)]
    ensemble_model.fit(X_train_graph, y_train, eval_set=eval_set_ensemble, verbose=False)

    random_idx_val = np.random.randint(0, len(X_test))
    X_test_instance = X_test.iloc[[random_idx_val]]
    X_test_graph_instance = X_test_graph.iloc[[random_idx_val]]
    instance_index = X_test_instance.index[0]
    
    lime_class_names = [str(c) for c in sorted(y_train.unique())] 

    # Get raw numerical prediction from the model
    _raw_model_prediction = ensemble_model.predict(X_test_graph_instance)
    # Extract the value properly to avoid NumPy deprecation warning
    raw_predicted_value = int(_raw_model_prediction[0].item()) # e.g., 0, 1, 2 or 1, 2, 3

    # Convert raw numerical prediction to string label using TARGET_COL_DICT
    # This is for the LLM and human-readable report.
    predicted_class_label_for_report = TARGET_COL_DICT.get(raw_predicted_value, str(raw_predicted_value))

    predicted_probas = ensemble_model.predict_proba(X_test_graph_instance)[0]
    target_label_idx_to_explain = lime_class_names.index(str(raw_predicted_value))


    predicted_proba_for_explained_class = predicted_probas[target_label_idx_to_explain]

    node_lime_weights = {} 
    for node, model_to_explain in node_models.items():
        features = optimized_node_groups.get(node, [])
        
        current_X_test_instance_features = X_test_instance[features]


        if len(features) > 1:
            training_data_for_lime = X_train[features].values + np.random.normal(0, 1e-8, X_train[features].values.shape)
            
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=training_data_for_lime,
                feature_names=features,
                class_names=lime_class_names, 
                mode='classification', 
                discretize_continuous=False 
            )
            explanation_obj = explainer.explain_instance(
                data_row=current_X_test_instance_features.values[0],
                predict_fn=model_to_explain.predict_proba,
                num_features=len(features),
                labels=(target_label_idx_to_explain,) 
            )
            
            current_node_weights = {}
            if target_label_idx_to_explain in explanation_obj.local_exp:
                exp_for_target_class = explanation_obj.local_exp[target_label_idx_to_explain]
                for feature_idx, weight_val in exp_for_target_class:
                    if 0 <= feature_idx < len(features):
                        feature_name = features[feature_idx]
                        current_node_weights[feature_name] = weight_val
            node_lime_weights[node] = current_node_weights

    ensemble_lime_weights = {}
    ensemble_feature_names = X_train_graph.columns.tolist()
    training_data_for_lime_ensemble = X_train_graph.values + np.random.normal(0, 1e-8, X_train_graph.values.shape)
    
    explainer_ensemble = lime.lime_tabular.LimeTabularExplainer(
        training_data=training_data_for_lime_ensemble, 
        feature_names=ensemble_feature_names, 
        class_names=lime_class_names, 
        mode='classification',
        discretize_continuous=False
    )
    explanation_ensemble_obj = explainer_ensemble.explain_instance(
        data_row=X_test_graph_instance.values[0],
        predict_fn=ensemble_model.predict_proba,
        num_features=X_train_graph.shape[1],
        labels=(target_label_idx_to_explain,)
    )

    if target_label_idx_to_explain in explanation_ensemble_obj.local_exp:
        exp_for_target_class_ensemble = explanation_ensemble_obj.local_exp[target_label_idx_to_explain]
        for feature_idx, weight_val in exp_for_target_class_ensemble:
            if 0 <= feature_idx < len(ensemble_feature_names):
                feature_name = ensemble_feature_names[feature_idx]
                ensemble_lime_weights[feature_name] = weight_val

    lime_graph_image_path = ""
    if raw_graph_nx_kg is not None and optimized_node_groups:
        for node in list(optimized_node_groups.keys()):
             if node not in raw_graph_nx_kg:
                 if node in optimized_node_groups: del optimized_node_groups[node] # Check if key exists
             elif 'entity_type' not in raw_graph_nx_kg.nodes[node]:
                 raw_graph_nx_kg.nodes[node]['entity_type'] = 'INTERMEDIATE_NODE'
        
        for node, features_list in optimized_node_groups.items(): # Renamed features to features_list
            for feat in features_list: # Use features_list here
                if 'entity_type' not in raw_graph_nx_kg.nodes[feat]: # Check if feat exists before accessing
                    raw_graph_nx_kg.nodes[feat]['entity_type'] = 'INPUT_NODE'
                    # else: feat is not in raw_graph_nx_kg.nodes, already warned.

        lime_graph_image_dir = "images"
        os.makedirs(lime_graph_image_dir, exist_ok=True)
        # Create a unique filename that includes the instance index
        lime_graph_filename = f"{DATASET_NAME}_instance_{instance_index}_lime.png"
        lime_graph_image_path = os.path.join(lime_graph_image_dir, lime_graph_filename)
    
        # Delete any existing files to avoid using cached versions
        import glob
        for old_file in glob.glob(f"{lime_graph_image_dir}/{DATASET_NAME}*_lime.png"):
            try:
                os.remove(old_file)
            except:
                pass
    
        visualize_lime_knowledge_graph(
            G_raw=raw_graph_nx_kg, 
            dataset=f"{DATASET_NAME}_instance_{instance_index}", 
            node_groups=optimized_node_groups, 
            node_lime_weights=node_lime_weights,
            ensemble_lime_weights=ensemble_lime_weights,
            output_base_dir=lime_graph_image_dir,
            predicted_class_label=predicted_class_label_for_report
        )

    _, dataset_description = get_feature_names_and_description()
    report_workflow = PatientReportWorkflow() 
    current_instance_all_features = X_test.loc[instance_index]

    report_responses = report_workflow.run(
        instance_features=current_instance_all_features,
        node_lime_weights=node_lime_weights,
        ensemble_lime_weights=ensemble_lime_weights,
        predicted_class_label_str=predicted_class_label_for_report, # Pass the string label
        predicted_proba=predicted_proba_for_explained_class,
        target_col=TARGET_COL,
        dataset_name=DATASET_NAME,
        dataset_description=dataset_description,
        study_type=STUDY_TYPE,
        raw_graph_nx=raw_graph_nx_kg,
        optimized_node_groups=optimized_node_groups
    )

    report_data_content = None
    for resp in report_responses:
        if hasattr(resp, 'content') and isinstance(resp.content, dict) and 'report_data' in resp.content:
            report_data_content = resp.content['report_data']
            break 

    if report_data_content:
        if isinstance(report_data_content, PatientReport):

            pdf_output_dir = "reports"
            os.makedirs(pdf_output_dir, exist_ok=True)
            pdf_output_path = os.path.join(pdf_output_dir, f"{DATASET_NAME}_patient_report.pdf")
            
            generate_pdf_report(
                output_path=pdf_output_path,
                report_title=f"GRACE Explainability Report: {DATASET_NAME}",
                graph_image_path=lime_graph_image_path, 
                report_data=report_data_content,
                instance_index=instance_index,
                predicted_class_label=predicted_class_label_for_report,
                predicted_proba=predicted_proba_for_explained_class,
                node_lime_weights=node_lime_weights,
                ensemble_lime_weights=ensemble_lime_weights
            )

if __name__ == "__main__":
    run_lime_interpretability() 