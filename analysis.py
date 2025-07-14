import shap
import numpy as np
from agno.agent import Agent
from visualizations import visualize_knowledge_graph
from params import LLM_MODEL, TARGET_COL, USER_AIM
from pydantic import BaseModel
from typing import List
import textwrap
import os
import networkx as nx
import pandas as pd

class InteractionExplanation(BaseModel):
    feature1: str
    feature2: str
    explanation: str

class ExplanationList(BaseModel):
    explanations: List[InteractionExplanation]

def run_interaction_analysis(model, X, feature_names, dataset_name, optimized_results):
    """
    Focuses on interpreting the feature interactions from the optimized model,
    generating a SHAP-based interaction graph and a detailed text report with AI-generated explanations and graph topology analysis.
    """
    explainer = shap.TreeExplainer(model)
    shap_interaction_values = explainer.shap_interaction_values(X)

    if isinstance(shap_interaction_values, list):
        shap_interaction_values = shap_interaction_values[1]

    mean_abs_shap_interactions = np.abs(shap_interaction_values).mean(0)
    
    best_constraints = optimized_results['best_constraints']
    best_labels = optimized_results['best_labels']
    feature_to_labels_map = optimized_results['feature_to_labels_map']

    top_interactions = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            if mean_abs_shap_interactions[i, j] > 0.001:
                top_interactions.append(((feature_names[i], feature_names[j]), mean_abs_shap_interactions[i, j]))
    
    top_interactions.sort(key=lambda x: x[1], reverse=True)
    
    # --- 1. Graph Creation for Topology Analysis ---
    G = nx.Graph()
    for group in best_constraints:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                G.add_edge(group[i], group[j])
    
    # --- 2. Generate Explanations for Top 5 Interactions (for graph and highlights) ---
    interactions_to_explain = top_interactions[:5]

    with open(f'dataset_info/{dataset_name}_info.txt', 'r') as f:
        dataset_info = f.read()

    interaction_prompt_part = "\n".join([f"- {feats[0]} <--> {feats[1]} (SHAP Value: {val:.4f})" for feats, val in interactions_to_explain])
    
    prompt = f"""
    You are a medical expert providing concise, specific, and clinically relevant explanations for feature interactions.

    **Context:**
    - **Dataset:** {dataset_name}
    - **Prediction Target:** {TARGET_COL} (predicting the presence or severity of this condition)
    - **Clinical Background:** {dataset_info}

    **Task:**
    For each of the following top 5 feature interactions, provide a biologically plausible explanation for how they **jointly** influence the target, `{TARGET_COL}`.
    Your explanation must be specific and grounded in the provided clinical context and the interaction's SHAP value. Avoid generic statements.

    **Constraint: Each explanation MUST be 8 words or less.**

    **Interactions to Explain:**
    {interaction_prompt_part}

    Return the explanations in a structured JSON format. For each interaction, provide the two features and the short, context-aware explanation.
    """

    agent = Agent(
        model=LLM_MODEL,
        response_model=ExplanationList,
        instructions="You are a medical expert providing concise explanations for feature interactions. Adhere strictly to word limits and use the provided context."
    )
    
    structured_explanations = agent.run(prompt).content.explanations

    # --- 3. Visualize the Graph with Top 5 Explanations ---
    edge_labels = {
        (exp.feature1, exp.feature2): textwrap.fill(exp.explanation, width=20)
        for exp in structured_explanations
    }

    visualize_knowledge_graph(
        constraints=best_constraints,
        labels=best_labels,
        filename=f'{dataset_name}_shap_interpreted_graph.png',
        feature_to_labels_map=feature_to_labels_map,
        edge_labels=edge_labels
    )

    # --- 4. Generate Comprehensive MD Report ---
    report_filename = f'results/{dataset_name}_interaction_report.md'
    
    # Prepare topology strings for the prompt
    if G.number_of_nodes() > 0:
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        top_5_degree = sorted(degree_centrality.items(), key=lambda item: item[1], reverse=True)[:5]
        top_5_betweenness = sorted(betweenness_centrality.items(), key=lambda item: item[1], reverse=True)[:5]
        
        top_5_degree_str = "\n".join([f"  - {node}: {value:.4f}" for node, value in top_5_degree])
        top_5_betweenness_str = "\n".join([f"  - {node}: {value:.4f}" for node, value in top_5_betweenness])
        connectivity_str = "Connected" if nx.is_connected(G) else "Not fully connected"
        density_str = f"{nx.density(G):.4f}"

    top_5_interactions_str = "\n".join([f"  - {exp.feature1} <--> {exp.feature2} (SHAP Value: {next((f'{val:.4f}' for (f1, f2), val in interactions_to_explain if {exp.feature1, exp.feature2} == {f1, f2}), 'N/A')}): {exp.explanation}" for exp in structured_explanations])

    report_prompt = f"""
You are a senior medical analyst and scientific writer creating a clinical report based on a machine learning analysis.
The report must be in Markdown format, clear, and easily understood by physicians.

**User's Goal:**
{USER_AIM}

**Analysis Results:**
---
**1. Model & Graph Summary:**
- Final model trained with {len(best_constraints)} interaction constraint groups.
- Optimized Graph Features: {G.number_of_nodes()}
- Optimized Graph Interactions: {G.number_of_edges()}

**2. Graph Topology Analysis:**
- Graph Density: {density_str}
- Connectivity: {connectivity_str}
- Top 5 Most Central Features (by Degree):
{top_5_degree_str}
- Top 5 Bridging Features (by Betweenness):
{top_5_betweenness_str}

**3. Top 5 Significant Feature Interactions:**
{top_5_interactions_str}
---

**Instructions for Report Generation:**

Generate a Markdown report with the following three sections. Use the provided analysis results and user's goal to inform your writing.

### 1. Executive Summary of Results
Provide a high-level summary of the key quantitative findings. Mention the model's complexity (features, interactions) and highlight the most central features from the network analysis. Keep this section brief and data-focused.

### 2. Clinical Interpretation
Synthesize the findings into a cohesive clinical narrative.
- Explain the significance of the most central features in the context of ICU mortality.
- Interpret the top 5 feature interactions, discussing how they jointly contribute to patient risk.
- Explain what the overall graph structure (e.g., density, connectivity) implies.

### 3. Clinical & Research Suggestions
Based on the analysis and the user's goal, provide actionable suggestions.
- What clinical insights can be derived from these findings?
- What are potential implications for patient monitoring or treatment?
- What are promising avenues for future research?

Ensure the language is professional and tailored for a clinical audience.
"""
    report_agent = Agent(
        model=LLM_MODEL,
        instructions="You are a medical analyst writing a detailed clinical report in Markdown."
    )
    
    final_report = report_agent.run(report_prompt).content
    
    os.makedirs('results', exist_ok=True)
    with open(report_filename, 'w') as f:
        f.write(final_report)