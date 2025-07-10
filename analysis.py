import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import shap
import json
import joblib
import os
from agno.agent import Agent
from agno.knowledge.arxiv import ArxivKnowledgeBase
from agno.vectordb.chroma import ChromaDb
from pydantic import BaseModel
from typing import List, Dict
from params import DATASET_NAME, TARGET_COL, DATASET_PATH, EMBEDDING_MODEL, LLM_MODEL, KEYWORDS
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# Analysis parameters
LOAD_AGENT_RESPONSE = False  # Set to False to generate new interpretations

class FeatureInterpretation(BaseModel):
    feature_name: str
    shap_value: float
    mechanism: str
    short_interpretation: str  # For edge labels (max 8 words)
    detailed_interpretation: str
    clinical_significance: str
    biological_pathway: str

class InteractionInterpretation(BaseModel):
    feature1: str
    feature2: str
    interaction_strength: float
    mechanism_link: str
    short_interpretation: str  # For edge labels (max 8 words)
    detailed_biological_rationale: str
    clinical_relevance: str
    pathway_interaction: str

class InterpretationResults(BaseModel):
    dataset: str
    target_condition: str
    feature_interpretations: List[FeatureInterpretation]
    interaction_interpretations: List[InteractionInterpretation]
    overall_assessment: str
    clinical_implications: str
    key_findings: List[str]
    recommendations: List[str]

def load_optimized_kg_and_model(dataset_name):
    """Load optimized KG structure and trained model."""
    # Load optimized KG
    G = nx.read_graphml(f'kg/{dataset_name}_optimized_kg.graphml')
    
    # Load constraint data
    with open(f'kg/{dataset_name}_interaction_constraints.json', 'r') as f:
        constraint_data = json.load(f)
    
    # Load the trained model
    model = joblib.load(f'models/{dataset_name}_grace_model.joblib')
    
    # Load dataset and prepare for model using the exact features the model was trained on
    df = pd.read_csv(f'datasets/{dataset_name}.csv')
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    # Get the exact features the model was trained on
    model_features = model.feature_names_
    X_model = X[model_features]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_model, y, test_size=0.2, random_state=42)
    
    return G, model, X_test, y_test, constraint_data

def calculate_shap_values(model, X_test):
    """Calculate SHAP values for the model."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        # XGBoost multiclass: list of arrays
        shap_array = np.array(shap_values).transpose(1, 2, 0)
        shap_values_signed = np.mean(np.mean(shap_array, axis=0), axis=1)
        shap_values_abs = np.mean(np.mean(np.abs(shap_array), axis=0), axis=1)
        return shap_values_abs, shap_values_signed, explainer
    elif len(shap_values.shape) == 3:
        # LightGBM multiclass: 3D array (samples, features, classes)
        shap_values_signed = np.mean(shap_values, axis=(0, 2))  # Mean across samples and classes
        shap_values_abs = np.mean(np.abs(shap_values), axis=(0, 2))
        return shap_values_abs, shap_values_signed, explainer
    else:
        # Binary classification: 2D array (samples, features)
        shap_values_signed = np.mean(shap_values, axis=0)  # Mean across samples
        shap_values_abs = np.mean(np.abs(shap_values), axis=0)  # Mean abs across samples
        return shap_values_abs, shap_values_signed, explainer

def calculate_shap_interactions(model, X_test, top_n=15):
    """Calculate SHAP interaction values."""
    explainer = shap.TreeExplainer(model)
    
    # Get a subset for interaction calculation (computationally expensive)
    X_subset = X_test.iloc[:min(100, len(X_test))]
    shap_interaction = explainer.shap_interaction_values(X_subset)
    
    # For multiclass, take the mean absolute interaction across classes
    if isinstance(shap_interaction, list):
        shap_interaction = np.mean(np.abs(shap_interaction), axis=0)
    
    # Get top interactions
    n_features = shap_interaction.shape[1]
    interactions = []
    
    for i in range(n_features):
        for j in range(i+1, n_features):
            mean_interaction = np.mean(np.abs(shap_interaction[:, i, j]))
            if mean_interaction > 0:
                interactions.append({
                    'feature1': X_test.columns[i],
                    'feature2': X_test.columns[j],
                    'strength': mean_interaction
                })
    
    # Sort by strength and return top N
    interactions.sort(key=lambda x: x['strength'], reverse=True)
    return interactions[:top_n]

def interpret_with_agent(shap_values_abs, shap_values_signed, shap_interactions, X_test, G, constraint_data):
    """Use agno agent to interpret SHAP values and interactions."""
    
    # Get SHAP values per feature (already averaged)
    feature_importance = list(zip(X_test.columns, shap_values_abs, shap_values_signed))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    # Get feature labels from graph
    feature_labels = {}
    for node in G.nodes():
        if 'labels' in G.nodes[node]:
            feature_labels[node] = G.nodes[node]['labels']
    
    # Check if we should load existing interpretation
    report_file = f'manuscript_files/{DATASET_NAME}_shap_interpretation_report.json'
    if LOAD_AGENT_RESPONSE and os.path.exists(report_file):
        print("Loading existing agent interpretation from file...")
        with open(report_file, 'r') as f:
            existing_report = json.load(f)
            # Convert back to InterpretationResults model
            interpretation = InterpretationResults(**existing_report)
        return interpretation, feature_importance, feature_labels
    
    # Otherwise generate new interpretation
    print("Generating new agent interpretation...")
    
    # Setup knowledge base
    vector_db = ChromaDb(
        collection=f"{DATASET_NAME}_interpretation",
        embedder=EMBEDDING_MODEL,
    )
    knowledge_base = ArxivKnowledgeBase(
        queries=KEYWORDS,
        vector_db=vector_db,
    )
    
    # Create interpretation agent
    agent = Agent(
        model=LLM_MODEL,
        response_model=InterpretationResults,
        knowledge=knowledge_base,
        search_knowledge=True,
        instructions="""You are a medical expert explaining WHY features contribute to disease outcomes.
        For each feature and interaction, provide:
        1. A SHORT explanation (MAXIMUM 8 words) explaining the CAUSE or mechanism
        2. A DETAILED explanation of WHY this feature causes the outcome
        3. The biological MECHANISM behind the relationship
        
        Focus on CAUSALITY and MECHANISMS:
        - WHY does this feature increase/decrease risk?
        - WHAT biological process is happening?
        - HOW do features interact to cause outcomes?
        - WHAT is the underlying pathophysiology?
        
        Short explanations should explain CAUSES like:
        - "Inflammatory damage to organs"
        - "Reduced oxygen delivery causes failure"
        - "Neuronal death from protein aggregation"
        - "Metabolic dysfunction triggers cascade"
        - "Kidney damage reduces filtration"
        
        Always explain the underlying biological WHY, not just what the feature is."""
    )
    
    agent.knowledge.load(recreate=False)
    
    # Create interpretation prompt
    interpretation_prompt = f"""
    **Target Condition:** {TARGET_COL}
    **Dataset:** {DATASET_NAME.upper()}
    
    **Top Features by SHAP Importance (with direction):**
    {chr(10).join([f"- {feat}: SHAP={imp:.4f} (signed={signed:.4f}, Mechanism: {feature_labels.get(feat, 'Unknown')})" 
                   for feat, imp, signed in feature_importance[:10]])}
    
    **Top Feature Interactions:**
    {chr(10).join([f"- {inter['feature1']} × {inter['feature2']}: strength={inter['strength']:.4f}" 
                   for inter in shap_interactions])}
    
    **Constraint Groups:**
    {chr(10).join([f"- Group {i+1} ({label}): {', '.join(group)}" 
                   for i, (group, label) in enumerate(zip(constraint_data['interaction_constraints'], 
                                                          constraint_data['constraint_labels']))])}
    
    Provide interpretations for:
    1. All important features (top 10) with SHORT (max 8 words) and DETAILED interpretations
    2. All feature interactions with SHORT (max 8 words) and DETAILED interpretations
    3. Overall assessment with key findings
    4. Clinical implications and specific recommendations
    
    Remember: SHORT interpretations must be 8 words or less for graph labels!
    """
    
    interpretation = agent.run(interpretation_prompt).content
    
    return interpretation, feature_importance, feature_labels

def create_shap_knowledge_graph(G, shap_values_abs, shap_values_signed, shap_interactions, X_test, 
                               interpretation, feature_importance, feature_labels, constraint_data):
    """Create SHAP-weighted knowledge graph using same style as visualize_knowledge_graph."""
    
    plt.figure(figsize=(24, 16))
    ax = plt.gca()
    ax.set_title(f'SHAP-Weighted Knowledge Graph - {DATASET_NAME.upper()}', fontsize=20, fontweight='bold')
    ax.set_aspect('equal')
    
    # Get constraints and labels
    constraints = constraint_data['interaction_constraints']
    labels = constraint_data['constraint_labels']
    
    # Only include features that are in the model
    model_features = set(X_test.columns)
    filtered_constraints = []
    filtered_labels = []
    for group, label in zip(constraints, labels):
        filtered_group = [f for f in group if f in model_features]
        if filtered_group:
            filtered_constraints.append(filtered_group)
            filtered_labels.append(label)
    
    # Create graph EXACTLY like visualize_knowledge_graph
    KG = nx.Graph()
    all_features = set(feat for group in filtered_constraints for feat in group)
    for group in filtered_constraints:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                KG.add_edge(group[i], group[j])
    KG.add_nodes_from(all_features)
    
    # Create feature to labels map EXACTLY like visualize_knowledge_graph
    feature_to_labels_map = {}
    for node in G.nodes():
        if 'labels' in G.nodes[node] and node in model_features:
            feature_to_labels_map[node] = G.nodes[node]['labels'].split(', ')
    
    # Get unique labels and colors EXACTLY like visualize_knowledge_graph
    flat_labels = [l for sublist in filtered_labels for l in (sublist if isinstance(sublist, list) else [sublist])]
    all_label_names = list(set(flat_labels))
    colors = plt.cm.Set3(range(len(all_label_names)))
    label_color_map = dict(zip(all_label_names, colors))
    
    # Create layout with target at center and features in circle around it
    n_groups = len(filtered_constraints)
    pos = {}
    
    # Add target node at center
    KG.add_node(TARGET_COL)
    pos[TARGET_COL] = (0, 0)
    
    # Add edges from target to all features
    for feature in all_features:
        KG.add_edge(TARGET_COL, feature)
    
    # Arrange all features in a circle around target
    n_features = len(all_features)
    if n_features > 0:
        radius = 3.0  # Fixed radius for all features
        angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False)
        
        # Sort features for consistent positioning
        sorted_features = sorted(list(all_features))
        
        for i, feature in enumerate(sorted_features):
            x = radius * np.cos(angles[i])
            y = radius * np.sin(angles[i])
            pos[feature] = (x, y)
    
    # Get SHAP values for sizing
    feature_shap_dict = dict([(f[0], (f[1], f[2])) for f in feature_importance])
    max_shap = max([abs(v[1]) for v in feature_shap_dict.values()]) if feature_shap_dict else 1
    
    # Draw target node first
    nx.draw_networkx_nodes(KG, pos, nodelist=[TARGET_COL], node_color='lightgreen', 
                         node_size=1500, alpha=0.9, edgecolors='black', linewidths=2)
    
    # Color nodes by mechanism but size by SHAP (like visualize_knowledge_graph)
    single_color_nodes, multi_color_nodes = {}, {}
    for node, node_labels in feature_to_labels_map.items():
        if node in KG.nodes():
            unique_labels = list(set(node_labels))
            if len(unique_labels) > 1:
                multi_color_nodes[node] = [label_color_map[l] for l in unique_labels if l in label_color_map]
            elif len(unique_labels) == 1:
                label = unique_labels[0]
                if label not in single_color_nodes: single_color_nodes[label] = []
                single_color_nodes[label].append(node)
    
    # Draw single-color nodes with equal sizing
    for label, nodes in single_color_nodes.items():
        if label in label_color_map:
            nx.draw_networkx_nodes(KG, pos, nodelist=nodes, node_color=[label_color_map[label]], 
                                 node_size=1200, alpha=0.8, edgecolors='black', linewidths=1)
    
    # Draw multi-color nodes (pie charts) with equal sizing
    if pos and multi_color_nodes:
        radius = 0.08  # Fixed radius for all pie charts
        for node, node_colors in multi_color_nodes.items():
            if node in pos and node_colors:
                x, y = pos[node]
                angles = np.linspace(0, 360, len(node_colors) + 1)
                for i in range(len(node_colors)):
                    ax.add_patch(patches.Wedge((x, y), radius, angles[i], angles[i+1], 
                                             facecolor=node_colors[i], edgecolor='black', 
                                             linewidth=1, zorder=5))
    
    # Draw remaining nodes with equal sizing
    drawn_nodes = set(multi_color_nodes.keys()) | {n for nodes in single_color_nodes.values() for n in nodes}
    remaining_nodes = [n for n in KG.nodes() if n not in drawn_nodes and n != TARGET_COL]
    if remaining_nodes:
        nx.draw_networkx_nodes(KG, pos, nodelist=remaining_nodes, node_color='lightgrey', 
                             node_size=1200, alpha=0.8, edgecolors='black', linewidths=1)
    
    # Get interpretations for edge labels
    interaction_dict = {(i['feature1'], i['feature2']): i['strength'] for i in shap_interactions}
    interp_dict = {}
    for inter in interpretation.interaction_interpretations:
        key = (inter.feature1, inter.feature2)
        interp_dict[key] = inter.short_interpretation
        interp_dict[(inter.feature2, inter.feature1)] = inter.short_interpretation
    
    # Draw edges with SHAP-based scaling to target
    for edge in KG.edges():
        if edge[0] == TARGET_COL or edge[1] == TARGET_COL:
            # Edge to target - scale width by SHAP importance
            feature = edge[1] if edge[0] == TARGET_COL else edge[0]
            if feature in feature_shap_dict:
                shap_abs, shap_signed = feature_shap_dict[feature]
                width = 0.5 + (shap_abs / max_shap) * 4  # Scale edge width by SHAP
                # Color edge by SHAP direction
                if shap_signed > 0:
                    edge_color = 'red'
                else:
                    edge_color = 'blue'
                nx.draw_networkx_edges(KG, pos, [(edge[0], edge[1])], width=width, 
                                     edge_color=edge_color, alpha=0.7)
            else:
                nx.draw_networkx_edges(KG, pos, [(edge[0], edge[1])], width=0.5, 
                                     edge_color='gray', alpha=0.5)
        else:
            # Regular feature-feature edges - highlight if labeled
            if edge in interp_dict or (edge[1], edge[0]) in interp_dict:
                # Highlighted edge with label
                nx.draw_networkx_edges(KG, pos, [(edge[0], edge[1])], width=3, 
                                     edge_color='orange', alpha=0.8)
            else:
                # Regular gray edge
                nx.draw_networkx_edges(KG, pos, [(edge[0], edge[1])], width=0.5, 
                                     edge_color='gray', alpha=0.3)
    
    # Add interaction labels for significant interactions
    for edge in KG.edges():
        if edge in interaction_dict or (edge[1], edge[0]) in interaction_dict:
            strength = interaction_dict.get(edge, interaction_dict.get((edge[1], edge[0]), 0))
            if strength > 0.001 and edge in interp_dict:
                x = (pos[edge[0]][0] + pos[edge[1]][0]) / 2
                y = (pos[edge[0]][1] + pos[edge[1]][1]) / 2
                ax.text(x, y, interp_dict[edge], fontsize=9, 
                       bbox=dict(boxstyle="round,pad=0.4", facecolor="yellow", alpha=0.9, edgecolor='orange', linewidth=2),
                       ha='center', va='center', weight='bold')
    
    # Add node labels EXACTLY like visualize_knowledge_graph
    for node in KG.nodes():
        if node in pos:
            # Just show node name, no SHAP values
            label = node
            
            plt.text(pos[node][0], pos[node][1], label, ha='center', va='center', zorder=10, 
                    fontdict={'size': 10, 'weight': 'bold'}, 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', 
                             linewidth=0.5, boxstyle='round,pad=0.2'))
    
    # Add feature contribution labels for top features
    for feat_interp in interpretation.feature_interpretations[:3]:
        if feat_interp.feature_name in pos:
            # Add label near the node
            x = pos[feat_interp.feature_name][0] + 0.3
            y = pos[feat_interp.feature_name][1] + 0.3
            ax.text(x, y, feat_interp.short_interpretation, fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
                   ha='center', va='center')
    
    # Add SHAP value legend
    legend_elements = [
        patches.Patch(color='red', label='Positive SHAP edge (increases risk)'),
        patches.Patch(color='blue', label='Negative SHAP edge (decreases risk)'),
        patches.Patch(color='orange', label='Labeled interaction edge'),
        patches.Patch(color='gray', label='Regular interaction edge'),
        patches.Circle((0, 0), 0.1, facecolor='lightgreen', edgecolor='black', 
                      label='Target variable'),
        patches.Rectangle((0, 0), 1, 0.1, facecolor='white', edgecolor='black',
                         label='Edge width = SHAP magnitude'),
    ]
    
    # Add disease mechanism legend EXACTLY like visualize_knowledge_graph
    for label, color in sorted(label_color_map.items()):
        legend_elements.append(patches.Patch(color=color, label=label.replace('/', '\n')))
    
    plt.legend(handles=legend_elements, title='Disease Mechanisms', bbox_to_anchor=(1.02, 1), 
              loc='upper left', borderaxespad=0., fontsize=12)
    
    plt.axis('off')
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(f'manuscript_files/images/{DATASET_NAME}_shap_interpreted_graph.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved SHAP interpretation graph to manuscript_files/images/{DATASET_NAME}_shap_interpreted_graph.png")

def save_interpretation_report(interpretation, output_file):
    """Save detailed interpretation report to JSON."""
    # Convert Pydantic model to dict, excluding non-serializable fields
    report_dict = interpretation.model_dump(exclude_unset=True)
    
    # Save to JSON with pretty formatting
    with open(output_file, 'w') as f:
        json.dump(report_dict, f, indent=2, default=str)
    
    print(f"Saved detailed interpretation report to {output_file}")

def run_shap_kg_analysis():
    """Run complete SHAP-based KG analysis with interpretations."""
    print(f"\n=== SHAP-Based Knowledge Graph Analysis for {DATASET_NAME} ===")
    
    # Load optimized KG and model
    print("Loading optimized knowledge graph and model...")
    G, model, X_test, y_test, constraint_data = load_optimized_kg_and_model(DATASET_NAME)
    
    # Calculate SHAP values
    print("Calculating SHAP values...")
    shap_values_abs, shap_values_signed, explainer = calculate_shap_values(model, X_test)
    
    # Calculate SHAP interactions
    print("Calculating SHAP interaction values...")
    shap_interactions = calculate_shap_interactions(model, X_test)
    
    # Get interpretations from agent
    print("Generating clinical interpretations with AI agent...")
    interpretation, feature_importance, feature_labels = interpret_with_agent(
        shap_values_abs, shap_values_signed, shap_interactions, X_test, G, constraint_data
    )
    
    # Save detailed interpretation report
    report_file = f'manuscript_files/{DATASET_NAME}_shap_interpretation_report.json'
    save_interpretation_report(interpretation, report_file)
    
    # Create visualization
    print("Creating SHAP-weighted knowledge graph...")
    create_shap_knowledge_graph(
        G, shap_values_abs, shap_values_signed, shap_interactions, X_test, 
        interpretation, feature_importance, feature_labels, constraint_data
    )
    
    print("\n=== Analysis Complete ===")
    print(f"Results saved to:")
    print(f"  - Graph: manuscript_files/images/{DATASET_NAME}_shap_interpreted_graph.png")
    print(f"  - Report: {report_file}")

if __name__ == "__main__":
    run_shap_kg_analysis() 