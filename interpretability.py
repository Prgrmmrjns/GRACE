from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union

class EdgeAnnotation(BaseModel):
    """Annotation for an edge in the knowledge graph."""
    feature: str = Field(description="Name of the input feature")
    value: str = Field(description="Value of the feature for the current patient")
    relationship: str = Field(description="Description of how this feature influences the target node")
    target: str = Field(description="Target intermediate node that this feature influences")
    importance: float = Field(description="Importance score of this relationship (positive or negative)")

class IntermediateNodeInsight(BaseModel):
    """Insight about an intermediate node in the knowledge graph."""
    node_name: str = Field(description="Name of the intermediate node")
    conclusion: str = Field(description="Clinical conclusion about this concept for the patient")
    contributing_features: List[str] = Field(description="List of input features contributing to this node")
    prediction_value: float = Field(description="Model's prediction value for this node")
    clinical_interpretation: str = Field(description="Clinical interpretation of this node's value")

class InputFeatureInsight(BaseModel):
    """Insight about a key input feature."""
    feature_name: str = Field(description="Name of the input feature")
    value: Union[float, str] = Field(description="Value of the feature for this patient")
    normal_range: Optional[str] = Field(description="Normal range for this feature, if applicable")
    clinical_significance: str = Field(description="Clinical significance of this feature's value")
    abnormality: Optional[str] = Field(description="Description of abnormality, if present")

class LLMGraphReasoning(BaseModel):
    """Structured reasoning about a patient using a knowledge graph approach."""
    patient_prediction: str = Field(description="Overall prediction for the patient with explanation")
    key_intermediate_features: List[IntermediateNodeInsight] = Field(description="Insights about key intermediate nodes")
    key_input_features: List[InputFeatureInsight] = Field(description="Insights about key input features")
    conclusion: str = Field(description="Clinical conclusion synthesizing all the information")
    edge_annotations: List[EdgeAnnotation] = Field(description="Annotations for edges in the visualization")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "patient_prediction": "This patient has a 75% probability of Alzheimer's Disease based on cognitive assessments and neuroimaging markers.",
                "key_intermediate_features": [
                    {
                        "node_name": "GlobalCognition",
                        "conclusion": "Moderate cognitive impairment present",
                        "contributing_features": ["MMSE", "MOCA", "CDRSB"],
                        "prediction_value": 0.72,
                        "clinical_interpretation": "The patient's cognitive testing results suggest significant impairment in global cognition, consistent with early Alzheimer's Disease."
                    }
                ],
                "key_input_features": [
                    {
                        "feature_name": "MMSE",
                        "value": 21.0,
                        "normal_range": "24-30",
                        "clinical_significance": "Below normal range, indicating cognitive impairment",
                        "abnormality": "Mild-to-moderate cognitive impairment"
                    }
                ],
                "conclusion": "Based on the pattern of cognitive deficits, particularly in memory and executive function, along with supporting biomarkers, this patient's presentation is most consistent with early Alzheimer's Disease.",
                "edge_annotations": [
                    {
                        "feature": "MMSE",
                        "value": "21.0",
                        "relationship": "Reduced MMSE score (21.0) indicates impaired cognitive function across multiple domains, particularly affecting memory encoding and orientation, suggesting hippocampal dysfunction",
                        "target": "GlobalCognition",
                        "importance": 0.85
                    },
                    {
                        "feature": "CDRSB",
                        "value": "4.5",
                        "relationship": "Elevated CDRSB score (4.5) reflects functional impairment in daily activities, indicating frontal-subcortical circuit disruption affecting executive abilities",
                        "target": "GlobalCognition",
                        "importance": 0.75
                    },
                    {
                        "feature": "GlobalCognition",
                        "value": "0.72",
                        "relationship": "Impaired global cognition (0.72) with deficits in memory, orientation, and executive function indicates widespread cortical dysfunction consistent with neurodegenerative disease",
                        "target": "Diagnosis",
                        "importance": 0.9
                    }
                ]
            }
        }
    }

def model_schema():
    """Return the JSON schema for the LLMGraphReasoning model."""
    import json
    schema = LLMGraphReasoning.model_json_schema()
    return json.dumps(schema, indent=2)

import os
import json
import numpy as np
import pandas as pd
import networkx as nx
import pickle
from openai import OpenAI
import re
import dotenv

dotenv.load_dotenv()

def load_patient_data(dataset='mimic', patient_idx=None):
    """Load patient data and models from the model directory"""
    model_dir = f"./models/{dataset}"
    
    # Load data
    with open(f"{model_dir}/data.pkl", 'rb') as f:
        data = pickle.load(f)
    
    # Load models
    with open(f"{model_dir}/final_model.pkl", 'rb') as f:
        final_model = pickle.load(f)
    
    with open(f"{model_dir}/intermediate_models.pkl", 'rb') as f:
        intermediate_models = pickle.load(f)
    
    with open(f"{model_dir}/feature_mappings.pkl", 'rb') as f:
        mappings = pickle.load(f)
        intermediate_to_selected_features = mappings['intermediate_to_selected_features']
    
    # Load graph structure
    graph = nx.read_graphml(f"{model_dir}/pruned_graph.graphml")
    
    # Select random patient if index not provided
    if patient_idx is None:
        patient_idx = np.random.randint(len(data['X_test']))
    
    patient_sample = data['X_test'].iloc[patient_idx]
    true_label = data['y_test'].iloc[patient_idx]
    
    # Convert intermediate predictions to float
    X_train_intermediate_float = data['X_train_intermediate'].astype(float)
    
    return {
        'patient': patient_sample,
        'true_label': true_label,
        'final_model': final_model,
        'intermediate_models': intermediate_models,
        'intermediate_to_features': intermediate_to_selected_features,
        'X_train_intermediate': X_train_intermediate_float,
        'is_binary': data['is_binary'],
        'graph': graph,
        'metric': data['metric']
    }

def get_patient_prediction(patient_data):
    """Get model prediction for the patient"""
    patient = patient_data['patient']
    intermediate_models = patient_data['intermediate_models']
    intermediate_to_features = patient_data['intermediate_to_features']
    final_model = patient_data['final_model']
    is_binary = patient_data['is_binary']
    
    # Generate intermediate node predictions
    intermediate_preds = {}
    for node, model in intermediate_models.items():
        if node in intermediate_to_features:
            features = intermediate_to_features[node]
            if features and all(f in patient.index for f in features):
                try:
                    if is_binary:
                        # Convert to float explicitly
                        proba = float(model.predict_proba(patient[features].values.reshape(1, -1))[0, 1])
                        intermediate_preds[node] = proba
                    else:
                        # For multiclass, store the predicted class probability
                        probas = model.predict_proba(patient[features].values.reshape(1, -1))[0]
                        # Convert to float values
                        intermediate_preds[node] = float(np.argmax(probas))
                except Exception as e:
                    print(f"Error predicting for {node}: {e}")
                    # Default to 0.5 for binary, random for multiclass
                    intermediate_preds[node] = 0.5 if is_binary else 0.0
    
    # Create DataFrame for the final model
    df_preds = pd.DataFrame([intermediate_preds])
    
    # Ensure all values are float
    for col in df_preds.columns:
        df_preds[col] = df_preds[col].astype(float)
    
    # Make final prediction
    try:
        pred_class = final_model.predict(df_preds)[0]
        if is_binary:
            pred_proba = final_model.predict_proba(df_preds)[0, 1]
        else:
            pred_proba = final_model.predict_proba(df_preds)[0]
    except Exception as e:
        print(f"Error in final prediction: {e}")
        pred_class = 0
        pred_proba = 0.5 if is_binary else np.array([0.5, 0.5, 0.5])
    
    return {
        'class': pred_class,
        'probability': pred_proba,
        'intermediate_values': intermediate_preds
    }

def get_feature_influences(patient_data):
    """Extract feature influences on intermediate nodes"""
    patient = patient_data['patient']
    intermediate_models = patient_data['intermediate_models']
    intermediate_to_features = patient_data['intermediate_to_features']
    
    influences = {}
    
    for node, model in intermediate_models.items():
        if node not in intermediate_to_features:
            continue
            
        features = intermediate_to_features[node]
        if not features:
            continue
        
        try:
            # Get feature importance from model
            feature_importance = model.feature_importances_
            
            # Create influence dictionary for this node
            node_influences = []
            for i, feature in enumerate(features):
                if i < len(feature_importance) and feature in patient:
                    value = patient[feature]
                    if not pd.isna(value):
                        # Determine direction based on feature importance
                        direction = "positively influences" if feature_importance[i] > 0 else "negatively influences"
                        
                        node_influences.append({
                            "feature": feature,
                            "value": float(value) if isinstance(value, (int, float)) else str(value),
                            "importance": float(feature_importance[i]),
                            "direction": direction
                        })
            
            if node_influences:
                # Sort by absolute importance
                node_influences.sort(key=lambda x: abs(x["importance"]), reverse=True)
                influences[node] = node_influences
        except Exception as e:
            print(f"Error analyzing features for {node}: {e}")
    
    return influences

def generate_patient_specific_prompt(patient_data, prediction, influences):
    """Generate a prompt for LLM to create patient-specific interpretation"""
    patient = patient_data['patient']
    graph = patient_data['graph']
    dataset = patient_data['metric']
    is_binary = patient_data['is_binary']
    
    # Extract intermediate node descriptions
    intermediate_nodes = []
    for node, data in graph.nodes(data=True):
        if data.get('entity_type') == 'INTERMEDIATE_NODE':
            intermediate_nodes.append({
                "name": node,
                "description": data.get('description', '')
            })
    
    # Format patient features
    patient_features = {}
    for feature, value in patient.items():
        if isinstance(value, (int, float)) and not pd.isna(value):
            patient_features[feature] = float(value)
    
    # Format prediction info
    if is_binary:
        pred_info = {
            "class": int(prediction['class']),
            "probability": float(prediction['probability'])
        }
    else:
        pred_info = {
            "class": int(prediction['class']),
            "probability": prediction['probability'].tolist() if isinstance(prediction['probability'], np.ndarray) else float(prediction['probability'])
        }
    
    # Format intermediate values
    intermediate_values = {}
    for node, value in prediction['intermediate_values'].items():
        intermediate_values[node] = float(value)
    
    # Get schema for structured output
    llm_reasoning_schema = model_schema()
    
    # Get dataset-specific background information
    dataset_info = ""
    try:
        with open(f"dataset_info/{dataset}_info.txt", "r") as file:
            dataset_info = file.read()
    except:
        dataset_info = f"No additional information available for {dataset} dataset."

    # Craft the prompt with extremely specific requirements
    prompt = f"""# Patient-Specific Structured Interpretation Task

## Dataset Background
{dataset_info}

## Your Task
Create a SPECIFIC, PERSONALIZED interpretation of this patient's data using a structured format.

## EDGE ANNOTATION REQUIREMENTS:
For the edge_annotations field, follow these STRICT requirements:

1. MAX LENGTH: 15 words ONLY (non-negotiable)
2. Explain the edge with medically relevant terms
3. NO repetition across annotations - each must be unique
4. Create patient-specific interpretations. No generic, general interpretations. Relate to the feature value.

Your response MUST follow this JSON schema exactly:
{llm_reasoning_schema}
"""
    return prompt

def get_llm_interpretation(prompt, api_key=None):
    """Get interpretation from OpenAI API or use a fallback if API key not available"""
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    
    try:
        print(f"OpenAI API key found: {api_key[:5]}...{api_key[-4:] if len(api_key) > 8 else ''}")
        print("Attempting to call OpenAI API...")
        
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="o3-mini",  
            messages=[
                {"role": "system", "content": "You are a board-certified physician specialized in both neurology and critical care medicine. You provide precise, medically accurate interpretations of patient data with physiological reasoning."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        print("OpenAI API call successful!")
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting LLM interpretation: {e}")
        # Fall back to simple interpretation on API error

def extract_edge_annotations(interpretation):
    """Extract edge annotations from the LLM's interpretation"""
    try:
        # Try to parse the interpretation as JSON
        reasoning_data = json.loads(interpretation)
        
        # Check if the interpretation follows our expected structure
        if 'edge_annotations' in reasoning_data:
            # Extract the edge annotations directly
            return reasoning_data['edge_annotations']
        else:
            # Fall back to text parsing if JSON doesn't have expected structure
            return extract_annotations_from_text(interpretation)
    except json.JSONDecodeError:
        # If interpretation is not valid JSON, fall back to text parsing
        return extract_annotations_from_text(interpretation)

def extract_annotations_from_text(interpretation):
    """Extract edge annotations from text interpretation"""
    annotations = []
    
    lines = interpretation.split('\n')
    for line in lines:
        if '->' in line:
            # Extract annotation in format: "Feature (value) -> specifically indicates -> Clinical meaning"
            parts = line.strip().split(' -> ')
            if len(parts) >= 2:
                source = parts[0].strip()
                relationship = parts[1].strip()
                target = parts[2].strip() if len(parts) > 2 else ""
                
                # Extract feature name and value
                feature_parts = source.split('(')
                feature_name = feature_parts[0].strip()
                feature_value = feature_parts[1].replace(')', '').strip() if len(feature_parts) > 1 else ""
                
                # Default importance from weight if available, otherwise 0.5
                importance = 0.5
                if "weight" in line or "impact" in line:
                    # Try to extract weight/impact value
                    weight_match = re.search(r'(weight|impact):\s*([-+]?\d*\.\d+|\d+)', line)
                    if weight_match:
                        importance = float(weight_match.group(2))
                
                annotations.append({
                    "feature": feature_name,
                    "value": feature_value,
                    "relationship": relationship,
                    "target": target,
                    "importance": importance
                })
    
    return annotations

def interpret_patient(dataset='mimic', patient_idx=None, api_key=None, output_dir="./patient_interpretations"):
    """Main function to interpret a specific patient with complete edge annotation coverage"""
    # Load patient data
    patient_data = load_patient_data(dataset, patient_idx)
    
    # Store dataset name explicitly to avoid confusion
    patient_data['dataset_name'] = dataset  # Add this line to store dataset name directly
    
    # Get prediction
    prediction = get_patient_prediction(patient_data)
    
    # Get feature influences
    influences = get_feature_influences(patient_data)
    
    # Generate initial prompt
    prompt = generate_patient_specific_prompt(patient_data, prediction, influences)
    
    try:
        # Get LLM interpretation
        interpretation = get_llm_interpretation(prompt, api_key)
        
        # Process the interpretation
        try:
            # Parse as JSON
            reasoning_data = json.loads(interpretation)
            
            # Extract edge annotations if present
            if 'edge_annotations' in reasoning_data:
                annotations = reasoning_data['edge_annotations']
                
                # Validate if all edges have annotations
                missing_edges = validate_edge_coverage(patient_data, annotations)
                
                # If there are missing edges, make additional LLM calls
                if missing_edges:
                    print(f"Missing annotations for {len(missing_edges)} edges. Making additional LLM call...")
                    
                    # Generate focused prompt for missing edges
                    focused_prompt = generate_focused_edge_prompt(patient_data, missing_edges)
                    
                    # Get additional annotations
                    additional_interpretation = get_llm_interpretation(focused_prompt, api_key)
                    
                    try:
                        # Parse additional annotations
                        additional_data = json.loads(additional_interpretation)
                        
                        if 'edge_annotations' in additional_data:
                            # Add new annotations to the existing ones
                            annotations.extend(additional_data['edge_annotations'])
                            print(f"Added {len(additional_data['edge_annotations'])} annotations for missing edges")
                            
                            # Update the reasoning data with complete annotations
                            reasoning_data['edge_annotations'] = annotations
                    except json.JSONDecodeError:
                        print("Error parsing additional annotations - falling back to original set")
                
                # Format for visualization - ensure each annotation has the required fields
                for annotation in annotations:
                    if 'importance' not in annotation:
                        annotation['importance'] = 0.5
                        
                # Create human-readable text from structured data
                text_interpretation = format_reasoning_as_text(reasoning_data)
            else:
                # If JSON doesn't have expected structure, generate default text
                text_interpretation = "Interpretation data structure is missing edge annotations."
                annotations = []
        except json.JSONDecodeError:
            # If not valid JSON, use as-is and extract annotations from text
            text_interpretation = interpretation
            annotations = extract_edge_annotations(interpretation)
    except Exception as e:
        print(f"Error in LLM interpretation: {e}")
        text_interpretation = f"Error generating interpretation: {str(e)}"
        
        # Generate fallback annotations for all required edges
        print("Generating fallback annotations...")
        all_edges = []
        for node, features in patient_data['intermediate_to_features'].items():
            for feature in features:
                if feature in patient_data['patient'].index:
                    all_edges.append((feature, node))
                
        target = "Diagnosis" if patient_data.get('dataset_name', '').lower() == 'adni' else "Mortality Risk"
        for node in patient_data['intermediate_to_features'].keys():
            all_edges.append((node, target))
        
        annotations = create_fallback_annotations(patient_data, all_edges)
        print(f"Generated {len(annotations)} fallback annotations")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save prompt
    with open(f"{output_dir}/prompt.txt", 'w') as f:
        f.write(prompt)
    
    # Save raw interpretation
    with open(f"{output_dir}/raw_interpretation.json", 'w') as f:
        f.write(interpretation if 'interpretation' in locals() else "Error generating interpretation")
    
    # Save human-readable interpretation
    with open(f"{output_dir}/patient_specific_interpretation.txt", 'w') as f:
        f.write(text_interpretation)
    
    # Save annotations
    with open(f"{output_dir}/edge_annotations.json", 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"Patient interpretation saved to {output_dir}/patient_specific_interpretation.txt")
    print(f"Edge annotations saved to {output_dir}/edge_annotations.json")
    
    # Return result with validated annotations
    result = {
        'patient': patient_data['patient'],
        'prediction': prediction,
        'influences': influences,
        'interpretation': text_interpretation,
        'annotations': annotations,
        'intermediate_models': patient_data['intermediate_models'],
        'intermediate_to_features': patient_data['intermediate_to_features'],
        'X_train_intermediate': patient_data['X_train_intermediate'],
        'graph': patient_data['graph'],
        'final_model': patient_data['final_model'],
        'is_binary': patient_data['is_binary'],
        'dataset': dataset
    }
    
    return result

def validate_edge_coverage(patient_data, annotations):
    """Validate if all edges have annotations and return missing edges"""
    # Extract graph structure
    patient = patient_data['patient']
    intermediate_to_features = patient_data['intermediate_to_features']
    dataset_name = patient_data.get('dataset_name', '')  # Get dataset name from the explicitly stored field
    
    # Build a simplified graph structure
    edge_set = set()
    
    # Add feature -> intermediate edges
    for node, features in intermediate_to_features.items():
        for feature in features:
            if feature in patient.index:
                edge_set.add((feature, node))
    
    # Add intermediate -> target edges (assuming a single target)
    target = "Diagnosis" if dataset_name.lower() == 'adni' else "Mortality Risk"
    for node in intermediate_to_features.keys():
        edge_set.add((node, target))
    
    # Check which edges have annotations
    annotated_edges = set()
    for annotation in annotations:
        feature = annotation.get('feature')
        target = annotation.get('target')
        if feature and target:
            annotated_edges.add((feature, target))
    
    # Find missing edges
    missing_edges = edge_set - annotated_edges
    
    print(f"Total edges: {len(edge_set)}, Annotated edges: {len(annotated_edges)}, Missing: {len(missing_edges)}")
    
    return list(missing_edges)  # Convert to list for JSON serialization

def generate_focused_edge_prompt(patient_data, missing_edges):
    """Generate a focused prompt just for missing edge annotations"""
    dataset_name = patient_data.get('dataset_name', '')  # Get dataset name from the stored field
    
    # Format missing edges for display
    missing_edges_formatted = []
    for source, target in missing_edges:
        missing_edges_formatted.append({"source": source, "target": target})
    
    # Start with a simplified prompt focused only on edge annotations
    prompt = f"""# Edge Annotation Task - MISSING EDGES ONLY

## Critical Requirement
You must generate precise medical annotations for THESE SPECIFIC EDGES ONLY:

```
{json.dumps(missing_edges_formatted, indent=2)}
```

## EDGE ANNOTATION REQUIREMENTS:
1. MAX LENGTH: 15 words ONLY
2. Use medical reasoning to explain the impact of the start node on the end node
3. Make each annotation completely unique
4. Create patient-specific interpretations. No generic, general interpretations. Relate to the feature value.

## FOLLOW THESE EXACT STRUCTURES:

For {dataset_name} dataset:

## Response Format
Return JSON with an array of edge_annotations objects with these fields:
- feature: The source node (EXACTLY as provided in the missing edges list)
- target: The target node (EXACTLY as provided in the missing edges list)
- relationship: Your 15-word medical reasoning annotation
- importance: A value from 0-1 indicating importance

CRITICAL: Your entire response must be valid JSON containing ONLY annotations for the specified edges.
"""
    
    return prompt

def format_reasoning_as_text(reasoning_data):
    """Convert structured reasoning data to human-readable text format"""
    text = "# Patient-Specific Analysis\n\n"
    
    # Add patient prediction section
    text += f"## Patient Prediction\n{reasoning_data.get('patient_prediction', 'Not available')}\n\n"
    
    # Add key intermediate features
    if 'key_intermediate_features' in reasoning_data:
        text += "## Key Intermediate Features\n\n"
        for feature in reasoning_data['key_intermediate_features']:
            text += f"### {feature.get('node_name', 'Unknown')}\n"
            text += f"**Conclusion:** {feature.get('conclusion', 'Not available')}\n\n"
            text += f"**Clinical Interpretation:** {feature.get('clinical_interpretation', 'Not available')}\n\n"
            
            if 'contributing_features' in feature and feature['contributing_features']:
                text += f"**Contributing Features:** {', '.join(feature['contributing_features'])}\n\n"
            
            text += f"**Prediction Value:** {feature.get('prediction_value', 'N/A')}\n\n"
    
    # Add key input features
    if 'key_input_features' in reasoning_data:
        text += "## Key Input Features\n\n"
        for feature in reasoning_data['key_input_features']:
            text += f"### {feature.get('feature_name', 'Unknown')}\n"
            
            value = feature.get('value', 'N/A')
            normal_range = feature.get('normal_range', '')
            range_text = f" (Normal range: {normal_range})" if normal_range else ""
            
            text += f"**Value:** {value}{range_text}\n\n"
            text += f"**Clinical Significance:** {feature.get('clinical_significance', 'Not available')}\n\n"
            
            if 'abnormality' in feature and feature['abnormality']:
                text += f"**Abnormality:** {feature['abnormality']}\n\n"
    
    # Add conclusion
    if 'conclusion' in reasoning_data:
        text += f"## Conclusion\n{reasoning_data['conclusion']}\n\n"
    
    # Add note about edge annotations
    if 'edge_annotations' in reasoning_data:
        text += f"## Edge Annotations\n"
        text += f"{len(reasoning_data['edge_annotations'])} annotations have been created for the visualization graph.\n\n"
    
    return text

def create_enhanced_interpretation_from_lime(lime_explanation, patient_data):
    """Create an enhanced interpretation using LIME results without needing OpenAI API"""
    patient = patient_data.get('patient', {})
    intermediate_to_features = patient_data.get('intermediate_to_features', {})
    is_binary = patient_data.get('is_binary', True)
    dataset = patient_data.get('dataset', 'unknown')
    
    # Extract top features from LIME
    top_features = lime_explanation["features"][:10]
    
    # Map features to intermediate nodes
    feature_to_node = {}
    for node, features in intermediate_to_features.items():
        if features:
            for feature in features:
                feature_to_node[feature] = node
    
    # Start building the interpretation
    interpretation = "# Patient-Specific Analysis\n\n"
    
    # Add patient overview
    interpretation += "## Patient Overview\n"
    
    # For ADNI dataset (Alzheimer's)
    if dataset.lower() == 'adni':
        # Extract key cognitive scores
        moca = patient.get('MOCA', 'N/A')
        mmse = patient.get('MMSCORE', 'N/A')
        cdr = patient.get('CDGLOBAL', 'N/A')
        
        interpretation += f"This patient has a Montreal Cognitive Assessment (MoCA) score of {moca}, "
        interpretation += f"Mini-Mental State Examination (MMSE) score of {mmse}, "
        interpretation += f"and Clinical Dementia Rating (CDR) of {cdr}.\n\n"
        
    # For MIMIC dataset (ICU)
    elif dataset.lower() == 'mimic':
        # Try to extract some key values that might be present
        age = patient.get('Age', patient.get('age', 'N/A'))
        gender = patient.get('Gender', patient.get('gender', 'N/A'))
        bun = patient.get('BUN', 'N/A')
        
        interpretation += f"This patient (age: {age}, gender: {gender}) "
        if bun != 'N/A':
            interpretation += f"has BUN levels of {bun}, "
        interpretation += "and other key measurements analyzed below.\n\n"
    
    # Add key influencing features section
    interpretation += "## Key Features Influencing Prediction\n\n"
    
    # Group features by intermediate node
    node_to_features = {}
    for feature_info in top_features:
        feature_name = feature_info['name']
        if feature_name in feature_to_node:
            node = feature_to_node[feature_name]
            if node not in node_to_features:
                node_to_features[node] = []
            node_to_features[node].append(feature_info)
    
    # Display features by intermediate node
    for node, features in node_to_features.items():
        interpretation += f"### {node}\n"
        for feature_info in features:
            name = feature_info['name']
            value = feature_info['actual_value']
            weight = feature_info['weight']
            direction = "positively influencing" if weight > 0 else "negatively influencing"
            
            interpretation += f"- **{name}** ({value}) → {direction} → {node} (impact: {weight:.4f})\n"
        
        interpretation += "\n"
    
    # Add summary section
    interpretation += "## Summary of Patient Analysis\n\n"
    
    # Customize based on dataset
    if dataset.lower() == 'adni':
        # Check if we have cognitive scores to make summary more specific
        if 'MMSCORE' in patient and 'CDGLOBAL' in patient:
            mmse = patient['MMSCORE']
            cdr = patient['CDGLOBAL']
            
            if mmse < 24 or cdr > 0:
                interpretation += "This patient shows signs of cognitive impairment based on key cognitive assessments. "
                interpretation += "The combination of test scores suggests changes in memory and executive function typical of patients with cognitive decline. "
            else:
                interpretation += "This patient's cognitive test scores are generally within normal ranges. "
                
            interpretation += "The prediction model has identified specific cognitive measures that most strongly influence the diagnosis.\n"
            
    elif dataset.lower() == 'mimic':
        interpretation += "This analysis highlights the key clinical factors influencing this patient's mortality risk assessment. "
        interpretation += "The prediction is based on specific physiological measurements that have been identified as most important for this individual case.\n"
    
    # Final note on model confidence
    interpretation += "\n**Note:** This interpretation was automatically generated based on LIME feature importance without using an LLM. "
    interpretation += "For a more nuanced clinical interpretation, please consider using the OpenAI API integration.\n"
    
    return interpretation

def create_fallback_annotations(patient_data, missing_edges):
    """Create fallback annotations when LLM fails"""
    dataset_name = patient_data.get('dataset_name', '')
    annotations = []
    
    # Define dataset-specific annotation templates
    if dataset_name.lower() == 'adni':
        templates = [
            "Hippocampal neurogenesis impairment",
            "Amygdala volume reduction",
            "Entorhinal tau accumulation",
            "Prefrontal synaptic pruning",
            "Cholinergic neuron degeneration",
            "Corticobasal tract demyelination",
            "Parietal cortex hypometabolism",
            "Temporal lobe atrophy",
            "White matter tract disconnection",
            "Frontotemporal network disruption"
        ]
    else:  # MIMIC or others
        templates = [
            "Renal tubular epithelial injury",
            "Hepatic lactate clearance impairment",
            "Alveolar-capillary membrane dysfunction",
            "Myocardial oxygen consumption elevation",
            "Cerebral autoregulation disruption",
            "Microvascular endothelial damage",
            "Systemic inflammatory cascade activation",
            "Catecholamine-induced cardiomyopathy",
            "Pulmonary compliance reduction",
            "Glomerular filtration rate depression"
        ]
    
    # Create annotations for each missing edge
    for i, (source, target) in enumerate(missing_edges):
        # Use template index based on hash of edge to ensure consistency
        template_idx = hash(f"{source}_{target}") % len(templates)
        relationship = templates[template_idx]
        
        annotations.append({
            "feature": source,
            "target": target,
            "relationship": relationship,
            "importance": 0.7  # Default importance
        })
    
    return annotations

if __name__ == "__main__":
    # Optional: Set OpenAI API key here if not in environment
    # os.environ["OPENAI_API_KEY"] = "your-api-key"
    
    # Interpret a random patient
    interpret_patient(dataset='mimic') 