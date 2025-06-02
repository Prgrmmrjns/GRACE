import os
import csv
import networkx as nx
from params import DATASET_PATH, DATASET_NAME, TARGET_COL, LLM_MODEL
from typing import List, Iterator
from agno.agent import Agent, RunResponse
from agno.knowledge.arxiv import ArxivKnowledgeBase
from agno.storage.json import JsonStorage
from agno.workflow import Workflow
from graph_utils import ResearchReport, BatchFeatureInteractions, Keywords

def get_feature_names_and_description():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        feature_names = [col for col in header if col.lower() != TARGET_COL.lower()]
    with open(f"dataset_info/{DATASET_NAME.lower()}_info.txt", 'r') as f:
        dataset_description = f.read().strip()
    return feature_names, dataset_description

def verify_and_clean_interactions(all_feature_interactions):
    """
    Verify feature interactions and remove any self-loops.
    Returns cleaned interactions and prints warnings for any issues found.
    """
    cleaned_interactions = []
    self_loops_found = 0
    
    for interaction in all_feature_interactions:
        original_count = len(interaction['interactions'])
        # Remove self-loops
        cleaned_interaction_list = [target for target in interaction['interactions'] if target != interaction['feature']]
        
        if len(cleaned_interaction_list) < original_count:
            self_loops_found += 1
            print(f"WARNING: Removed self-loop for feature '{interaction['feature']}' (was interacting with itself)")
        
        # Create new interaction dict with cleaned list
        cleaned_interaction = {
            'feature': interaction['feature'],
            'interactions': cleaned_interaction_list
        }
        cleaned_interactions.append(cleaned_interaction)
    
    if self_loops_found > 0:
        print(f"VERIFICATION: Found and removed {self_loops_found} self-loops from feature interactions")
    else:
        print("VERIFICATION: No self-loops detected - all interactions are valid")
    
    return cleaned_interactions

# --- Workflows ---
class NoMemoryWorkflow(Workflow):
    def get_workflow_session(self):
        ws = super().get_workflow_session()
        ws.memory = None
        return ws

class KeywordGenerationWorkflow(NoMemoryWorkflow):
    description: str = "Generates relevant keywords for literature search based on dataset and prediction task."

    def run(self, feature_names: List[str], dataset_description: str, recreate_search: bool = False) -> Iterator[RunResponse]:
        cache_key = f"keywords_{DATASET_NAME}"
        if not recreate_search and self.session_state.get(cache_key):
            keywords = self.session_state[cache_key]
            yield RunResponse(run_id=self.run_id, content={"step": "keywords", "keywords": keywords})
            return
        
        agent = Agent(model=LLM_MODEL, response_model=Keywords)
        
        prompt = f"""Based on the following dataset description, features, and prediction task, generate up to 10 relevant keywords for searching medical/scientific literature.

Dataset Description: {dataset_description}
Available Features: {feature_names}
Prediction Task: Predicting {TARGET_COL}

Instructions:
- Generate keywords that would help find literature about disease mechanisms, pathophysiology, and risk factors related to the prediction task
- Focus on medical terms, disease processes, organ systems, and biological mechanisms
- Include both specific and general terms that could reveal connections between the input features and the target outcome
- Keywords should be suitable for searching medical/scientific databases like ArXiv, PubMed, etc.
Return a list of up to 10 relevant keywords."""

        resp = agent.run(prompt)
        keywords = resp.content.keywords
        print(f'Generated keywords: {keywords}')
        self.session_state[cache_key] = keywords
        yield RunResponse(run_id=self.run_id, content={"step": "keywords", "keywords": keywords})

class ResearchWorkflow(NoMemoryWorkflow):
    description: str = "Conducts extensive research on disease mechanisms using the knowledge base."

    def run(self, keywords: List[str], feature_names: List[str], dataset_description: str, arxiv_kb: ArxivKnowledgeBase, recreate_search: bool = False) -> Iterator[RunResponse]:
        cache_key = f"research_report_{DATASET_NAME}"
        if not recreate_search and self.session_state.get(cache_key):
            cached_report = self.session_state[cache_key]
            if isinstance(cached_report, dict):
                report_data = ResearchReport(**cached_report)
            else:
                report_data = cached_report
            yield RunResponse(run_id=self.run_id, content={"step": "research", "report_data": report_data})
            return
        
        agent = Agent(model=LLM_MODEL, knowledge=arxiv_kb, search_knowledge=True, response_model=ResearchReport)
        
        prompt = f"""Based on the literature in the knowledge base, conduct extensive research on disease mechanisms that connect the input features to the prediction target.

Dataset Description: {dataset_description}
Available Features: {feature_names}
Prediction Task: Predicting {TARGET_COL}
Search Keywords Used: {keywords}

Instructions:
- Search the knowledge base thoroughly for literature related to the prediction task and input features
- Identify key disease mechanisms, pathophysiological processes, and biological pathways
- Explain how different input features might be connected to the target outcome through these mechanisms
- Focus on causal relationships and biological plausibility
- Provide specific citations from the literature to support your findings
- Generate an extensive research report that will serve as the foundation for identifying specific disease mechanisms

Your research should help identify the most relevant biological/medical mechanisms that explain how the input features influence the target outcome."""

        resp = agent.run(prompt)
        report_data = resp.content
        
        self.session_state[cache_key] = report_data
        yield RunResponse(run_id=self.run_id, content={"step": "research", "report_data": report_data})

class FeatureInteractionConstraintsWorkflow(NoMemoryWorkflow):
    description: str = "Identifies feature interactions for every feature based on research findings."

    def run(self, research_report, feature_names, dataset_description, recreate_search: bool = False):
        cache_key = f"feature_interactions_{DATASET_NAME}"
        if not recreate_search and self.session_state.get(cache_key):
            cached_interactions = self.session_state[cache_key]
            yield RunResponse(run_id=self.run_id, content={"step": "feature_interactions", "all_feature_interactions": cached_interactions})
            return
        
        agent = Agent(model=LLM_MODEL, response_model=BatchFeatureInteractions)
        
        # Process features in batches of 20
        batch_size = 20
        all_feature_interactions = []
        
        for i in range(0, len(feature_names), batch_size):
            batch_features = feature_names[i:i+batch_size]
            
            print(f"Processing batch {i//batch_size + 1}: {len(batch_features)} features ({batch_features[:3]}...)")
            prompt = f"""Based on the research findings below, identify feature interactions for each feature in the current batch.

Research Report:
{research_report.report}

Key Findings:
{chr(10).join(research_report.key_findings) if hasattr(research_report, 'key_findings') else 'No key findings available'}

Dataset Description: {dataset_description}
Prediction Task: Predicting {TARGET_COL}

Current Batch Features: {batch_features}
All Available Features: {feature_names}

Instructions:
- For EACH feature in the 'Current Batch Features' list, identify which OTHER features from 'All Available Features' it should interact with.
- A feature should NOT interact with itself (NO SELF-LOOPS ALLOWED).
- CRITICAL: Never include the feature itself in its interaction list.
- Focus on biologically/medically meaningful interactions based on the research findings. Ensure these are scientific FACTS.
- Each feature should interact with 8-15 other features on average to create a dense, interconnected network.
- Consider BOTH direct and indirect biological relationships:
  * Direct: Features within the same biological system (e.g., cardiovascular, renal, metabolic).
  * Indirect: Features across different systems that influence each other (e.g., cardiovascular affecting renal, metabolic affecting cardiovascular).
  * Compensatory: Features that compensate for dysfunction in other systems.
  * Cascade effects: Features that trigger downstream effects in other systems.
- For each feature in the 'Current Batch Features', provide:
  * The feature name (must be one of the 'Current Batch Features').
  * A list of other features it should interact with (use exact feature names from 'All Available Features') - aim for 8-15 interactions per feature.
  * A concise and factual explanation of why these interactions are biologically relevant, based on scientific evidence.
  * Citations from the research supporting these interactions.
- Examples of dense interactions:
  * Heart rate might interact with: blood pressure, vasopressors, lactate, temperature, oxygen saturation, creatinine, BUN, pH, bicarbonate, hemoglobin, white blood cell count, glucose, age, mechanical ventilation.
  * Creatinine might interact with: BUN, urine output, electrolytes, blood pressure, heart rate, fluid balance, hemoglobin, pH, bicarbonate, lactate, vasopressors, diuretics, age, comorbidities.
  * Lactate might interact with: pH, bicarbonate, oxygen levels, blood pressure, heart rate, temperature, glucose, hemoglobin, mechanical ventilation, vasopressors, creatinine, liver function tests.

Return a list of interaction objects. This list should ONLY contain interaction objects for the features explicitly listed in 'Current Batch Features' for this specific batch. Ensure the response is concise, scientifically accurate, and avoids any redundancies.
The number of interaction objects in your response must exactly match the number of features in 'Current Batch Features'.
"""

            resp = agent.run(prompt)
            batch_interactions_from_llm = resp.content.interactions
            
            # Filter batch_interactions to only include features from the current batch_features
            current_batch_set = set(batch_features)
            
            # Deduplicate and filter
            processed_features_in_this_batch = set()
            final_batch_interactions = []

            for fi in batch_interactions_from_llm:
                if fi.feature in current_batch_set and fi.feature not in processed_features_in_this_batch:
                    # Ensure interactions are not with the feature itself
                    cleaned_interactions = [inter_f for inter_f in fi.interactions if inter_f != fi.feature]
                    
                    # Convert to dictionary format
                    interaction_dict = {
                        'feature': fi.feature,
                        'interactions': cleaned_interactions
                    }
                    final_batch_interactions.append(interaction_dict)
                    processed_features_in_this_batch.add(fi.feature)
            
            # If the LLM didn't return interactions for all features in the batch, log a warning or handle as needed.
            # For now, we'll just proceed with what was returned and correctly filtered.
            if len(final_batch_interactions) < len(batch_features):
                print(f"  WARNING: LLM returned interactions for {len(final_batch_interactions)} features, but batch size was {len(batch_features)}. Missing features: {current_batch_set - processed_features_in_this_batch}")

            all_feature_interactions.extend(final_batch_interactions)
            
            print(f"  Successfully processed and added interactions for {len(final_batch_interactions)} features in batch.")
        
        print(f'Generated interactions for {len(all_feature_interactions)} features total')
        for interaction in all_feature_interactions:
            print(f'  - {interaction["feature"]}: {len(interaction["interactions"])} interactions')
        
        self.session_state[cache_key] = all_feature_interactions
        yield RunResponse(run_id=self.run_id, content={"step": "feature_interactions", "all_feature_interactions": all_feature_interactions})

class MissingFeatureAssignmentWorkflow(NoMemoryWorkflow):
    description: str = "Generates interactions for any missing features."

    def run(self, all_feature_interactions, unassigned_features, feature_names, dataset_description, arxiv_kb, recreate_search: bool = False):
        if not unassigned_features:
            yield RunResponse(run_id=self.run_id, content={"step": "missing_features", "updated_interactions": all_feature_interactions})
            return
            
        cache_key = f"missing_features_{DATASET_NAME}_{len(unassigned_features)}"
        if not recreate_search and self.session_state.get(cache_key):
            cached_interactions = self.session_state[cache_key]
            yield RunResponse(run_id=self.run_id, content={"step": "missing_features", "updated_interactions": cached_interactions})
            return
        
        print(f"Generating interactions for {len(unassigned_features)} missing features...")
        
        # Create interactions for missing features by assigning them to 3-5 random existing features
        import random
        random.seed(42)  # For reproducibility
        
        updated_interactions = all_feature_interactions.copy()
        assigned_features = [interaction['feature'] for interaction in all_feature_interactions]
        
        for feature in unassigned_features:
            # Select 8-12 random features to interact with (dense network)
            available_features = [f for f in feature_names if f != feature]
            num_interactions = min(12, max(8, len(available_features)))
            selected_interactions = random.sample(available_features, num_interactions)
            
            new_interaction = {
                'feature': feature,
                'interactions': selected_interactions
            }
            updated_interactions.append(new_interaction)
            print(f"Assigned {feature} to interact with {len(selected_interactions)} features: {selected_interactions}")
        
        print(f'Success: Generated interactions for all {len(unassigned_features)} missing features')
        
        self.session_state[cache_key] = updated_interactions
        yield RunResponse(run_id=self.run_id, content={"step": "missing_features", "updated_interactions": updated_interactions})

def run_kg_workflows(arxiv_kb: ArxivKnowledgeBase, recreate_search: bool = False):
    """
    Run the comprehensive workflow:
    1. Generate keywords
    2. Conduct research using the knowledge base
    3. Identify mechanisms and assign features
    4. Ensure all features are assigned (iterative process)
    """
    
    # Get feature names and dataset description once at the beginning
    feature_names, dataset_description = get_feature_names_and_description()
    feature_names_set = set(feature_names)
    
    # 1. Generate keywords
    print("Step 1: Generating keywords...")
    storage_dir_keywords = os.path.join("storage", f"{DATASET_NAME}_keywords")
    os.makedirs(storage_dir_keywords, exist_ok=True)
    workflow_storage_keywords = JsonStorage(dir_path=storage_dir_keywords)
    wf_keywords = KeywordGenerationWorkflow(session_id=f"keywords-{DATASET_NAME}", storage=workflow_storage_keywords)
    for resp in wf_keywords.run(feature_names=feature_names, dataset_description=dataset_description, recreate_search=recreate_search):
        if hasattr(resp, 'content') and isinstance(resp.content, dict) and 'keywords' in resp.content:
            keywords = resp.content['keywords']
    
    # 2. Update the knowledge base with the generated keywords and load once
    print(f"Step 2: Loading knowledge base with {len(keywords)} keywords...")
    arxiv_kb.queries = keywords
    arxiv_kb.load(recreate=recreate_search)
    
    # 3. Conduct research
    print("Step 3: Conducting research...")
    storage_dir_research = os.path.join("storage", f"{DATASET_NAME}_research")
    os.makedirs(storage_dir_research, exist_ok=True)
    workflow_storage_research = JsonStorage(dir_path=storage_dir_research)
    wf_research = ResearchWorkflow(session_id=f"research-{DATASET_NAME}", storage=workflow_storage_research)
    
    for resp in wf_research.run(keywords=keywords, feature_names=feature_names, dataset_description=dataset_description, arxiv_kb=arxiv_kb, recreate_search=recreate_search):
        if hasattr(resp, 'content') and isinstance(resp.content, dict) and 'report_data' in resp.content:
            research_report = resp.content['report_data']
    
    # 4. Identify feature interaction constraints
    print("Step 4: Identifying feature interaction constraints...")
    storage_dir_interactions = os.path.join("storage", f"{DATASET_NAME}_interactions")
    os.makedirs(storage_dir_interactions, exist_ok=True)
    workflow_storage_interactions = JsonStorage(dir_path=storage_dir_interactions)
    wf_interactions = FeatureInteractionConstraintsWorkflow(session_id=f"interactions-{DATASET_NAME}", storage=workflow_storage_interactions)
    
    all_feature_interactions = None
    for resp in wf_interactions.run(research_report=research_report, feature_names=feature_names, dataset_description=dataset_description, recreate_search=recreate_search):
        if hasattr(resp, 'content') and isinstance(resp.content, dict) and 'all_feature_interactions' in resp.content:
            all_feature_interactions = resp.content['all_feature_interactions']
    
    if all_feature_interactions is None:
        raise ValueError("Could not generate feature interactions")
    
    # Verify and clean interactions to remove any self-loops
    print("Step 4.5: Verifying and cleaning feature interactions...")
    all_feature_interactions = verify_and_clean_interactions(all_feature_interactions)
    
    # 5. Check for unassigned features and assign them iteratively
    iteration = 0
    
    while True:
        iteration += 1
        print(f"Step 5.{iteration}: Checking for unassigned features...")
        
        # Check which features have interactions defined
        assigned_features = set()
        for interaction in all_feature_interactions:
            if interaction['feature'] in feature_names_set:
                assigned_features.add(interaction['feature'])
        
        unassigned_features = [f for f in feature_names if f not in assigned_features]
        
        if not unassigned_features:
            print("All features have interactions defined!")
            break
        
        print(f"Found {len(unassigned_features)} unassigned features: {unassigned_features}")
        print("Generating interactions for missing features...")
        
        # Use the missing feature assignment workflow
        storage_dir_missing = os.path.join("storage", f"{DATASET_NAME}_missing_features")
        os.makedirs(storage_dir_missing, exist_ok=True)
        workflow_storage_missing = JsonStorage(dir_path=storage_dir_missing)
        wf_missing = MissingFeatureAssignmentWorkflow(session_id=f"missing-{DATASET_NAME}-{iteration}", storage=workflow_storage_missing)
        
        updated_interactions = None
        for resp in wf_missing.run(all_feature_interactions=all_feature_interactions, unassigned_features=unassigned_features, feature_names=feature_names, dataset_description=dataset_description, arxiv_kb=arxiv_kb, recreate_search=recreate_search):
            if hasattr(resp, 'content') and isinstance(resp.content, dict) and 'updated_interactions' in resp.content:
                updated_interactions = resp.content['updated_interactions']
        
        all_feature_interactions = updated_interactions
    
    # Final verification to ensure no self-loops exist
    print("Step 5.final: Final verification of all feature interactions...")
    all_feature_interactions = verify_and_clean_interactions(all_feature_interactions)
    
    # 6. Build the NetworkX graph with feature interactions
    print("Step 6: Building knowledge graph...")
    
    G = nx.DiGraph()
    
    # Add all feature nodes
    for feature in feature_names:
        if not G.has_node(feature):
            G.add_node(feature, entity_type='INPUT_NODE')
    
    # Add edges for feature interactions
    total_edges = 0
    self_loops_prevented = 0
    for interaction in all_feature_interactions:
        if interaction['feature'] in feature_names_set:
            for target_feature in interaction['interactions']:
                if target_feature in feature_names_set:
                    # Prevent self-loops at graph level as final safety check
                    if interaction['feature'] == target_feature:
                        self_loops_prevented += 1
                        print(f"GRAPH WARNING: Prevented self-loop for {interaction['feature']}")
                        continue
                    
                    # Add bidirectional edges for interactions
                    if not G.has_edge(interaction['feature'], target_feature):
                        G.add_edge(interaction['feature'], target_feature, 
                                 relationship="interaction",
                                 explanation="",
                                 citations="")
                        total_edges += 1
                    if not G.has_edge(target_feature, interaction['feature']):
                        G.add_edge(target_feature, interaction['feature'], 
                                 relationship="interaction",
                                 explanation="",
                                 citations="")
                        total_edges += 1
    
    if self_loops_prevented > 0:
        print(f"GRAPH VERIFICATION: Prevented {self_loops_prevented} self-loops at graph building stage")
    
    # Save the graph
    kg_dir = "kg"
    os.makedirs(kg_dir, exist_ok=True)
    kg_path = os.path.join(kg_dir, f"{DATASET_NAME}.graphml")
    nx.write_graphml(G, kg_path)
    
    print(f"Knowledge graph created with {len(all_feature_interactions)} feature interaction definitions and {total_edges} interaction edges")
    return G
