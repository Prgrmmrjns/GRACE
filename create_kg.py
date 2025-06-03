import os
import csv
import networkx as nx
from params import DATASET_PATH, DATASET_NAME, TARGET_COL, LLM_MODEL
from typing import List, Iterator
from agno.agent import Agent, RunResponse
from agno.knowledge.arxiv import ArxivKnowledgeBase
from agno.storage.json import JsonStorage
from agno.workflow import Workflow
from graph_utils import ResearchReport, Keywords, DiseaseMechanisms, MechanismFeatureAssignments
import warnings
warnings.filterwarnings("ignore")


def get_feature_names_and_description():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        feature_names = [col for col in header if col.lower() != TARGET_COL.lower()]
    with open(f"dataset_info/{DATASET_NAME.lower()}_info.txt", 'r') as f:
        dataset_description = f.read().strip()
    return feature_names, dataset_description

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
            report_data = ResearchReport(**cached_report)
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

class DiseaseMechanismWorkflow(NoMemoryWorkflow):
    description: str = "Identifies 5-10 central disease mechanisms for the prediction task."

    def run(self, research_report, feature_names, dataset_description, recreate_search: bool = False):
        cache_key = f"disease_mechanisms_{DATASET_NAME}"
        if not recreate_search and self.session_state.get(cache_key):
            cached_mechanisms = self.session_state[cache_key]
            yield RunResponse(run_id=self.run_id, content={"step": "mechanisms", "mechanisms": cached_mechanisms})
            return
        
        agent = Agent(model=LLM_MODEL, response_model=DiseaseMechanisms)
        
        prompt = f"""Based on the research report below, identify 5-10 central disease mechanisms that are most relevant for the prediction task.

Research Report:
{research_report.report}

Citations:
{chr(10).join(research_report.citations)}

Dataset Description: {dataset_description}
Available Features: {feature_names}
Prediction Task: Predicting {TARGET_COL}

Instructions:
- Identify 5-10 core disease mechanisms/pathways that are central to the prediction task
- Each mechanism should represent a major biological system or process (e.g., cardiovascular dysfunction, metabolic dysregulation, inflammatory response, etc.)
- Mechanisms should be broad enough to encompass multiple features but specific enough to be medically meaningful
- Focus on mechanisms that have strong literature support and clear connections to the target outcome
- Provide detailed descriptions of how each mechanism relates to the prediction task
- Include supporting citations from the research

Return 5-10 disease mechanisms that will serve as the foundation for organizing feature interactions."""

        resp = agent.run(prompt)
        mechanisms = resp.content.mechanisms
        print(f'Identified {len(mechanisms)} disease mechanisms:')
        for mech in mechanisms:
            print(f'  - {mech.name}')
        
        self.session_state[cache_key] = mechanisms
        yield RunResponse(run_id=self.run_id, content={"step": "mechanisms", "mechanisms": mechanisms})

class MechanismFeatureAssignmentWorkflow(NoMemoryWorkflow):
    description: str = "Assigns features to disease mechanisms."

    def run(self, mechanisms, feature_names, dataset_description, recreate_search: bool = False):
        cache_key = f"mechanism_assignments_{DATASET_NAME}"
        if not recreate_search and self.session_state.get(cache_key):
            cached_assignments = self.session_state[cache_key]
            yield RunResponse(run_id=self.run_id, content={"step": "assignments", "assignments": cached_assignments})
            return
        
        agent = Agent(model=LLM_MODEL, response_model=MechanismFeatureAssignments)
        
        mechanism_list = [f"{m.name}: {m.description}" for m in mechanisms]
        
        prompt = f"""Assign each feature to the most relevant disease mechanisms based on biological/medical knowledge.

Disease Mechanisms:
{chr(10).join(mechanism_list)}

Dataset Description: {dataset_description}
Available Features: {feature_names}
Prediction Task: Predicting {TARGET_COL}

Instructions:
- CRITICAL: EVERY single feature from the Available Features list MUST be assigned to at least one mechanism
- Features should be assigned to multiple mechanisms when biologically relevant (aim for 2-3 mechanisms per feature on average)
- Each mechanism should have many features assigned to it (aim for 15-30 features per mechanism)
- Focus on biological/medical relevance and scientific accuracy
- Be generous with assignments - if a feature could plausibly relate to a mechanism, include it
- Consider both direct and indirect relationships (e.g., metabolic features affecting cardiovascular mechanisms)
- Provide explanations for why features belong to each mechanism

VERIFICATION: The total number of unique features across all mechanism assignments must equal {len(feature_names)} features.

Return assignments of features to each disease mechanism."""

        resp = agent.run(prompt)
        assignments = resp.content.assignments
        
        # Check for unassigned features and retry if needed
        assigned_features = set()
        for assignment in assignments:
            assigned_features.update(assignment.features)
            print(f'{assignment.mechanism}: {len(assignment.features)} features')
        
        unassigned = set(feature_names) - assigned_features
        
        # If there are unassigned features, make a second call to assign them
        if unassigned:
            print(f'Found {len(unassigned)} unassigned features. Assigning them now...')
            
            retry_prompt = f"""The following features were not assigned to any disease mechanism. Please assign each of these features to the most appropriate mechanism(s) from the list below.

Unassigned Features: {list(unassigned)}

Disease Mechanisms:
{chr(10).join(mechanism_list)}

Instructions:
- Assign EVERY feature in the unassigned list to at least one mechanism
- Be generous with assignments - consider indirect relationships
- Features can be assigned to multiple mechanisms if relevant
- Provide explanations for the assignments

Return assignments for the unassigned features."""

            retry_resp = agent.run(retry_prompt)
            retry_assignments = retry_resp.content.assignments
            
            # Merge the retry assignments with original assignments
            assignment_dict = {a.mechanism: a for a in assignments}
            
            for retry_assignment in retry_assignments:
                if retry_assignment.mechanism in assignment_dict:
                    # Add features to existing mechanism
                    existing_features = set(assignment_dict[retry_assignment.mechanism].features)
                    new_features = set(retry_assignment.features)
                    combined_features = list(existing_features | new_features)
                    assignment_dict[retry_assignment.mechanism].features = combined_features
                else:
                    # This shouldn't happen but handle it
                    assignments.append(retry_assignment)
            
            assignments = list(assignment_dict.values())
        
        # Final verification
        final_assigned_features = set()
        for assignment in assignments:
            final_assigned_features.update(assignment.features)
            print(f'{assignment.mechanism}: {len(assignment.features)} features')
        
        final_unassigned = set(feature_names) - final_assigned_features
        if final_unassigned:
            print(f'WARNING: {len(final_unassigned)} features still unassigned: {list(final_unassigned)}')
        else:
            print(f'SUCCESS: All {len(feature_names)} features assigned to mechanisms')
        
        self.session_state[cache_key] = assignments
        yield RunResponse(run_id=self.run_id, content={"step": "assignments", "assignments": assignments})

def create_interactions_from_mechanisms(assignments, feature_names):
    """Create feature interactions based on mechanism assignments."""
    feature_interactions = []
    
    # Create a mapping of features to their mechanisms
    feature_to_mechanisms = {}
    for assignment in assignments:
        for feature in assignment.features:
            if feature not in feature_to_mechanisms:
                feature_to_mechanisms[feature] = []
            feature_to_mechanisms[feature].append(assignment.mechanism)
    
    # For each feature, find interactions with other features in the same mechanisms
    for feature in feature_names:
        if feature not in feature_to_mechanisms:
            continue
            
        interactions = set()
        
        # Find all features that share at least one mechanism with this feature
        for mechanism in feature_to_mechanisms[feature]:
            for assignment in assignments:
                if assignment.mechanism == mechanism:
                    for other_feature in assignment.features:
                        if other_feature != feature and other_feature in feature_names:
                            interactions.add(other_feature)
        
        if interactions:
            feature_interactions.append({
                'feature': feature,
                'interactions': list(interactions)
            })
    
    return feature_interactions

def run_kg_workflows(arxiv_kb: ArxivKnowledgeBase, recreate_search: bool = False):
    """
    Run the mechanism-based workflow:
    1. Generate keywords
    2. Conduct research using the knowledge base
    3. Identify central disease mechanisms
    4. Assign features to mechanisms
    5. Create interactions based on mechanism assignments
    """
    
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
    
    # 4. Identify disease mechanisms
    print("Step 4: Identifying disease mechanisms...")
    storage_dir_mechanisms = os.path.join("storage", f"{DATASET_NAME}_mechanisms")
    os.makedirs(storage_dir_mechanisms, exist_ok=True)
    workflow_storage_mechanisms = JsonStorage(dir_path=storage_dir_mechanisms)
    wf_mechanisms = DiseaseMechanismWorkflow(session_id=f"mechanisms-{DATASET_NAME}", storage=workflow_storage_mechanisms)
    
    mechanisms = None
    for resp in wf_mechanisms.run(research_report=research_report, feature_names=feature_names, dataset_description=dataset_description, recreate_search=recreate_search):
        if hasattr(resp, 'content') and isinstance(resp.content, dict) and 'mechanisms' in resp.content:
            mechanisms = resp.content['mechanisms']
    
    # 5. Assign features to mechanisms
    print("Step 5: Assigning features to mechanisms...")
    storage_dir_assignments = os.path.join("storage", f"{DATASET_NAME}_assignments")
    os.makedirs(storage_dir_assignments, exist_ok=True)
    workflow_storage_assignments = JsonStorage(dir_path=storage_dir_assignments)
    wf_assignments = MechanismFeatureAssignmentWorkflow(session_id=f"assignments-{DATASET_NAME}", storage=workflow_storage_assignments)
    
    assignments = None
    for resp in wf_assignments.run(mechanisms=mechanisms, feature_names=feature_names, dataset_description=dataset_description, recreate_search=recreate_search):
        if hasattr(resp, 'content') and isinstance(resp.content, dict) and 'assignments' in resp.content:
            assignments = resp.content['assignments']
    
    # 6. Create feature interactions from mechanism assignments
    print("Step 6: Creating feature interactions from mechanisms...")
    all_feature_interactions = create_interactions_from_mechanisms(assignments, feature_names)
    
    # 7. Build the NetworkX graph with feature interactions
    print("Step 7: Building knowledge graph...")
    G = nx.DiGraph()
    
    # Add all feature nodes
    for feature in feature_names:
        if not G.has_node(feature):
            G.add_node(feature, entity_type='INPUT_NODE')
    
    # Add mechanism nodes
    for assignment in assignments:
        mechanism_name = f"MECHANISM_{assignment.mechanism.replace(' ', '_').upper()}"
        G.add_node(mechanism_name, entity_type='INTERMEDIATE_NODE')
    
    # Add edges for feature interactions
    total_edges = 0
    for interaction in all_feature_interactions:
        if interaction['feature'] in feature_names_set:
            for target_feature in interaction['interactions']:
                G.add_edge(interaction['feature'], target_feature, 
                            relationship="interaction",
                            explanation="",
                            citations="")
                total_edges += 1
    
    # Save the graph
    kg_dir = "kg"
    os.makedirs(kg_dir, exist_ok=True)
    kg_path = os.path.join(kg_dir, f"{DATASET_NAME}.graphml")
    nx.write_graphml(G, kg_path)
    
    print(f"Knowledge graph created with {len(mechanisms)} mechanisms, {len(all_feature_interactions)} feature interaction definitions and {total_edges} interaction edges")
    return G
