import os
import csv
import networkx as nx
from params import DATASET_PATH, DATASET_NAME, TARGET_COL, MODEL, LLM_PROVIDER, KEYWORDS, LOAD_AGENT_RESPONSES
from typing import List, Iterator
from graph_utils import NodeType, IntermediateNodes, SelectedFeatures, MissingFeatureAssignments
from agno.agent import Agent, RunResponse
from pydantic import BaseModel, Field
from agno.models.openai import OpenAIChat
from agno.models.ollama import Ollama
from agno.vectordb.pgvector import PgVector, SearchType
from agno.knowledge.arxiv import ArxivKnowledgeBase
from agno.storage.json import JsonStorage
from agno.workflow import Workflow
from agno.embedder.openai import OpenAIEmbedder

# --- Helper functions ---
def get_model():
    if LLM_PROVIDER == "openai":
        return OpenAIChat(id=MODEL, temperature=0)
    else:
        return Ollama(id=MODEL)

def get_feature_names_and_description():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        feature_names = [col for col in header if col.lower() != TARGET_COL.lower()]
    with open(f"dataset_info/{DATASET_NAME.lower()}_info.txt", 'r') as f:
        dataset_description = f.read().strip()
    return feature_names, dataset_description

def get_knowledge_base():
    db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"
    vector_db = PgVector(table_name="articles", db_url=db_url, search_type=SearchType.hybrid, embedder=OpenAIEmbedder())
    return ArxivKnowledgeBase(queries=KEYWORDS, vector_db=vector_db)

# --- Workflows ---
# Base workflow class that disables memory
class NoMemoryWorkflow(Workflow):
    def get_workflow_session(self):
        ws = super().get_workflow_session()
        ws.memory = None
        return ws

class IntermediateNodeFinderWorkflow(NoMemoryWorkflow):
    description: str = "Finds and caches intermediate nodes for a dataset."

    def run(self, recreate_search: bool = False) -> Iterator[RunResponse]:
        cache_key = f"intermediate_nodes_{DATASET_NAME}"
        if not recreate_search and self.session_state.get(cache_key):
            nodes = self.session_state[cache_key]
            yield RunResponse(run_id=self.run_id, content={"step": "intermediate_nodes", "nodes": nodes})
            return
        model = get_model()
        feature_names, dataset_description = get_feature_names_and_description()
        knowledge_base = get_knowledge_base()
        agent = Agent(model=model, knowledge=knowledge_base, search_knowledge=True, response_model=IntermediateNodes)
        prompt = initial_prompt(feature_names, dataset_description, TARGET_COL)
        resp = agent.run(prompt)
        nodes = resp.content.nodes
        print(f'Suggested nodes: {nodes}')
        self.session_state[cache_key] = nodes
        yield RunResponse(run_id=self.run_id, content={"step": "intermediate_nodes", "nodes": nodes})

def initial_prompt(feature_names: List[str], dataset_description: str, target_name: str) -> str:
    return f"""Based on the following research on these keywords: {KEYWORDS}, and the following dataset description and features, identify the most relevant 
mechanisms, organ systems, factors or other entities that can best explain the target variable ({target_name}).
Dataset Description: {dataset_description}
Available Features: {feature_names}
Instructions:
- Propose 5-10 entities that best capture the key mechanisms connecting the features to {target_name}. Make sure each features can be assigned to one of the entities.
- These entities should be highly specific and detailed.
- These entities (intermediate nodes) should be made-up terms (e.g., mechanisms, organ systems, etc.) and must NOT be any of the dataset features.
- Only the input nodes (features) should be from the provided feature list.
- Do NOT invent any input nodes or features.
- Never use the target column ('{target_name}') or any column not in the provided feature list as an input feature.
- Any names not in the provided feature list will be ignored.
- Examples may be: Cardiovascular System, Endocrine System, Inflammation, Laboratory markers, Cognitive function markers.
Return a list of string with the names of the entities."""

def add_features_to_intermediate_node_prompt(entity_name: str, candidate_features: List[str], target_name: str, dataset_description: str) -> str:
    return f"""Given the following research, entity, dataset description, and a list of candidate features, select only those features that are relevant to the entity and for connecting them to {target_name}.
Dataset Description: {dataset_description}
Entity: {entity_name}
Candidate features: {candidate_features}
Instructions:
 - From the candidate features, select only those relevant to the entity.
 - Use the exact feature names provided.
 - Do NOT invent any features or use any names not in the candidate features list.
 - Never use the target column ('{target_name}') or any column not in the provided feature list as an input feature.
 - Any names not in the provided feature list will be ignored.
Return a list of the selected features.
"""

def add_missing_features_prompt(entity_names: list, remaining_features: list, target_name: str, dataset_description: str) -> str:
    """Prompt to assign remaining features to one or more relevant entities, including dataset description."""
    return f"""Given the following research, dataset description, entities, and a list of remaining features, assign each feature to one or more relevant entities. Use the exact feature names provided.
Dataset Description: {dataset_description}
The entities are mechanisms, organ systems, factors or other entities that best connect the input features to the target variable ({target_name}).
Entities: {entity_names}
Remaining features: {remaining_features}
Instructions:
- Only assign features from the provided list.
- Do NOT invent any features or use any names not in the provided list.
- Never use the target column ('{target_name}') or any column not in the provided feature list as an input feature.
- Any names not in the provided feature list will be ignored.
Return a dictionary where the keys are entity names and the values are lists of feature names assigned to each entity. Make sure to assign all features to at least one entity."""

class RAGContext(BaseModel):
    summary: str = Field(..., description="High‑level summary relevant for this node / edge")
    citations: List[str] = Field(..., description="List of literature references that support the summary")

class FeatureAssignmentWorkflow(NoMemoryWorkflow):
    description: str = "Assigns features to each intermediate node and caches the results."

    def run(self, intermediate_nodes: list, recreate_search: bool = False) -> Iterator[RunResponse]:
        model = get_model()
        feature_names, dataset_description = get_feature_names_and_description()
        knowledge_base = get_knowledge_base()
        assigned_features = set()
        features_per_node = {}
        for inter in intermediate_nodes:
            cache_key = f"selected_features_{DATASET_NAME}_{inter}"
            if not recreate_search and self.session_state.get(cache_key):
                feats = self.session_state[cache_key]
            else:
                agent = Agent(model=model, knowledge=knowledge_base, search_knowledge=True, response_model=SelectedFeatures)
                prompt = add_features_to_intermediate_node_prompt(inter, feature_names, TARGET_COL, dataset_description)
                resp = agent.run(prompt)
                feats = resp.content.features
                print(f'selected features for {inter}: {feats}')
                self.session_state[cache_key] = feats
            features_per_node[inter] = feats
            assigned_features.update(feats)
            yield RunResponse(run_id=self.run_id, content={"step": "selected_features", "intermediate_node": inter, "features": feats})

        G = nx.DiGraph()
        G.add_node(TARGET_COL, entity_type=NodeType.TARGET.value)
        for inter, feats in features_per_node.items():
            G.add_node(inter, entity_type=NodeType.INTERMEDIATE.value)
            G.add_edge(inter, TARGET_COL, relationship="evidence")
            for ft in feats:
                G.add_node(ft, entity_type=NodeType.INPUT.value)
                G.add_edge(ft, inter, relationship="evidence")

class MissingFeatureAssignmentWorkflow(NoMemoryWorkflow):
    description: str = "Assigns remaining features to one or more intermediate nodes and caches the result."

    def run(self, intermediate_nodes: list, remaining_features: list, recreate_search: bool = False) -> Iterator[RunResponse]:
        if not remaining_features:
            yield RunResponse(run_id=self.run_id, content={"step": "missing_features", "edges": []})
            return
        model = get_model()
        _, dataset_description = get_feature_names_and_description()
        knowledge_base = get_knowledge_base()
        cache_key = f"missing_features_{DATASET_NAME}"
        if not recreate_search and self.session_state.get(cache_key):
            edges = self.session_state[cache_key]
            yield RunResponse(run_id=self.run_id, content={"step": "missing_features", "edges": edges})
            return
        agent = Agent(model=model, knowledge=knowledge_base, search_knowledge=True, response_model=MissingFeatureAssignments)
        prompt = add_missing_features_prompt(intermediate_nodes, remaining_features, TARGET_COL, dataset_description)
        resp = agent.run(prompt)
        edges = resp.content.edges
        self.session_state[cache_key] = edges
        yield RunResponse(run_id=self.run_id, content={"step": "missing_features", "edges": edges})

def run_kg_workflows(recreate_search: bool = False):
    """
    Run the intermediate node, feature assignment, and missing feature assignment workflows.
    Returns:
        nodes: list of intermediate nodes
        features_per_node: dict mapping intermediate node to features
        missing_edges: list of missing edges
        G: the in-memory networkx.DiGraph built for visualization
    """
    # Use LOAD_AGENT_RESPONSES to control cache usage
    recreate_search = not LOAD_AGENT_RESPONSES


    # 1. Find intermediate nodes
    storage_dir_nodes = os.path.join("storage", f"{DATASET_NAME}_intermediate_nodes")
    os.makedirs(storage_dir_nodes, exist_ok=True)
    workflow_storage_nodes = JsonStorage(dir_path=storage_dir_nodes)
    wf_nodes = IntermediateNodeFinderWorkflow(session_id=f"intermediate-nodes-{DATASET_NAME}", storage=workflow_storage_nodes)
    responses_nodes = wf_nodes.run(recreate_search=recreate_search)
    nodes = None
    for resp in responses_nodes:
        if hasattr(resp, 'content') and isinstance(resp.content, dict) and 'nodes' in resp.content:
            nodes = resp.content['nodes']

    # 2. Assign features to each intermediate node, print remaining, and visualize
    storage_dir_feats = os.path.join("storage", f"{DATASET_NAME}_feature_assignment")
    os.makedirs(storage_dir_feats, exist_ok=True)
    workflow_storage_feats = JsonStorage(dir_path=storage_dir_feats)
    wf_feats = FeatureAssignmentWorkflow(session_id=f"feature-assignment-{DATASET_NAME}", storage=workflow_storage_feats)
    responses_feats = wf_feats.run(intermediate_nodes=nodes, recreate_search=recreate_search)
    features_per_node = {}
    assigned_features = set()
    feature_names, _ = get_feature_names_and_description()
    feature_names_set = set(feature_names)
    # Build the graph for downstream use
    G = nx.DiGraph()
    G.add_node(TARGET_COL, entity_type=NodeType.TARGET.value)
    for resp in responses_feats:
        inter = resp.content['intermediate_node']
        feats = resp.content['features']
        valid_feats = [f for f in feats if f in feature_names_set]
        features_per_node[inter] = valid_feats
        assigned_features.update(valid_feats)
        G.add_node(inter, entity_type=NodeType.INTERMEDIATE.value)
        G.add_edge(inter, TARGET_COL, relationship="evidence")
        for ft in valid_feats:
            G.add_node(ft, entity_type=NodeType.INPUT.value)
            G.add_edge(ft, inter, relationship="evidence")

    # 3. Assign remaining features iteratively until all are assigned or no progress
    remaining_features = [f for f in feature_names if f not in assigned_features]
    storage_dir_missing = os.path.join("storage", f"{DATASET_NAME}_missing_feature_assignment")
    os.makedirs(storage_dir_missing, exist_ok=True)
    workflow_storage_missing = JsonStorage(dir_path=storage_dir_missing)
    wf_missing = MissingFeatureAssignmentWorkflow(session_id=f"missing-features-{DATASET_NAME}", storage=workflow_storage_missing)
    missing_edges = []
    while remaining_features:
        responses_missing = wf_missing.run(intermediate_nodes=list(features_per_node.keys()), remaining_features=remaining_features, recreate_search=recreate_search)
        new_edges = []
        for resp in responses_missing:
            valid_edges = []
            for edge in resp.content['edges']:
                valid_edges.append(edge)
            new_edges.extend(valid_edges)

        for edge in new_edges:
            if edge.source not in assigned_features:
                assigned_features.add(edge.source)
            G.add_node(edge.source, entity_type=NodeType.INPUT.value)
            G.add_edge(edge.source, edge.target, relationship="evidence")
        missing_edges.extend(new_edges)
        remaining_features = [f for f in feature_names if f not in assigned_features]

    # --- Check for any features still not assigned ---
    assigned_features_final = set(assigned_features)
    for edge in missing_edges:
        assigned_features_final.add(edge.source)
    # Save the graph to disk for reproducibility
    kg_dir = "kg"
    os.makedirs(kg_dir, exist_ok=True)
    kg_path = os.path.join(kg_dir, f"{DATASET_NAME}.graphml")
    nx.write_graphml(G, kg_path)
    return G

if __name__ == "__main__":
    run_kg_workflows(recreate_search=False)