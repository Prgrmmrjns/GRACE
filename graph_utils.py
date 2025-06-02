from typing import List
from pydantic import BaseModel, Field
from enum import Enum


class NodeType(str, Enum):
    INPUT = "INPUT_NODE"
    INTERMEDIATE = "INTERMEDIATE_NODE"
    TARGET = "TARGET_NODE"
    ISOLATED = "ISOLATED_NODE"
    COMBINED = "COMBINED_NODE"

class Edge(BaseModel):
    source: str = Field(description="Source node name")
    target: str = Field(description="Target node name")

class Node(BaseModel):
    name: str = Field(description="Node name")
    node_type: NodeType = Field(description="Type of node (INPUT_NODE, INTERMEDIATE_NODE, TARGET_NODE, or ISOLATED_NODE)")
    edges: List[Edge] = Field(description="List of edges connecting this node to others", default_factory=list)
    context: str = Field(default="", description="Node-level summary for context")

class IntermediateNode(BaseModel):
    name: str = Field(..., description="Name of the intermediate node")
    explanation: str = Field(..., description="Explanation of why this node is relevant and how it connects features to the target, based on literature.")
    citations: List[str] = Field(..., description="List of citations (e.g., Arxiv IDs or paper titles from the knowledge base) that support the explanation.")

class IntermediateNodes(BaseModel):
    nodes: List[IntermediateNode] = Field(..., description="List of intermediate nodes with their names, explanations, and citations.")

class SelectedFeatures(BaseModel):
    features: List[str] = Field(..., description="List of selected features")

# Response model for missing feature assignments as a list of edges
class MissingFeatureAssignments(BaseModel):
    """Response model for assigning missing features to intermediate nodes as a list of edges."""
    edges: List[Edge] = Field(..., description="List of edges from feature (source) to intermediate node (target)")

class FeatureInteractions(BaseModel):
    feature: str = Field(..., description="Feature name")
    interactions: List[str] = Field(..., description="List of interacting features")
    explanation: str = Field(..., description="Explanation of why this feature is relevant and how it connects to other features, based on literature.")
    citations: List[str] = Field(..., description="List of citations (e.g., Arxiv IDs or paper titles from the knowledge base) that support the explanation.")

class ResearchReport(BaseModel):
    """Response model for the research report."""
    report: str = Field(..., description="The research report")
    citations: List[str] = Field(..., description="List of citations (e.g., Arxiv IDs or paper titles from the knowledge base) that support the report.")

class BatchFeatureInteractions(BaseModel):
    """Response model for a batch of feature interactions."""
    interactions: List[FeatureInteractions] = Field(..., description="List of feature interactions")

class Keywords(BaseModel):
    """Response model for the keywords."""
    keywords: List[str] = Field(..., description="List of keywords")


def nx_to_feature_interactions(nx_graph):
    """Extract feature interactions from NetworkX graph."""
    
    feature_interactions = []
    feature_interaction_map = {}
    
    # Collect all interaction edges between features
    for u, v, data in nx_graph.edges(data=True):
        # Check if this is an interaction edge
        if data.get('relationship') == 'interaction':
            # Add to interaction map
            if u not in feature_interaction_map:
                feature_interaction_map[u] = []
            if v not in feature_interaction_map:
                feature_interaction_map[v] = []
            
            feature_interaction_map[u].append(v)
    
    # Convert to simple dictionaries
    for feature, interactions in feature_interaction_map.items():
        feature_interactions.append({
            'feature': feature,
            'interactions': interactions
        })
    
    return feature_interactions