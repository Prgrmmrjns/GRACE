from typing import List
from pydantic import BaseModel, Field

class ResearchReport(BaseModel):
    """Response model for the research report."""
    report: str = Field(..., description="The research report")
    citations: List[str] = Field(..., description="List of citations (e.g., Arxiv IDs or paper titles from the knowledge base) that support the report.")

class Keywords(BaseModel):
    """Response model for the keywords."""
    keywords: List[str] = Field(..., description="List of keywords")

class DiseaseMechanism(BaseModel):
    name: str = Field(..., description="Name of the disease mechanism")
    description: str = Field(..., description="Detailed description of the mechanism and how it relates to the prediction task")
    citations: List[str] = Field(..., description="Supporting citations from literature")

class DiseaseMechanisms(BaseModel):
    mechanisms: List[DiseaseMechanism] = Field(..., description="List of 5-10 central disease mechanisms")

class MechanismFeatureAssignment(BaseModel):
    mechanism: str = Field(..., description="Name of the disease mechanism")
    features: List[str] = Field(..., description="List of features assigned to this mechanism")
    explanation: str = Field(..., description="Explanation of why these features belong to this mechanism")

class MechanismFeatureAssignments(BaseModel):
    assignments: List[MechanismFeatureAssignment] = Field(..., description="Feature assignments for each mechanism")

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