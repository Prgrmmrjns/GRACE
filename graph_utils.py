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

class IntermediateNodes(BaseModel):
    nodes: List[str] = Field(..., description="List of intermediate node names")

class SelectedFeatures(BaseModel):
    features: List[str] = Field(..., description="List of selected features")

# Response model for missing feature assignments as a list of edges
class MissingFeatureAssignments(BaseModel):
    """Response model for assigning missing features to intermediate nodes as a list of edges."""
    edges: List[Edge] = Field(..., description="List of edges from feature (source) to intermediate node (target)")


def nx_to_node_groups(nx_graph):
    node_groups = {}
    for u, v, data in nx_graph.edges(data=True):
        node_u_data = nx_graph.nodes.get(u, {})
        node_v_data = nx_graph.nodes.get(v, {})
        if node_u_data.get('entity_type') == 'INPUT_NODE' and node_v_data.get('entity_type') == 'INTERMEDIATE_NODE':
            if v not in node_groups:
                node_groups[v] = []
            node_groups[v].append(u)
    return node_groups