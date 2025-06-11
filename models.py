from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# --- Pydantic Models for Knowledge Graph ---
class Node(BaseModel):
    id: str = Field(..., description="Unique identifier for the node (e.g., feature name, mechanism name).")
    node_type: str = Field(..., description="Type of the node (e.g., 'feature', 'disease_mechanism', 'target').")
    attributes: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional attributes for the node.")

class Edge(BaseModel):
    source: str = Field(..., description="ID of the source node.")
    target: str = Field(..., description="ID of the target node.")
    relationship: str = Field(..., description="Description of the relationship (e.g., 'interacts_with', 'part_of_mechanism').")
    attributes: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional attributes like weight or rationale.")

class KnowledgeGraphModel(BaseModel):
    nodes: List[Node]
    edges: List[Edge]
    rationale: str = Field(..., description="A high-level explanation for the current structure of the knowledge graph.") 

    def add_node(self, node: Node):
        if node.id not in [n.id for n in self.nodes]:
            self.nodes.append(node)

    def remove_node(self, node_id: str):
        self.nodes = [n for n in self.nodes if n.id != node_id]
        self.edges = [e for e in self.edges if e.source != node_id and e.target != node_id]

    def add_edge(self, edge: Edge):
        if not self.has_edge(edge.source, edge.target):
            self.edges.append(edge)

    def has_edge(self, source_id: str, target_id: str) -> bool:
        return any(e.source == source_id and e.target == target_id for e in self.edges)

    def remove_edge(self, edge: Edge):
        self.edges = [e for e in self.edges if not (e.source == edge.source and e.target == edge.target and e.relationship == edge.relationship)]

class KnowledgeGraphRefinement(BaseModel):
    nodes_to_add: List[Node] = Field(default_factory=list, description="List of nodes to add to the graph.")
    nodes_to_remove: List[str] = Field(default_factory=list, description="List of node IDs to remove from the graph.")
    edges_to_add: List[Edge] = Field(default_factory=list, description="List of edges to add to the graph.")
    edges_to_remove: List[Edge] = Field(default_factory=list, description="List of edges to remove from the graph.")
    rationale: str = Field(..., description="The rationale for the proposed changes.") 