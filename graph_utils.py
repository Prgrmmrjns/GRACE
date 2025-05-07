from typing import List, Dict
from pydantic import BaseModel, Field
from enum import Enum
import networkx as nx
import lightgbm as lgb

class Removed_Nodes_List(BaseModel):
    removed_nodes: List[str] = Field(description="List of removed nodes")

class NodeType(str, Enum):
    INPUT = "INPUT_NODE"
    INTERMEDIATE = "INTERMEDIATE_NODE"
    TARGET = "TARGET_NODE"
    ISOLATED = "ISOLATED_NODE"
    COMBINED = "COMBINED_NODE"

    @classmethod
    def _missing_(cls, value):
        """Handle missing values by defaulting to INTERMEDIATE"""
        print(f"WARNING: Unknown node type '{value}', defaulting to INTERMEDIATE_NODE")
        return cls.INTERMEDIATE

class Edge(BaseModel):
    source: str = Field(description="Source node name")
    target: str = Field(description="Target node name")

class Node(BaseModel):
    name: str = Field(description="Node name")
    node_type: NodeType = Field(description="Type of node (INPUT_NODE, INTERMEDIATE_NODE, TARGET_NODE, or ISOLATED_NODE)")
    edges: List[Edge] = Field(description="List of edges connecting this node to others", default_factory=list)
    context: str = Field(default="", description="Node-level summary for context")

class KnowledgeGraph(BaseModel):
    nodes: List[Node] = Field(description="List of all nodes in the graph")
    isolated_nodes: List[Node] = Field(description="List of isolated nodes", default_factory=list)

class IntermediateNodes(BaseModel):
    nodes: List[str] = Field(..., description="List of intermediate node names")

class SelectedFeatures(BaseModel):
    features: List[str] = Field(..., description="List of selected features")

# Response model for missing feature assignments as a list of edges
class MissingFeatureAssignments(BaseModel):
    """Response model for assigning missing features to intermediate nodes as a list of edges."""
    edges: List[Edge] = Field(..., description="List of edges from feature (source) to intermediate node (target)")

def get_active_nodes(graph: KnowledgeGraph, removed_nodes: set) -> List[str]:
    """Get list of active input nodes from the graph excluding removed nodes"""
    active_input_nodes = [n.name for n in graph.nodes if n.node_type == NodeType.INPUT and n.name not in removed_nodes]
    return active_input_nodes

def implement_graph_changes(graph: KnowledgeGraph, changes: dict) -> KnowledgeGraph:
    """
    Implement suggested changes to the current graph structure.
    """
    # Create a copy of the current graph
    current_nodes = {node.name: node for node in graph.nodes}
    
    # Remove nodes
    if "remove_nodes" in changes:
        # Actually remove the nodes instead of just marking them as isolated
        nodes_to_remove = set(changes["remove_nodes"])
        
        # Remove the nodes from the active nodes list
        graph.nodes = [n for n in graph.nodes if n.name not in nodes_to_remove]
        
        # Remove any edges that reference the removed nodes
        for node in graph.nodes:
            node.edges = [e for e in node.edges if e.source not in nodes_to_remove and e.target not in nodes_to_remove]
    
    # Add new nodes
    if "add_nodes" in changes:
        for node_data in changes["add_nodes"]:
            new_node = Node(**node_data)
            graph.nodes.append(new_node)
    
    # Remove edges
    if "remove_edges" in changes:
        for edge in changes["remove_edges"]:
            source = edge["source"]
            target = edge["target"]
            if source in current_nodes:
                node = current_nodes[source]
                node.edges = [e for e in node.edges if e.target != target]
    
    # Add new edges
    if "add_edges" in changes:
        for edge_data in changes["add_edges"]:
            source = edge_data["source"]
            if source in current_nodes:
                new_edge = Edge(**edge_data)
                current_nodes[source].edges.append(new_edge)
    
    return graph

def nx_to_knowledgegraph(G_nx: nx.Graph) -> KnowledgeGraph:
    """Convert NetworkX graph to KnowledgeGraph Pydantic model"""
    nodes = []
    isolated_nodes = []
    
    # Convert to directed graph if it isn't already
    if not isinstance(G_nx, nx.DiGraph):
        G_nx = nx.DiGraph(G_nx)
    
    for node, data in G_nx.nodes(data=True):
        # Clean up node type string (remove quotes if present)
        raw_node_type = data.get('entity_type', NodeType.INTERMEDIATE.value)
        node_type = NodeType(raw_node_type)
            
        edges_for_node = []
        
        # Get all edges for this node
        for _, tgt, _ in G_nx.out_edges(node, data=True):
            
            edges_for_node.append(Edge(
                source=node.strip('"'),
                target=tgt.strip('"'),
            ))
        
        
        # Create the node
        new_node = Node(
            name=node.strip('"'),
            node_type=node_type,
            edges=edges_for_node,
        )
        
        # If it's marked as isolated in the graphml, add to isolated_nodes
        if node_type == NodeType.ISOLATED:
            isolated_nodes.append(new_node)
        else:
            nodes.append(new_node)
    
    return KnowledgeGraph(nodes=nodes, isolated_nodes=isolated_nodes)

def create_knowledge_graph(graph: KnowledgeGraph, target_col: str) -> nx.DiGraph:
    """Create a NetworkX graph from the KnowledgeGraph model"""
    G = nx.DiGraph()

    # Add target node first
    G.add_node(
        target_col,
        entity_type=NodeType.TARGET.value,
        description=f"Target variable indicating {target_col}"
    )

    # Add nodes and edges from the LLM suggestions
    for node in graph.nodes:
        # Add node with attributes
        G.add_node(
            node.name,
            entity_type=node.node_type.value,
            description=node.description
        )
        # Add edges
        for edge in node.edges:
            G.add_edge(
                edge.source,
                edge.target,
            )

    # Ensure every intermediate node connects to the target node
    intermediate_nodes = [n for n, d in G.nodes(data=True) 
                        if d.get('entity_type') == NodeType.INTERMEDIATE.value]
    for node in intermediate_nodes:
        if not G.has_edge(node, target_col):
            G.add_edge(node, target_col)
    return G