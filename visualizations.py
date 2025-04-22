import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
from graph_utils import nx_to_knowledgegraph
from matplotlib.lines import Line2D
import hashlib

# Global dictionary to store initial node positions
_initial_positions = {}

# Common node colors and sizes - INCREASED NODE SIZES
NODE_COLORS = {
    'INPUT_NODE': '#ff7f0e',    # Orange for input nodes
    'INTERMEDIATE_NODE': '#1f77b4',  # Blue for intermediate nodes
    'TARGET_NODE': '#2ca02c',   # Green for target node
    'ISOLATED_NODE': '#808080', # Grey for isolated nodes
}

# Significantly increased node sizes for better visibility
NODE_SIZES = {
    'INPUT_NODE': 4000,         # Increased from 1500
    'INTERMEDIATE_NODE': 14000,  # Increased from 6000
    'TARGET_NODE': 18000,       # Increased from 8000
    'ISOLATED_NODE': 2000,      # Increased from 1500
}

# Common edge styles - INCREASED LINE WIDTH
EDGE_STYLES = {
    'input_to_intermediate': {
        'color': '#FFA07A',  # Light salmon
        'style': 'solid',
        'width': 2.0,        # Increased from 1.0
        'alpha': 0.6         # Increased from 0.4
    },
    'intermediate_to_target': {
        'color': '#4169E1',  # Royal blue
        'style': 'solid',
        'width': 3.5,        # Increased from 2.5
        'alpha': 0.9         # Increased from 0.8
    }
}

def setup_visualization(figsize=(30, 30)):
    """Set up the matplotlib figure for visualization"""
    plt.figure(figsize=figsize)

def draw_nodes_by_type(G, pos, node_types, removed_nodes=None):
    """Draw nodes by their types with appropriate colors and sizes"""
    removed_nodes = removed_nodes or set()
    
    # First draw removed nodes if they exist
    removed_nodelist = [n for n in removed_nodes if n in pos]
    if removed_nodelist:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=removed_nodelist,
            node_color=NODE_COLORS['ISOLATED_NODE'],
            node_size=NODE_SIZES['ISOLATED_NODE'],
            alpha=0.5  # More transparent for removed nodes
        )
    
    # Then draw active nodes by type
    for node_type in node_types:
        nodelist = [n for n, d in G.nodes(data=True) 
                   if d.get('entity_type') == node_type and n in pos and n not in removed_nodes]
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=nodelist,
            node_color=NODE_COLORS[node_type],
            node_size=NODE_SIZES[node_type],
            alpha=0.7
        )

def draw_node_labels(G, pos, node_types, removed_nodes=None, font_sizes=None):
    """Draw node labels with different styles for each node type."""
    if removed_nodes is None:
        removed_nodes = set()
    
    if font_sizes is None:
        font_sizes = {
            'TARGET_NODE': 20,
            'INTERMEDIATE_NODE': 16,
            'INPUT_NODE': 12,
            'ISOLATED_NODE': 8
        }
    
    # Handle the case where node_types is a list instead of a dictionary
    if isinstance(node_types, list):
        # Convert list of node types to a dictionary mapping node types to nodes
        node_type_dict = {}
        for node_type in node_types:
            node_type_dict[node_type] = [n for n, d in G.nodes(data=True) 
                                        if d.get('entity_type') == node_type]
        node_types = node_type_dict
    
    # Draw labels for each node type using its color
    for node_type, nodes in node_types.items():
        if node_type in font_sizes:
            font_size = font_sizes[node_type]
            node_labels = {}
            
            for node in nodes:
                if node not in removed_nodes and G.has_node(node):
                    # For intermediate nodes, remove LIME values from label
                    if node_type == 'INTERMEDIATE_NODE':
                        # Get only the base node name without LIME value
                        label = node.split(' (LIME')[0] if ' (LIME' in node else node
                        node_labels[node] = label
                    else:
                        node_labels[node] = node
            
            nx.draw_networkx_labels(
                G, pos, 
                labels=node_labels,
                font_size=font_size, 
                font_weight='bold',
                font_color='black'
            )

def draw_edge_labels(G, pos, removed_nodes=None):
    """Draw edge labels showing relationships with custom styled boxes"""
    removed_nodes = removed_nodes or set()
    
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        if u not in removed_nodes and v not in removed_nodes and 'relationship' in data:
            edge_labels[(u, v)] = data['relationship']
    
    # Instead of using nx.draw_networkx_edge_labels, draw each label individually with custom styling
    for (source, target), label in edge_labels.items():
        # Skip if source or target are removed nodes or not in pos
        if source in removed_nodes or target in removed_nodes:
            continue
        if source not in pos or target not in pos:
            continue
            
        # Format relationship into multiple lines if needed
        words = label.split()
        formatted_rel = ""
        line_length = 0
        line_word_count = 0
        
        for word in words:
            if line_word_count >= 3 or line_length > 15:  # Tight formatting
                formatted_rel += "\n"
                line_length = 0
                line_word_count = 0
            formatted_rel += word + " "
            line_length += len(word) + 1
            line_word_count += 1
        
        # Determine edge color based on edge type
        if G.nodes[source].get('entity_type') == 'INPUT_NODE':
            edge_color = EDGE_STYLES['input_to_intermediate']['color']
        elif G.nodes[source].get('entity_type') == 'INTERMEDIATE_NODE':
            edge_color = EDGE_STYLES['intermediate_to_target']['color']
        else:
            edge_color = '#888888'  # Default gray
        
        # Create custom bbox properties with the edge's color
        bbox_props = dict(
            boxstyle="round,pad=0.3", 
            facecolor="white", 
            edgecolor=edge_color, 
            alpha=0.9
        )
        
        # Calculate position (midpoint of the edge)
        x1, y1 = pos[source]
        x2, y2 = pos[target]
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        
        # Draw the text with custom bbox
        plt.text(x, y, formatted_rel.strip(), 
                fontsize=10, 
                color='#555555',
                horizontalalignment='center', 
                verticalalignment='center',
                bbox=bbox_props, 
                zorder=2)  # zorder ensures labels are on top of other elements

def add_legend():
    """Add a legend to the visualization with specified elements"""
    
    legend_elements = []
    
# Add node types to legend
    legend_elements.extend([
        Line2D([0], [0], marker='o', color='w', markerfacecolor=NODE_COLORS['INPUT_NODE'], 
                markersize=15, label='Input Features'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=NODE_COLORS['INTERMEDIATE_NODE'], 
                markersize=15, label='Intermediate Concepts'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=NODE_COLORS['TARGET_NODE'], 
                markersize=15, label='Target Outcome'),
    ])
    
    # Add removed and isolated nodes
    legend_elements.extend([
        Line2D([0], [0], marker='o', color='w', markerfacecolor=NODE_COLORS['ISOLATED_NODE'], 
                markersize=12, alpha=0.6, label='Isolated Nodes'),
    ])
    
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)

def finalize_visualization(output_path):
    """Finalize and save the visualization"""
    
    plt.axis('off')
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the visualization
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def get_nodes_by_type(G):
    """Extract nodes by type from the graph"""
    target_nodes = [n for n, d in G.nodes(data=True) if d.get('entity_type') == 'TARGET_NODE']
    intermediate_nodes = [n for n, d in G.nodes(data=True) if d.get('entity_type') == 'INTERMEDIATE_NODE']
    input_nodes = [n for n, d in G.nodes(data=True) if d.get('entity_type') == 'INPUT_NODE']
    isolated_nodes = [n for n, d in G.nodes(data=True) if d.get('entity_type') == 'ISOLATED_NODE']
    
    return target_nodes, intermediate_nodes, input_nodes, isolated_nodes

def create_radial_layout(target_nodes, intermediate_nodes, input_nodes, isolated_nodes=None):
    """Create a radial layout with nodes positioned in concentric circles"""
    pos = {}
    
    # Place target node at center
    if target_nodes:
        pos[target_nodes[0]] = (0, 0)
    
    # Place intermediate nodes in a circle
    radius_intermediate = 2.0
    for i, node in enumerate(intermediate_nodes):
        angle = 2 * np.pi * i / len(intermediate_nodes)
        pos[node] = (
            radius_intermediate * np.cos(angle),
            radius_intermediate * np.sin(angle)
        )
    
    # Place input nodes in a larger circle
    radius_input = 4.0
    for i, node in enumerate(input_nodes):
        angle = 2 * np.pi * i / len(input_nodes)
        pos[node] = (
            radius_input * np.cos(angle),
            radius_input * np.sin(angle)
        )
    
    # Place isolated nodes in the outermost circle if provided
    if isolated_nodes:
        radius_isolated = 6.0
        for i, node in enumerate(isolated_nodes):
            angle = 2 * np.pi * i / len(isolated_nodes)
            pos[node] = (
                radius_isolated * np.cos(angle),
                radius_isolated * np.sin(angle)
            )
    
    return pos

def create_sectored_layout(G, target_nodes, intermediate_nodes, input_nodes, isolated_nodes=None):
    """Create a layout with input nodes positioned near their connected intermediate nodes"""
    pos = {}
    
    # Set a fixed random seed for reproducibility
    np.random.seed(42)
    
    # Place target node at center
    if target_nodes:
        pos[target_nodes[0]] = (0, 0)
    
    # Increase spacing between intermediate and input nodes
    radius_intermediate = 4  
    radius_input = 6.5  # Slightly increased radius for more space
    radius_isolated = 9.0  # Also increased
    
    # Place intermediate nodes in a circle and store their angles
    intermediate_angles = {}
    for i, node in enumerate(intermediate_nodes):
        angle = 2 * np.pi * i / len(intermediate_nodes) if len(intermediate_nodes) > 0 else 0
        # Add small random perturbation to each node position to reduce perfect alignment
        random_offset = np.random.uniform(-0.2, 0.2, 2)  # Reduced random offset
        pos[node] = (
            radius_intermediate * np.cos(angle) + random_offset[0],
            radius_intermediate * np.sin(angle) + random_offset[1]
        )
        # Store the angle of this intermediate node
        intermediate_angles[node] = angle
    
    # Group input nodes by the intermediate node they connect to
    input_groups = {}
    connected_input_nodes = set()
    
    for input_node in input_nodes:
        # Find which intermediate node this input connects to most strongly
        connected_to = []
        for edge in G.out_edges(input_node, data=True):
            source, target, data = edge
            if target in intermediate_nodes:
                connected_to.append(target)
                connected_input_nodes.add(input_node)
        if connected_to:
            # Group by the first intermediate node connection
            input_groups.setdefault(connected_to[0], []).append(input_node)
    
    # Add unconnected input nodes to None group
    unconnected_input_nodes = set(input_nodes) - connected_input_nodes
    if unconnected_input_nodes:
        input_groups.setdefault(None, []).extend(unconnected_input_nodes)
    
    # Calculate narrower sector width based on number of intermediate nodes
    # More intermediate nodes = narrower sectors to prevent overlap
    base_sector_width = min(np.pi/6, 2*np.pi / (len(intermediate_nodes) * 1.5) if intermediate_nodes else np.pi/6)
    
    # Position input nodes in sectors aligned with their connected intermediate nodes
    for intermediate_node, group in input_groups.items():
        if not group:
            continue
            
        # For connected nodes, use the angle of the intermediate node they connect to
        if intermediate_node is not None:
            base_angle = intermediate_angles[intermediate_node]
            
            # Adjust sector width based on group size
            # Larger groups need slightly wider sectors
            sector_width = base_sector_width * (1.0 + min(len(group)/10, 1.0))
            
            # Reduce node spacing within rows for more compact layout
            node_spacing_factor = 0.8  # Reduce spacing between nodes in the same row
            
            # If there are more than 4 nodes in a group, arrange them in multiple layers
            if len(group) > 4:
                inner_layer = group[:len(group)//2]
                outer_layer = group[len(group)//2:]
                
                # Position inner layer in a row formation centered at the intermediate node angle
                angle_per_node_inner = sector_width / max(len(inner_layer), 1) * node_spacing_factor
                for i, input_node in enumerate(inner_layer):
                    # Center the nodes around the intermediate node angle
                    angle = base_angle + (i - len(inner_layer)/2 + 0.5) * angle_per_node_inner
                    pos[input_node] = (
                        radius_input * np.cos(angle),
                        radius_input * np.sin(angle)
                    )
                
                # Position outer layer slightly farther out
                angle_per_node_outer = sector_width / max(len(outer_layer), 1) * node_spacing_factor
                for i, input_node in enumerate(outer_layer):
                    angle = base_angle + (i - len(outer_layer)/2 + 0.5) * angle_per_node_outer
                    pos[input_node] = (
                        (radius_input + 1.2) * np.cos(angle),  # Reduced offset from 1.5 to 1.2
                        (radius_input + 1.2) * np.sin(angle)
                    )
            else:
                # For smaller groups, position in a tighter row centered at the intermediate node angle
                angle_per_node = sector_width / max(len(group), 1) * node_spacing_factor
                for i, input_node in enumerate(group):
                    angle = base_angle + (i - len(group)/2 + 0.5) * angle_per_node
                    pos[input_node] = (
                        radius_input * np.cos(angle),
                        radius_input * np.sin(angle)
                    )
        else:
            # For unconnected nodes, position them in a separate area
            angle_per_node = 2 * np.pi / max(len(group), 1) * 0.8  # Also tighter spacing
            for i, input_node in enumerate(group):
                angle = i * angle_per_node
                pos[input_node] = (
                    radius_input * np.cos(angle),
                    radius_input * np.sin(angle)
                )
    
    # Place isolated nodes in the outermost circle with increased visibility
    if isolated_nodes:
        angle_per_node = 2 * np.pi / len(isolated_nodes) if len(isolated_nodes) > 0 else 0
        for i, node in enumerate(isolated_nodes):
            angle = i * angle_per_node
            # Add a small random offset to avoid perfect alignment
            random_offset = np.random.uniform(-0.2, 0.2, 2)
            pos[node] = (
                radius_isolated * np.cos(angle) + random_offset[0],
                radius_isolated * np.sin(angle) + random_offset[1]
            )
    
    # Reset random seed
    np.random.seed(None)
    
    return pos

def categorize_edges(G, pos, removed_nodes=None):
    """Categorize edges based on the node types they connect"""
    removed_nodes = removed_nodes or set()
    
    edge_categories = {
        'input_to_intermediate': [],
        'intermediate_to_intermediate': [],
        'intermediate_to_target': []
    }
    
    for (u, v) in G.edges():
        if u in pos and v in pos and u not in removed_nodes:  # Only consider edges from active nodes
            u_type = G.nodes[u].get('entity_type')
            v_type = G.nodes[v].get('entity_type')
            
            if u_type == 'INPUT_NODE' and v_type == 'INTERMEDIATE_NODE':
                edge_categories['input_to_intermediate'].append((u, v))
            elif u_type == 'INTERMEDIATE_NODE' and v_type == 'TARGET_NODE':
                edge_categories['intermediate_to_target'].append((u, v))
    
    return edge_categories

def draw_edges_by_category(G, pos, edge_categories):
    """Draw edges according to their categories and styles"""
    for category, style in EDGE_STYLES.items():
        edges = edge_categories.get(category, [])
        if edges:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=edges,
                edge_color=style['color'],
                style=style['style'],
                width=style['width'],
                alpha=style['alpha'],
                arrows=True,
                arrowsize=20,
                arrowstyle='-|>'
            )

def visualize_graph_structure(G, dataset):
    """
    Create a structured visualization of the knowledge graph with emphasis on intermediate node relationships.
    
    Args:
        G: NetworkX graph
        dataset: Name of the dataset for the title
        removed_nodes: Set of node names that have been removed (will be shown in grey)
    """
    setup_visualization()
    target_nodes, intermediate_nodes, input_nodes, _ = get_nodes_by_type(G)
    pos = create_sectored_layout(G, target_nodes, intermediate_nodes, input_nodes)
    edge_categories = categorize_edges(G, pos)
    draw_edges_by_category(G, pos, edge_categories)
    draw_nodes_by_type(G, pos, ['INPUT_NODE', 'INTERMEDIATE_NODE', 'TARGET_NODE'])
    draw_node_labels(G, pos, ['INPUT_NODE', 'INTERMEDIATE_NODE', 'TARGET_NODE'])
    add_legend()
    output_path = f"images/{dataset}.png"
    plt.axis('off')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_post_training_graph(graphml_path, removed_nodes, dataset):
    """
    Visualize the optimized knowledge graph structure after training.
    
    Args:
        graphml_path: Path to the graphml file
        removed_nodes: Set of nodes to mark as removed
        dataset: Name of the dataset
    """
    
    # Load graph
    G = nx.read_graphml(graphml_path)
    G = nx.DiGraph(G)  # Ensure it's a directed graph
    
    # Convert to KnowledgeGraph
    kg = nx_to_knowledgegraph(G)
    
    # Create visualization graph
    G_vis = nx.DiGraph()
    
    # Add nodes to visualization graph
    for node in kg.nodes:
        G_vis.add_node(node.name, 
                      entity_type=node.node_type.value,
                      description=node.description)
    
    # Add edges to visualization graph
    for node in kg.nodes:
        for edge in node.edges:
            # Only add edge if neither source nor target is in removed_nodes
            if edge.source not in removed_nodes and edge.target not in removed_nodes:
                G_vis.add_edge(edge.source, edge.target, relationship=edge.relationship)
    
    # Identify input nodes connected only to removed intermediate nodes
    isolated_input_nodes = []
    for node, data in G_vis.nodes(data=True):
        if data.get('entity_type') == 'INPUT_NODE':
            # Check if this input node has any valid connections after removing nodes
            has_valid_connections = False
            for neighbor in list(G_vis.successors(node)):
                if neighbor not in removed_nodes:
                    has_valid_connections = True
                    break
            
            # If no valid connections, mark as isolated
            if not has_valid_connections:
                data['entity_type'] = 'ISOLATED_NODE'
                isolated_input_nodes.append(node)
    
    # Create visualization
    setup_visualization()
    
    # Get nodes by type (after updating isolated status)
    target_nodes, intermediate_nodes, input_nodes, isolated_nodes = get_nodes_by_type(G_vis)
    
    # Add the newly identified isolated input nodes to the isolated_nodes list
    isolated_nodes.extend(isolated_input_nodes)
    
    # Try to use stored positions if available
    global _initial_positions
    if _initial_positions:
        # Use initial positions but add any new nodes with sectored layout
        pos = _initial_positions.copy()
        
        # For any nodes not in the initial positions, calculate new positions
        missing_nodes = set(G_vis.nodes()) - set(pos.keys())
        if missing_nodes:
            # Create a temporary subgraph with missing nodes
            temp_G = G_vis.subgraph(missing_nodes)
            temp_target, temp_intermediate, temp_input, temp_isolated = get_nodes_by_type(temp_G)
            temp_pos = create_sectored_layout(temp_G, temp_target, temp_intermediate, temp_input, temp_isolated)
            
            # Add these positions to the main positions dict
            for node, position in temp_pos.items():
                pos[node] = position
    else:
        # Fall back to sectored layout if no initial positions
        pos = create_sectored_layout(G_vis, target_nodes, intermediate_nodes, input_nodes, isolated_nodes)
    
    # Draw edges
    edge_categories = categorize_edges(G_vis, pos, removed_nodes)
    draw_edges_by_category(G_vis, pos, edge_categories)
    
    # Draw nodes
    draw_nodes_by_type(G_vis, pos, ['INPUT_NODE', 'INTERMEDIATE_NODE', 'TARGET_NODE', 'ISOLATED_NODE'], removed_nodes)
    
    # Draw node labels
    draw_node_labels(G_vis, pos, ['INPUT_NODE', 'INTERMEDIATE_NODE', 'TARGET_NODE', 'ISOLATED_NODE'], removed_nodes)
    
    # Add a complete legend - include nodes but not edges
    add_legend()
    
    # Finalize and save
    output_path = f"images/{dataset}_optimized.png"
    plt.axis('off')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return G_vis, pos
    

def patient_prediction(patient, model, intermediate_models, intermediate_to_features, 
                      explanation, edge_annotations, is_binary=True, dataset="adni", 
                      output_dir='./images'):
    """Visualize prediction path with annotations for a specific patient"""
    global _initial_positions  # Access the global positions dictionary
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Extract LIME weights if available
    lime_weights = {}
    if explanation and 'features' in explanation:
        for feature_info in explanation['features']:
            feature_name = feature_info.get('name')
            if feature_name:
                lime_weights[feature_name] = feature_info.get('weight', 0)
    
    # Add nodes for features and intermediate concepts
    feature_nodes = []
    for feature in patient.index:
        # Get LIME weight for this feature (default to 0 if not available)
        weight = lime_weights.get(feature, 0)
        G.add_node(feature, type='feature', value=patient[feature], 
                  entity_type='INPUT_NODE', lime_weight=weight)
        feature_nodes.append(feature)
    
    # Add intermediate nodes
    intermediate_nodes = []
    intermediate_to_feature_influences = {}  # Store which features influence each intermediate node
    
    for node, model in intermediate_models.items():
        features = intermediate_to_features.get(node, [])
        if features and all(f in patient.index for f in features):
            try:
                if is_binary:
                    value = float(model.predict_proba(patient[features].values.reshape(1, -1))[0, 1])
                else:
                    value = float(model.predict(patient[features].values.reshape(1, -1))[0])
                
                # Calculate node importance for coloring
                # For binary: how far from 0.5 (neutral)
                # For multiclass: just use the value directly
                importance = abs(value - 0.5) * 2 if is_binary else min(value, 1.0)  
                
                # Add lime_weight to intermediate nodes for coloring
                # Positive if value > 0.5 for binary, otherwise based on actual value
                node_weight = (value - 0.5) * 2 if is_binary else value - 0.5
                
                G.add_node(node, type='intermediate', value=value, 
                          entity_type='INTERMEDIATE_NODE', 
                          importance=importance,
                          lime_weight=node_weight)
                          
                intermediate_nodes.append(node)
                
                # Find top influencing features for this intermediate node
                feature_influences = []
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    for i, feature in enumerate(features):
                        if i < len(importances) and feature in patient.index:
                            feature_influences.append({
                                'feature': feature, 
                                'importance': importances[i],
                                'value': float(patient[feature]) if isinstance(patient[feature], (int, float)) else str(patient[feature])
                            })
                
                # Sort by importance and keep top 3
                feature_influences.sort(key=lambda x: abs(x['importance']), reverse=True)
                intermediate_to_feature_influences[node] = feature_influences[:3]
                
                # Add edges from features to this intermediate node
                for feature in features:
                    G.add_edge(feature, node)
            except Exception as e:
                print(f"Error generating node {node}: {e}")
    
    # Add final outcome node with predicted diagnosis
    target_name = "Mortality Risk" if dataset == "mimic" else "Diagnosis"
    
    # Determine final prediction - use simpler method to avoid feature mismatch error
    if dataset.lower() == 'adni':
        # For ADNI, directly use intermediate node values to guess diagnosis
        sum_values = sum(G.nodes[n].get('value', 0) for n in intermediate_nodes)
        avg_value = sum_values / len(intermediate_nodes) if intermediate_nodes else 0
        
        # Map to diagnosis based on average intermediate value
        if avg_value < 0.5:
            diagnosis = "Normal"
            pred_color = 'green'
        elif avg_value < 1.5:
            diagnosis = "MCI"  # Mild Cognitive Impairment
            pred_color = 'gold'
        else:
            diagnosis = "Alzheimer's"
            pred_color = 'red'
            
        # Set confidence based on how close to threshold
        if 0.4 < avg_value < 0.6 or 1.4 < avg_value < 1.6:
            confidence = 0.6  # Near threshold, less confident
        else:
            confidence = 0.85  # Far from threshold, more confident
        
        pred_label = f"{target_name} = {diagnosis}\n(Confidence: {confidence:.2f})"
        pred_class = 0 if diagnosis == "Normal" else 1 if diagnosis == "MCI" else 2
        
    elif dataset.lower() == 'mimic':
        # For MIMIC (binary)
        avg_value = sum(G.nodes[n].get('value', 0) for n in intermediate_nodes) / len(intermediate_nodes) if intermediate_nodes else 0
        outcome = "DECEASED" if avg_value > 0.5 else "SURVIVED"
        confidence = abs(avg_value - 0.5) * 2  # 0 to 1 scale
        confidence = min(0.95, max(0.6, confidence))  # Cap between 0.6 and 0.95
        
        pred_label = f"{target_name} = {outcome}\n(Confidence: {confidence:.2f})"
        pred_class = 1 if outcome == "DECEASED" else 0
        pred_color = 'red' if pred_class == 1 else 'green'
    else:
        # Generic fallback
        pred_label = target_name
        pred_class = 0
        pred_color = NODE_COLORS['TARGET_NODE']
    
    print(f"Using diagnosis label: {pred_label}")
    
    G.add_node(target_name, type='outcome', entity_type='TARGET_NODE', pred_class=pred_class)
    target_nodes = [target_name]
    
    # Add edges from intermediate nodes to outcome
    for node in intermediate_nodes:
        G.add_edge(node, target_name)
    
    # Find isolated intermediate nodes - those that don't connect to target
    isolated_intermediate_nodes = []
    for node in intermediate_nodes:
        if not any(G.has_edge(node, t) for t in target_nodes):
            G.nodes[node]['entity_type'] = 'ISOLATED_NODE'
            isolated_intermediate_nodes.append(node)
    
    # Find isolated nodes - features that don't connect to any intermediate or connect only to isolated ones
    isolated_nodes = []
    for node in G.nodes():
        if G.degree(node) == 0:
            # Completely disconnected nodes
            G.nodes[node]['entity_type'] = 'ISOLATED_NODE'
            isolated_nodes.append(node)
        elif G.nodes[node].get('entity_type') == 'INPUT_NODE':
            # Check if all connections lead to isolated intermediate nodes
            all_targets_isolated = True
            for target in G.successors(node):
                if target not in isolated_intermediate_nodes and G.nodes[target].get('entity_type') == 'INTERMEDIATE_NODE':
                    all_targets_isolated = False
                    break
            
            if all_targets_isolated and G.out_degree(node) > 0:
                # This input node only connects to isolated intermediate nodes
                G.nodes[node]['entity_type'] = 'ISOLATED_NODE'
                isolated_nodes.append(node)
    
    # Get layout positions - first check for existing positions from previous visualizations
    if _initial_positions:
        # Use the existing positions
        pos = {}
        for node in G.nodes():
            if node in _initial_positions:
                pos[node] = _initial_positions[node]
            else:
                # For any new nodes not in the previous visualization, create positions
                pos[node] = (0, 0)  # Temporary default position
        
        # If any new nodes don't have positions, create a layout for just those nodes
        missing_nodes = [n for n in G.nodes() if pos[n] == (0, 0)]
        if missing_nodes:
            # Filter by type
            missing_target = [n for n in missing_nodes if G.nodes[n]['entity_type'] == 'TARGET_NODE']
            missing_intermediate = [n for n in missing_nodes if G.nodes[n]['entity_type'] == 'INTERMEDIATE_NODE']
            missing_input = [n for n in missing_nodes if G.nodes[n]['entity_type'] == 'INPUT_NODE']
            missing_isolated = [n for n in missing_nodes if G.nodes[n]['entity_type'] == 'ISOLATED_NODE']
            
            # Create layout for these nodes
            temp_pos = create_sectored_layout(G.subgraph(missing_nodes), 
                                             missing_target, 
                                             missing_intermediate, 
                                             missing_input,
                                             missing_isolated)
            
            # Update positions
            for node, position in temp_pos.items():
                pos[node] = position
    else:
        # If no previous positions exist, create sectored layout with all node types
        pos = create_sectored_layout(G, target_nodes, intermediate_nodes, feature_nodes, isolated_nodes)
        # Store positions for future use
        _initial_positions = pos.copy()
    
    # Setup visualization - use the same figure size as in other visualizations
    plt.figure(figsize=(30, 30))  # Match the setup_visualization function
    
    # Draw input nodes with corrected ADNI-specific color logic
    input_nodes = [n for n in G.nodes() if G.nodes[n].get('entity_type') == 'INPUT_NODE']
    input_node_colors = []
    input_node_alphas = []

    # Create mapping of which input nodes connect to which intermediate nodes
    input_to_intermediate_map = {}
    for u, v in G.edges():
        if G.nodes[u].get('entity_type') == 'INPUT_NODE' and G.nodes[v].get('entity_type') == 'INTERMEDIATE_NODE':
            if u not in input_to_intermediate_map:
                input_to_intermediate_map[u] = []
            input_to_intermediate_map[u].append(v)

    print("\n===== DEBUGGING NODE COLOR ALIGNMENT (REVISED) =====")
    print(f"Dataset: {dataset}, Is Binary: {is_binary}, Prediction Class: {pred_class}")
    print(f"Found {len(input_nodes)} input nodes and {len(intermediate_nodes)} intermediate nodes")

    # Define special ADNI features that are expected to be low in MCI/Alzheimer's
    adni_normally_low_features = {
        'CDGLOBAL', 'CDRSB', 'CDRSB.1', 'NPIQ', 'NPIGSEV', 'NPIHSEV', 'NPIJD', 'NPIL', 
        'NPIQ.1', 'NPIA', 'NPIB', 'NPIC', 'NPID', 'NPIE', 'NPIF', 'NPIK', 'NPIM', 'FAQ'
    }

    # Determine intermediate node alignment with prediction
    intermediate_alignments = {}
    for node in intermediate_nodes:
        value = G.nodes[node].get('value', 0.5)
        
        # Logic for determining alignment with prediction
        if dataset.lower() == 'adni':
            if pred_class == 0:  # Normal cognition
                alignment = 1 if value < 0.5 else -1  # Low values support normal
                align_str = "SUPPORTS normal" if alignment > 0 else "OPPOSES normal"
            elif pred_class == 2:  # Alzheimer's
                alignment = 1 if value > 1.5 else -1  # High values support Alzheimer's 
                align_str = "SUPPORTS alzheimer's" if alignment > 0 else "OPPOSES alzheimer's"
            else:  # MCI (class 1)
                alignment = 1 if 0.4 < value < 1.6 else -1  # Mid-range supports MCI
                align_str = "SUPPORTS mci" if alignment > 0 else "OPPOSES mci"
        else:  # MIMIC or other binary
            if pred_class == 1:  # Positive class (e.g., mortality)
                alignment = 1 if value > 0.5 else -1  # High values support mortality
                align_str = "SUPPORTS mortality" if alignment > 0 else "OPPOSES mortality"
            else:  # Negative class (e.g., survival)
                alignment = 1 if value < 0.5 else -1  # Low values support survival
                align_str = "SUPPORTS survival" if alignment > 0 else "OPPOSES survival"
        
        intermediate_alignments[node] = alignment
        print(f"  {node} alignment: {align_str} ({alignment})")

    # Now go through each input node and determine color
    print("\n--- INPUT NODE COLOR DETERMINATION (REVISED) ---")
    for i, node in enumerate(input_nodes):
        # Get feature value
        feature_value = float(patient[node]) if isinstance(patient[node], (int, float)) else 0.5
        is_high_value = feature_value > 0.5
        value_str = "HIGH" if is_high_value else "LOW"
        
        print(f"\nInput Node: {node}, Value: {feature_value:.3f} ({value_str})")
        
        # Get LIME weight for reference
        lime_weight = G.nodes[node].get('lime_weight', 0)
        lime_str = "POSITIVE" if lime_weight > 0 else "NEGATIVE"
        print(f"  LIME weight: {lime_weight:.3f} ({lime_str})")
        
        # Check connected intermediate nodes
        connected_intermediates = input_to_intermediate_map.get(node, [])
        
        # ADNI-SPECIFIC HANDLING: Some features in ADNI are expected to be low for MCI/Alzheimer's
        is_special_adni_feature = dataset.lower() == 'adni' and node in adni_normally_low_features
        if is_special_adni_feature:
            print(f"  Special ADNI feature: low values are expected for MCI/Alzheimer's")
        
        if connected_intermediates:
            print(f"  Connected to {len(connected_intermediates)} intermediate nodes:")
            
            total_alignment_score = 0
            for im_node in connected_intermediates:
                im_value = G.nodes[im_node].get('value', 0.5)
                im_status = "HIGH" if im_value > 0.5 else "LOW"
                
                # Determine feature-to-intermediate relationship - REVISED FOR ADNI
                if is_special_adni_feature and not is_high_value:
                    # For special ADNI features, LOW values actually SUPPORT MCI/Alzheimer's
                    feature_to_im_relationship = 1
                    rel_str = "CORRECTLY LOW FOR" if pred_class >= 1 else "INCORRECTLY LOW FOR"
                else:
                    # Normal relationship logic
                    feature_to_im_relationship = 1 if (is_high_value and im_value > 0.5) or (not is_high_value and im_value < 0.5) else -1
                    rel_str = "INCREASES" if feature_to_im_relationship > 0 else "DECREASES"
                
                # Get alignment of this intermediate with prediction
                im_alignment = intermediate_alignments.get(im_node, 0)
                im_align_str = "SUPPORTS prediction" if im_alignment > 0 else "OPPOSES prediction"
                
                # Calculate contribution: relationship * alignment
                contribution = feature_to_im_relationship * im_alignment
                contrib_str = "SUPPORTS prediction" if contribution > 0 else "OPPOSES prediction"
                
                print(f"    → {im_node} (value={im_value:.3f}, {im_status}): This input {rel_str} intermediate which {im_align_str}")
                print(f"      Therefore input {contrib_str} (score={contribution})")
                
                # Add to total score
                total_alignment_score += contribution
            
            # REVERSED COLOR LOGIC: For ADNI, invert colors for Alzheimer's predictions
            # So that supporting Alzheimer's = red, opposing Alzheimer's = green
            if dataset.lower() == 'adni' and pred_class >= 1:  # For MCI or Alzheimer's
                # Invert the score to reverse the colors
                total_alignment_score = -total_alignment_score
                
            # Determine color based on total score (now with reversed logic for ADNI)
            if total_alignment_score >= 0:
                color = 'green'
                color_str = "GREEN (opposes disease progression)" if dataset.lower() == 'adni' and pred_class >= 1 else "GREEN (supports prediction)"
            else:
                color = 'red'
                color_str = "RED (supports disease progression)" if dataset.lower() == 'adni' and pred_class >= 1 else "RED (opposes prediction)"
                
            print(f"  Total alignment score: {total_alignment_score}")
            print(f"  CHOSEN COLOR: {color_str}")
        else:
            print(f"  Not connected to any intermediate nodes!")
            # Default to LIME weight if no connections
            # ALSO REVERSE ADNI LOGIC FOR LIME WEIGHTS
            if dataset.lower() == 'adni' and pred_class >= 1:
                if lime_weight < 0:  # Negative LIME = opposing Alzheimer's = good
                    color = 'green'
                    color_str = "GREEN (opposes disease progression)"
                else:
                    color = 'red'
                    color_str = "RED (supports disease progression)"
            else:
                if lime_weight > 0:
                    color = 'green'
                    color_str = "GREEN (from positive LIME)"
                else:
                    color = 'red'
                    color_str = "RED (from negative LIME)"
            print(f"  CHOSEN COLOR: {color_str}")
        
        # Store final color and alpha
        input_node_colors.append(color)
        alpha = min(0.3 + abs(lime_weight) * 0.7, 1.0)
        input_node_alphas.append(alpha)

    print("\n--- SUMMARY ---")
    print(f"Input node colors: {len([c for c in input_node_colors if c == 'green'])} green, {len([c for c in input_node_colors if c == 'red'])} red")

    # Draw nodes with calculated colors
    if input_nodes:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=input_nodes,
            node_color=input_node_colors,
            node_size=NODE_SIZES['INPUT_NODE'],
            alpha=input_node_alphas
        )
    
    # Draw intermediate nodes with value-based coloring that aligns with target diagnosis
    intermediate_nodes = [n for n in G.nodes() if G.nodes[n].get('entity_type') == 'INTERMEDIATE_NODE']
    intermediate_node_colors = []
    intermediate_node_alphas = []
    
    for node in intermediate_nodes:
        value = G.nodes[node].get('value', 0.5)
        importance = G.nodes[node].get('importance', 0.5)
        
        # Determine color based on both value and target diagnosis
        # For ADNI with multiple classes
        if dataset.lower() == 'adni':
            if pred_class == 0:  # Normal is green
                # Green if supporting normal (low value), red if opposing
                intermediate_node_colors.append('green' if value < 0.5 else 'red')
            elif pred_class == 1:  # MCI is gold
                # Gold/yellow if supporting MCI (mid value), red/green if opposing
                if 0.4 < value < 1.2:
                    intermediate_node_colors.append('gold')
                elif value >= 1.2:  # Too high - supports Alzheimer's instead
                    intermediate_node_colors.append('red')
                else:  # Too low - supports Normal instead
                    intermediate_node_colors.append('green')
            else:  # Alzheimer's is red
                # Red if supporting Alzheimer's (high value), green if opposing
                intermediate_node_colors.append('red' if value > 1.2 else 'green')
        # For MIMIC binary
        elif dataset.lower() == 'mimic':
            if pred_class == 1:  # Deceased is red
                # Red if supporting death (high value), green if opposing
                intermediate_node_colors.append('red' if value > 0.5 else 'green')
            else:  # Survived is green
                # Green if supporting survival (low value), red if opposing
                intermediate_node_colors.append('green' if value <= 0.5 else 'red')
        else:
            # Generic fallback - use direct value
            intermediate_node_colors.append('green' if value <= 0.5 else 'red')
        
        # Determine opacity based on importance
        alpha = min(0.3 + importance * 0.7, 1.0)  # Scale to 0.3-1.0 range
        intermediate_node_alphas.append(alpha)
    
    # Draw intermediate nodes with colors and alphas
    if intermediate_nodes:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=intermediate_nodes,
            node_color=intermediate_node_colors,
            node_size=NODE_SIZES['INTERMEDIATE_NODE'],
            alpha=intermediate_node_alphas
        )
    
    # Draw target node with predicted color
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=target_nodes,
        node_color=pred_color,
        node_size=NODE_SIZES['TARGET_NODE'],
        alpha=0.7
    )
    
    # Draw isolated nodes
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=[n for n in G.nodes() if G.nodes[n].get('entity_type') == 'ISOLATED_NODE'],
        node_color=NODE_COLORS['ISOLATED_NODE'],
        node_size=NODE_SIZES['ISOLATED_NODE'],
        alpha=0.7
    )
    
    # Categorize edges - similar to categorize_edges function
    edge_categories = {
        'input_to_intermediate': [],
        'intermediate_to_target': []
    }
    
    for u, v in G.edges():
        if G.nodes[u].get('entity_type') == 'INPUT_NODE' and G.nodes[v].get('entity_type') == 'INTERMEDIATE_NODE':
            edge_categories['input_to_intermediate'].append((u, v))
        elif G.nodes[u].get('entity_type') == 'INTERMEDIATE_NODE' and G.nodes[v].get('entity_type') == 'TARGET_NODE':
            edge_categories['intermediate_to_target'].append((u, v))
    
    # Draw edges by category - similar to draw_edges_by_category function
    for category, style in EDGE_STYLES.items():
        edges = edge_categories.get(category, [])
        if edges:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=edges,
                edge_color=style['color'],
                style=style['style'],
                width=style['width'],
                alpha=style['alpha'],
                arrows=True,
                arrowsize=20,
                arrowstyle='-|>'
            )
    
    # Create node labels - now WITHOUT numerical values for intermediate nodes
    labels = {}
    for node in G.nodes():
        if G.nodes[node].get('entity_type') in ['INPUT_NODE', 'ISOLATED_NODE']:
            labels[node] = f"{node}"  # Just show feature name without value
        elif G.nodes[node].get('entity_type') == 'INTERMEDIATE_NODE':
            # Remove the value display, just show the node name
            labels[node] = f"{node}"
        elif G.nodes[node].get('entity_type') == 'TARGET_NODE':
            # Use the enhanced prediction label we created earlier
            labels[node] = pred_label
        else:
            labels[node] = node
    
    # Draw node labels using the exact same approach as in draw_node_labels function
    font_sizes = {
        'INPUT_NODE': 12,      
        'INTERMEDIATE_NODE': 20,
        'TARGET_NODE': 28,
        'ISOLATED_NODE': 10    
    }
    
    # Create label positions with the exact same offsets as in draw_node_labels
    label_pos = {}
    for node, (x, y) in pos.items():
        entity_type = G.nodes[node].get('entity_type', '')
        
        if entity_type == 'INPUT_NODE':
            # Position input labels below nodes
            label_pos[node] = (x, y - 0.15)
        elif entity_type == 'INTERMEDIATE_NODE':
            # Add deterministic variance based on node name
            hash_obj = hashlib.md5(str(node).encode())
            hash_val = int(hash_obj.hexdigest(), 16) % 100 / 100.0 - 0.5
            label_pos[node] = (x + hash_val * 0.1, y + 0.2)
        elif entity_type == 'TARGET_NODE':
            # Position target labels above nodes
            label_pos[node] = (x, y + 0.25)
        elif entity_type == 'ISOLATED_NODE':
            # For isolated nodes, position labels slightly to the right
            label_pos[node] = (x + 0.1, y)
        else:
            label_pos[node] = (x, y)
    
    # Draw labels for main node types with background boxes that match node colors
    for node_type in ['INPUT_NODE', 'INTERMEDIATE_NODE', 'TARGET_NODE']:
        nodes = [n for n in G.nodes() if G.nodes[n].get('entity_type') == node_type]
        if nodes:
            # Process labels individually to match box color with node color
            for node in nodes:
                # Get label text
                label_text = labels.get(node, node)
                
                # Determine appropriate box color based on node type and context
                if node_type == 'TARGET_NODE':
                    # Use prediction color for target node
                    box_edge_color = pred_color
                elif node_type == 'INTERMEDIATE_NODE':
                    # Find the index of this node in the intermediate_nodes list
                    if node in intermediate_nodes:
                        idx = intermediate_nodes.index(node)
                        if idx < len(intermediate_node_colors):
                            box_edge_color = intermediate_node_colors[idx]
                        else:
                            box_edge_color = 'gray'
                    else:
                        box_edge_color = 'gray'
                elif node_type == 'INPUT_NODE':
                    # Find the index of this node in the input_nodes list
                    if node in input_nodes:
                        idx = input_nodes.index(node)
                        if idx < len(input_node_colors):
                            box_edge_color = input_node_colors[idx]
                        else:
                            box_edge_color = 'gray'
                    else:
                        box_edge_color = 'gray'
                else:
                    # Default
                    box_edge_color = 'gray'
                
                # Create custom bbox properties with the node's color
                bbox_props = dict(
                    boxstyle="round,pad=0.3", 
                    fc="white", 
                    ec=box_edge_color, 
                    alpha=0.8
                )
                
                # Get position for this node's label
                if node in label_pos:
                    x, y = label_pos[node]
                    
                    # Draw the text with custom bbox
                    plt.text(
                        x, y, 
                        label_text, 
                        fontsize=font_sizes.get(node_type, 12),
                        fontweight='bold',
                        color='black',
                        horizontalalignment='center', 
                        verticalalignment='center',
                        bbox=bbox_props, 
                        zorder=3  # Ensure labels are on top of other elements
                    )

    # Draw isolated node labels differently - no background box at all
    isolated_nodes = [n for n in G.nodes() if G.nodes[n].get('entity_type') == 'ISOLATED_NODE']
    if isolated_nodes:
        for node in isolated_nodes:
            label_text = labels.get(node, node)
            if node in label_pos:
                x, y = label_pos[node]
                # Draw text with no bbox at all
                plt.text(
                    x, y, 
                    label_text, 
                    fontsize=font_sizes.get('ISOLATED_NODE', 10),
                    fontweight='normal',  # Not bold
                    color='#666666',      # Light gray text color
                    horizontalalignment='center', 
                    verticalalignment='center',
                    bbox=None,            # No box at all
                    zorder=2
                )
    
    # Create a dict to map intermediate nodes to their values for edge annotation
    intermediate_values = {n: G.nodes[n].get('value', 0.5) for n in intermediate_nodes}
    
    # Add edge annotations with multi-line formatting
    if edge_annotations:
        edge_labels = {}
        count_valid_annotations = 0
        
        # Create a set to track which edges have annotations
        annotated_edges = set()
        
        # First, apply specific annotations from the LLM
        for anno in edge_annotations:
            feature = anno.get('feature')
            target = anno.get('target')
            relationship = anno.get('relationship', '')
            
            # Check if feature and target exist
            feature_exists = feature in G.nodes()
            target_exists = target in G.nodes()
            edge_exists = G.has_edge(feature, target) if (feature_exists and target_exists) else False
            
            # Add edge label if edge exists
            if edge_exists:
                # Format relationship into multiple lines if needed
                words = relationship.split()
                formatted_rel = ""
                line_length = 0
                line_word_count = 0
                
                for word in words:
                    if line_word_count >= 3 or line_length > 15:  # Even tighter formatting
                        formatted_rel += "\n"
                        line_length = 0
                        line_word_count = 0
                    formatted_rel += word + " "
                    line_length += len(word) + 1
                    line_word_count += 1
                
                # Determine source and target types
                source_type = G.nodes[feature].get('entity_type', '')
                target_type = G.nodes[target].get('entity_type', '')
                
                # Store formatted label and annotation info
                edge_labels[(feature, target)] = {
                    'text': formatted_rel.strip(),
                    'source_type': source_type,
                    'target_type': target_type
                }
                annotated_edges.add((feature, target))
                count_valid_annotations += 1
            else:
                print(f"Warning: Edge annotation for {feature} → {target} couldn't be added (edge doesn't exist)")
        
        # Validate that we have annotations for all edges
        missing_edges = set(G.edges()) - annotated_edges
        if missing_edges:
            print(f"Warning: {len(missing_edges)} edges don't have annotations from LLM")
        
        # Draw edge labels with colored boxes
        if edge_labels:
            # Draw all edge labels with appropriate colored boxes
            for (source, target), label_info in edge_labels.items():
                text = label_info['text']
                source_type = label_info['source_type']
                target_type = label_info['target_type']
                
                # Determine edge box color based on node types and prediction
                if target_type == 'TARGET_NODE':
                    # For intermediate->target edges, use prediction color
                    edge_color = pred_color
                elif source_type == 'INPUT_NODE':
                    # For input->intermediate edges, match input node color
                    if source in input_nodes:
                        idx = input_nodes.index(source)
                        if idx < len(input_node_colors):
                            edge_color = input_node_colors[idx]
                        else:
                            edge_color = 'gray'
                    else:
                        edge_color = 'gray'
                else:
                    # Default
                    edge_color = 'gray'
                
                # Draw with custom colors - we need to draw each edge label individually
                # to have different colors for each edge
                bbox_props = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=edge_color, alpha=0.9)
                
                # Calculate position (midpoint of the edge)
                x1, y1 = pos[source]
                x2, y2 = pos[target]
                x = (x1 + x2) / 2
                y = (y1 + y2) / 2
                
                # Draw the text with custom bbox
                plt.text(x, y, text, fontsize=10, color='#555555', 
                        horizontalalignment='center', verticalalignment='center',
                        bbox=bbox_props, zorder=2)
            
            print(f"Added {count_valid_annotations} color-coded edge annotations to visualization")
        else:
            print("Warning: No valid edge annotations to display!")
    
    # Add legend with explanation of node coloring
    legend_elements = []
    
    # Add node types to legend with improved labels
    if dataset.lower() == 'adni':
        if pred_class == 0:  # Normal
            legend_elements.extend([
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', alpha=0.7,
                        markersize=15, label='Input Features (supporting normal cognition)'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', alpha=0.7,
                        markersize=15, label='Input Features (indicating cognitive decline)'),
            ])
        elif pred_class == 1:  # MCI
            legend_elements.extend([
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', alpha=0.7,
                        markersize=15, label='Input Features (supporting MCI diagnosis)'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', alpha=0.7,
                        markersize=15, label='Input Features (opposing MCI diagnosis)'),
            ])
        else:  # Alzheimer's
            legend_elements.extend([
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', alpha=0.7,
                        markersize=15, label='Input Features (opposing Alzheimer\'s diagnosis)'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', alpha=0.7,
                        markersize=15, label='Input Features (supporting Alzheimer\'s diagnosis)'),
            ])
    elif dataset.lower() == 'mimic':
        if pred_class == 0:  # Survived
            legend_elements.extend([
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', alpha=0.7,
                        markersize=15, label='Input Features (supporting survival)'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', alpha=0.7,
                        markersize=15, label='Input Features (increasing mortality risk)'),
            ])
        else:  # Deceased
            legend_elements.extend([
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', alpha=0.7,
                        markersize=15, label='Input Features (contributing to mortality)'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', alpha=0.7,
                        markersize=15, label='Input Features (protective factors)'),
            ])
    else:
        # Generic dataset
        legend_elements.extend([
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', alpha=0.7,
                    markersize=15, label='Input Features (supporting prediction)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', alpha=0.7,
                    markersize=15, label='Input Features (opposing prediction)'),
        ])
    
    # Add target node colors to legend based on dataset
    if dataset == 'mimic':
        legend_elements.extend([
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', alpha=0.7, 
                    markersize=15, label='Target: Survived'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', alpha=0.7, 
                    markersize=15, label='Target: Deceased')
        ])
    elif dataset == 'adni':
        legend_elements.extend([
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', alpha=0.7, 
                    markersize=15, label='Prediction: Normal'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gold', alpha=0.7, 
                    markersize=15, label='Prediction: Mild Cognitive Impairment'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', alpha=0.7, 
                    markersize=15, label='Prediction: Alzheimer\'s')
        ])
    else:
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=NODE_COLORS['TARGET_NODE'], alpha=0.7,
                    markersize=15, label='Target Outcome')
        )
    
    # Add isolated nodes to legend
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w', markerfacecolor=NODE_COLORS['ISOLATED_NODE'], 
                markersize=15, label='Isolated Nodes')
    )
    
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    plt.axis('off')
    
    # Save the visualization - use images directory as requested
    output_path = f"images/patient_example_annotated_{dataset}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    plt.close()
    
    return G

def annotate_knowledge_graph(graph, annotations, output_dir='./images'):
    """Annotate the knowledge graph with textual insights"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create positions using a spring layout
    pos = nx.spring_layout(graph, seed=42)
    
    # Create figure with improved resolution
    plt.figure(figsize=(24, 20), dpi=150)  # Increased from (20, 15), dpi=100
    
    # Draw nodes with different colors based on node type - INCREASED SIZES
    node_colors = []
    node_sizes = []
    for node in graph.nodes():
        if graph.nodes[node]['entity_type'] == 'INPUT_FEATURE':
            node_colors.append('orange')
            node_sizes.append(800)  # Increased from 500
        elif graph.nodes[node]['entity_type'] == 'INTERMEDIATE_NODE':
            node_colors.append('skyblue')
            node_sizes.append(1500)  # Increased from 700
        else:
            node_colors.append('lightgreen')
            node_sizes.append(2000)  # Increased from 900
    
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.85)  # Increased alpha from 0.8
    
    # Draw edges with INCREASED WIDTH
    nx.draw_networkx_edges(graph, pos, alpha=0.5, arrows=True, width=1.5, arrowsize=20)  # Increased alpha, width and arrowsize
    
    # Add labels with INCREASED FONT SIZE
    nx.draw_networkx_labels(graph, pos, font_size=14, font_weight='bold')  # Increased from 10
    
    # Add annotations to the plot with LARGER TEXT
    for annotation in annotations:
        feature = annotation.get('feature')
        target = annotation.get('target')
        relationship = annotation.get('relationship')
        
        # Find the nodes in the graph
        feature_node = None
        target_node = None
        
        for node in graph.nodes():
            if graph.nodes[node].get('name', node) == feature:
                feature_node = node
            elif graph.nodes[node].get('name', node) == target:
                target_node = node
        
        # If both nodes exist, add the annotation
        if feature_node and target_node:
            # Calculate position for the annotation (midpoint of edge)
            x1, y1 = pos[feature_node]
            x2, y2 = pos[target_node]
            x_mid = (x1 + x2) / 2
            y_mid = (y1 + y2) / 2
            
            # Offset to avoid overlap
            offset = 0.02
            x_mid += offset
            y_mid += offset
            
            # Add the annotation text with LARGER FONT
            plt.annotate(relationship, xy=(x_mid, y_mid), xytext=(x_mid, y_mid),
                        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9),  # Increased padding and alpha
                        ha='center', va='center', fontsize=11)  # Increased from 8
    
    plt.title('Annotated Knowledge Graph', fontsize=20)  # Increased from 16
    plt.axis('off')
    
    # Create legend with LARGER MARKERS
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=20, label='Input Features'),  # Increased from 15
        Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', markersize=20, label='Intermediate Nodes'),  # Increased from 15
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=20, label='Outcome')  # Increased from 15
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=14)  # Added fontsize
    
    # Finalize and save with higher DPI
    plt.savefig(f"{output_dir}/annotated_knowledge_graph.png", dpi=400, bbox_inches='tight')  # Increased from 300
    plt.close()