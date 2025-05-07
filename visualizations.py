import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os

NODE_COLORS = {
    'INPUT_NODE': '#ff7f0e',    # Orange for input nodes
    'INTERMEDIATE_NODE': '#1f77b4',  # Blue for intermediate nodes
    'TARGET_NODE': '#2ca02c',   # Green for target node
}

NODE_SIZES = {
    'INPUT_NODE': 3000,
    'INTERMEDIATE_NODE': 12000,
    'TARGET_NODE': 15000,
}

EDGE_STYLES = {
    'input_to_intermediate': {
        'color': '#FFA07A', # Light Salmon for original
        'style': 'solid',
        'width': 2.0,
        'alpha': 0.7
    },
    'added_input_to_intermediate': { # Style for added edges
        'color': '#32CD32', # LimeGreen for added
        'style': 'dashed',
        'width': 2.0,
        'alpha': 0.8
    },
    'intermediate_to_target': {
        'color': '#4169E1', # Royal Blue
        'style': 'solid',
        'width': 3.5,
        'alpha': 0.9
    }
}

def visualize_knowledge_graph(G, dataset, initial_node_groups, final_node_groups):
    """
    Visualize the knowledge graph with a radial layout and styled nodes/edges.
    - Target node at the center
    - Intermediate nodes in a middle circle
    - Input nodes in multiple concentric circles at the periphery
    - Distinguishes original vs. added input-to-intermediate edges.
    Only INPUT_NODE, INTERMEDIATE_NODE, and TARGET_NODE are shown.

    Args:
        G: NetworkX graph to visualize (should contain all nodes)
        dataset: Name for saving the graph
        initial_node_groups (dict): {intermediate_node: [initial_input_features]}
        final_node_groups (dict): {intermediate_node: [final_input_features]}
    """
    vis_G = G.copy() # Work on a copy to avoid modifying the original graph object
    
    # Remove isolated nodes from the graph copy before any visualization logic
    isolates = list(nx.isolates(vis_G))
    vis_G.remove_nodes_from(isolates)
    # if isolates: # Optional: for debugging if nodes disappear unexpectedly
    #     print(f"Visualization: Removed {len(isolates)} isolated nodes: {isolates}")

    # Identify nodes by type from the (potentially) modified vis_G
    input_nodes = [n for n, d in vis_G.nodes(data=True) if d.get('entity_type') == 'INPUT_NODE']
    intermediate_nodes = [n for n, d in vis_G.nodes(data=True) if d.get('entity_type') == 'INTERMEDIATE_NODE']
    target_nodes = [n for n, d in vis_G.nodes(data=True) if d.get('entity_type') == 'TARGET_NODE']
    
    # Calculate how many input nodes we have to adjust sizing
    total_input_count = len(input_nodes)
    
    # Scale node size and font size inversely with number of input nodes
    node_size_scale = max(0.3, min(1.0, 100.0 / max(1, total_input_count)))
    font_size_scale = max(0.4, min(1.0, 60.0 / max(1, total_input_count)))
    
    # Adjust node sizes based on scaling factor
    scaled_node_sizes = {
        'INPUT_NODE': int(NODE_SIZES['INPUT_NODE'] * node_size_scale),
        'INTERMEDIATE_NODE': NODE_SIZES['INTERMEDIATE_NODE'],
        'TARGET_NODE': NODE_SIZES['TARGET_NODE'],
    }
    
    # Calculate number of circles needed for input nodes
    max_nodes_per_circle = 30  # Maximum number of nodes per circle to prevent overcrowding
    num_input_circles = max(1, int(np.ceil(total_input_count / max_nodes_per_circle)))
    
    pos = {}
    # Place target node at center
    if target_nodes:
        pos[target_nodes[0]] = (0, 0)
        
    # Place intermediate nodes in a circle
    radius_intermediate = 2.0
    for i, node in enumerate(intermediate_nodes):
        angle = 2 * np.pi * i / max(1, len(intermediate_nodes))
        pos[node] = (
            radius_intermediate * np.cos(angle),
            radius_intermediate * np.sin(angle)
        )
    
    # Place input nodes across multiple concentric circles at the periphery
    # Start with the innermost circle for input nodes
    base_radius = 4.0  # Starting radius for input nodes
    radius_increment = 0.8  # Increase in radius per circle
    
    # Distribute input nodes evenly across the circles
    nodes_per_circle = [min(max_nodes_per_circle, total_input_count - i * max_nodes_per_circle) 
                        for i in range(num_input_circles)]
    
    input_idx = 0
    for circle_idx in range(num_input_circles):
        radius = base_radius + circle_idx * radius_increment
        nodes_this_circle = nodes_per_circle[circle_idx]
        
        for j in range(nodes_this_circle):
            if input_idx < total_input_count:
                node = input_nodes[input_idx]
                angle = 2 * np.pi * j / nodes_this_circle
                # Add a small angular offset for each circle to prevent nodes from different circles aligning
                angle_offset = circle_idx * (np.pi / max(1, nodes_this_circle * 2))
                pos[node] = (
                    radius * np.cos(angle + angle_offset),
                    radius * np.sin(angle + angle_offset)
                )
                input_idx += 1
    
    plt.figure(figsize=(28, 28))
    
    # Draw all input nodes with the same style
    nx.draw_networkx_nodes(
        vis_G, pos,
        nodelist=input_nodes,
        node_color=NODE_COLORS['INPUT_NODE'],
        node_size=scaled_node_sizes['INPUT_NODE'],
        alpha=0.9
    )
    
    # Draw intermediate and target nodes
    for node_type, nodelist in zip(['INTERMEDIATE_NODE', 'TARGET_NODE'], [intermediate_nodes, target_nodes]):
        nx.draw_networkx_nodes(
            vis_G, pos,
            nodelist=nodelist,
            node_color=NODE_COLORS[node_type],
            node_size=scaled_node_sizes[node_type],
            alpha=0.9
        )
    
    # Refined logic for collecting input-to-intermediate edges
    _original_input_edges_set = set()
    _added_input_edges_set = set()

    initial_features_lookup = {}
    # Ensure initial_node_groups is not None before iterating
    if initial_node_groups: 
        for inter, feats in initial_node_groups.items():
            # Ensure the intermediate node from the group exists in vis_G
            if vis_G.has_node(inter) and vis_G.nodes[inter].get('entity_type') == 'INTERMEDIATE_NODE':
                initial_features_lookup[inter] = set(feats)

    # Case 1: final_node_groups is primary, if it's not None AND not empty
    if final_node_groups: 
        for inter_node, features_in_final_group in final_node_groups.items():
            if not (vis_G.has_node(inter_node) and vis_G.nodes[inter_node].get('entity_type') == 'INTERMEDIATE_NODE'):
                continue
            current_initial_set = initial_features_lookup.get(inter_node, set())
            for input_node in features_in_final_group:
                if vis_G.has_node(input_node) and \
                   vis_G.nodes[input_node].get('entity_type') == 'INPUT_NODE' and \
                   vis_G.has_edge(input_node, inter_node): # Check edge existence in vis_G
                    edge = (input_node, inter_node)
                    if input_node in current_initial_set: 
                        _original_input_edges_set.add(edge)
                    else: 
                        _added_input_edges_set.add(edge)
    
    elif initial_node_groups: 
        for inter_node, features_in_initial_group in initial_node_groups.items():
            if not (vis_G.has_node(inter_node) and vis_G.nodes[inter_node].get('entity_type') == 'INTERMEDIATE_NODE'):
                continue
            for input_node in features_in_initial_group:
                if vis_G.has_node(input_node) and \
                   vis_G.nodes[input_node].get('entity_type') == 'INPUT_NODE' and \
                   vis_G.has_edge(input_node, inter_node): # Check edge existence in vis_G
                    _original_input_edges_set.add((input_node, inter_node))

    if not _original_input_edges_set and not _added_input_edges_set:
        for u, v_node in vis_G.edges(): # Iterate edges from vis_G
            if vis_G.has_node(u) and vis_G.nodes[u].get('entity_type') == 'INPUT_NODE' and \
               vis_G.has_node(v_node) and vis_G.nodes[v_node].get('entity_type') == 'INTERMEDIATE_NODE':
                _original_input_edges_set.add((u, v_node))
    
    original_input_edges = list(_original_input_edges_set)
    added_input_edges = list(_added_input_edges_set - _original_input_edges_set)

    intermediate_to_target = [(u, v) for u, v in vis_G.edges() if # Iterate edges from vis_G
                             vis_G.has_node(u) and vis_G.has_node(v) and 
                             vis_G.nodes[u].get('entity_type') == 'INTERMEDIATE_NODE' and
                             vis_G.nodes[v].get('entity_type') == 'TARGET_NODE']

    # Draw Original Input -> Intermediate Edges
    if original_input_edges:
        nx.draw_networkx_edges(
            vis_G, pos, # Use vis_G
            edgelist=original_input_edges,
            edge_color=EDGE_STYLES['input_to_intermediate']['color'],
            style=EDGE_STYLES['input_to_intermediate']['style'],
            width=EDGE_STYLES['input_to_intermediate']['width'],
            alpha=EDGE_STYLES['input_to_intermediate']['alpha'],
            arrows=True,
            arrowsize=20,
            arrowstyle='-|>'
        )

    # Draw Added Input -> Intermediate Edges
    if added_input_edges:
        nx.draw_networkx_edges(
            vis_G, pos, # Use vis_G
            edgelist=added_input_edges,
            edge_color=EDGE_STYLES['added_input_to_intermediate']['color'],
            style=EDGE_STYLES['added_input_to_intermediate']['style'],
            width=EDGE_STYLES['added_input_to_intermediate']['width'],
            alpha=EDGE_STYLES['added_input_to_intermediate']['alpha'],
            arrows=True,
            arrowsize=20,
            arrowstyle='-|>'
        )

    # Draw Intermediate -> Target Edges
    if intermediate_to_target:
        nx.draw_networkx_edges(
            vis_G, pos, # Use vis_G
            edgelist=intermediate_to_target,
            edge_color=EDGE_STYLES['intermediate_to_target']['color'],
            style=EDGE_STYLES['intermediate_to_target']['style'],
            width=EDGE_STYLES['intermediate_to_target']['width'],
            alpha=EDGE_STYLES['intermediate_to_target']['alpha'],
            arrows=True,
            arrowsize=25,
            arrowstyle='-|>'
        )
    
    # Draw labels for all nodes present in vis_G
    label_nodes = input_nodes + intermediate_nodes + target_nodes
    # Scale font size based on number of input nodes
    font_size = int(12 * font_size_scale)
    nx.draw_networkx_labels(vis_G, pos, labels={n: n for n in label_nodes}, font_size=font_size, font_color='black') # Use vis_G
    
    # Add a legend explaining the node colors and edge styles
    legend_labels = {
        'Input Features': plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=NODE_COLORS['INPUT_NODE'], markersize=10),
        'Intermediate Nodes': plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=NODE_COLORS['INTERMEDIATE_NODE'], markersize=15),
        'Target': plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=NODE_COLORS['TARGET_NODE'], markersize=20),
        'Input -> Intermediate Edge': plt.Line2D([0], [0], color=EDGE_STYLES['input_to_intermediate']['color'], linestyle=EDGE_STYLES['input_to_intermediate']['style'], linewidth=EDGE_STYLES['input_to_intermediate']['width']),
        'Intermediate->Target Edge': plt.Line2D([0], [0], color=EDGE_STYLES['intermediate_to_target']['color'], linestyle=EDGE_STYLES['intermediate_to_target']['style'], linewidth=EDGE_STYLES['intermediate_to_target']['width']),
    }
    
    # Conditionally add "Added Input Edge (Optimized)" to the legend if there are any such edges
    if added_input_edges:
        legend_labels['Added Input Edge (Optimized)'] = plt.Line2D([0], [0], color=EDGE_STYLES['added_input_to_intermediate']['color'], linestyle=EDGE_STYLES['added_input_to_intermediate']['style'], linewidth=EDGE_STYLES['added_input_to_intermediate']['width'])
    
    plt.legend(legend_labels.values(), legend_labels.keys(), loc='lower right', fontsize=14)
    
    plt.axis('off')
    os.makedirs('graphs', exist_ok=True)
    output_path = f"graphs/{dataset}_graph.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path