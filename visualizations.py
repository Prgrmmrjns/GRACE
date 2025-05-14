import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import matplotlib.patches as patches

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
        'color': '#FFA07A', # Orange
        'style': 'solid',
        'width': 2.0,
        'alpha': 0.7
    },
    'intermediate_to_target': {
        'color': '#4169E1', # Royal Blue
        'style': 'solid',
        'width': 3.5,
        'alpha': 0.9
    }
}

def visualize_knowledge_graph(G, dataset, node_groups, store_graph=False):
    graph = nx.DiGraph()
    node_attributes_from_G = {
        n: {
            'entity_type': G.nodes[n].get('entity_type', 'UNKNOWN'), 
            **G.nodes[n] 
           } 
        for n in G if G.has_node(n)
    }

    target_nodes_in_G = [n for n, d in G.nodes(data=True) if d.get('entity_type') == 'TARGET_NODE']
    for tn in target_nodes_in_G:
        graph.add_node(tn, **node_attributes_from_G[tn])

    for inter_node, final_features in node_groups.items():
        graph.add_node(inter_node, **node_attributes_from_G[inter_node])
        for input_feat in final_features:
            if input_feat in node_attributes_from_G and \
                node_attributes_from_G[input_feat].get('entity_type') == 'INPUT_NODE':
                if not graph.has_node(input_feat):
                    graph.add_node(input_feat, **node_attributes_from_G[input_feat])
                graph.add_edge(input_feat, inter_node)
        
        for tn in target_nodes_in_G:
            graph.add_edge(inter_node, tn)
    
    if store_graph:
        output_graphml_path = f"kg/{dataset}.graphml" 
        nx.write_graphml(graph, output_graphml_path)
    
    input_nodes = [n for n, d in graph.nodes(data=True) if d.get('entity_type') == 'INPUT_NODE']
    intermediate_nodes = [n for n, d in graph.nodes(data=True) if d.get('entity_type') == 'INTERMEDIATE_NODE']
    target_nodes = [n for n, d in graph.nodes(data=True) if d.get('entity_type') == 'TARGET_NODE']
    
    total_input_count = len(input_nodes)
    node_size_scale = max(0.3, min(1.0, 100.0 / max(1, total_input_count))) if total_input_count > 0 else 1.0
    font_size_scale = max(0.4, min(1.0, 60.0 / max(1, total_input_count))) if total_input_count > 0 else 1.0
    
    scaled_node_sizes = {
        'INPUT_NODE': int(NODE_SIZES.get('INPUT_NODE', 300) * node_size_scale),
        'INTERMEDIATE_NODE': NODE_SIZES.get('INTERMEDIATE_NODE', 1500),
        'TARGET_NODE': NODE_SIZES.get('TARGET_NODE', 2500),
    }
    
    max_nodes_per_circle = 30  
    num_input_circles = max(1, int(np.ceil(total_input_count / max_nodes_per_circle))) if total_input_count > 0 else 1
    
    pos = {}
    pos[target_nodes[0]] = (0, 0) 
    for i, tn in enumerate(target_nodes[1:]):
        angle = 2 * np.pi * i / max(1, len(target_nodes)-1)
        pos[tn] = (0.5 * np.cos(angle), 0.5 * np.sin(angle))

    radius_intermediate = 2.0
    for i, node in enumerate(intermediate_nodes):
        angle = 2 * np.pi * i / max(1, len(intermediate_nodes))
        pos[node] = (radius_intermediate * np.cos(angle), radius_intermediate * np.sin(angle))

    base_radius = 4.0 
    radius_increment = 0.8  
    
    nodes_per_circle_dist = [min(max_nodes_per_circle, total_input_count - i * max_nodes_per_circle) 
                            for i in range(num_input_circles)]
    input_idx = 0
    for circle_idx in range(num_input_circles):
        radius = base_radius + circle_idx * radius_increment
        nodes_this_circle = nodes_per_circle_dist[circle_idx]
        if nodes_this_circle == 0: continue
        
        for j in range(nodes_this_circle):
            if input_idx < total_input_count: 
                node = input_nodes[input_idx]
                angle = 2 * np.pi * j / nodes_this_circle
                angle_offset = circle_idx * (np.pi / max(1, nodes_this_circle * 2)) 
                pos[node] = (radius * np.cos(angle + angle_offset), radius * np.sin(angle + angle_offset))
                input_idx += 1
    
    plt.figure(figsize=(28, 28))
    
    nx.draw_networkx_nodes(graph, pos, nodelist=input_nodes, node_color=NODE_COLORS.get('INPUT_NODE', 'skyblue'), node_size=scaled_node_sizes['INPUT_NODE'], alpha=0.9)
    nx.draw_networkx_nodes(graph, pos, nodelist=intermediate_nodes, node_color=NODE_COLORS.get('INTERMEDIATE_NODE', 'lightgreen'), node_size=scaled_node_sizes['INTERMEDIATE_NODE'], alpha=0.9)
    nx.draw_networkx_nodes(graph, pos, nodelist=target_nodes, node_color=NODE_COLORS.get('TARGET_NODE', 'salmon'), node_size=scaled_node_sizes['TARGET_NODE'], alpha=0.9)
    
    # Convert to list for edges
    input_edges = []
    for u, v_node in graph.edges():
        if graph.nodes[u].get('entity_type') == 'INPUT_NODE' and graph.nodes[v_node].get('entity_type') == 'INTERMEDIATE_NODE':
            input_edges.append((u, v_node))

    intermediate_to_target = []
    for u, v_node in graph.edges():
        if graph.nodes[u].get('entity_type') == 'INTERMEDIATE_NODE' and graph.nodes[v_node].get('entity_type') == 'TARGET_NODE':
            intermediate_to_target.append((u,v_node))

    style_input_orig = EDGE_STYLES.get('input_to_intermediate', {'color': 'gray', 'style': 'solid', 'width': 1.5, 'alpha': 0.7})
    style_inter_target = EDGE_STYLES.get('intermediate_to_target', {'color': 'red', 'style': 'solid', 'width': 2.0, 'alpha': 0.8})

    node_sizes = [scaled_node_sizes[graph.nodes[n]['entity_type']] for n in graph.nodes()]
    
    nx.draw_networkx_edges(graph, pos, edgelist=input_edges, 
                            edge_color=style_input_orig['color'], style=style_input_orig['style'], 
                            width=style_input_orig['width'], alpha=style_input_orig['alpha'], 
                            arrows=True, arrowsize=20, arrowstyle='-|>',
                            node_size=node_sizes)

    nx.draw_networkx_edges(graph, pos, edgelist=intermediate_to_target, 
                            edge_color=style_inter_target['color'], style=style_inter_target['style'], 
                            width=style_inter_target['width'], alpha=style_inter_target['alpha'], 
                            arrows=True, arrowsize=25, arrowstyle='-|>',
                            node_size=node_sizes)
    
    label_nodes = [n for n in graph.nodes() if pos.get(n) is not None] 
    font_size = int(12 * font_size_scale)
    nx.draw_networkx_labels(graph, pos, labels={n: n for n in label_nodes}, font_size=font_size, font_color='black')
    
    legend_labels = {}
    legend_labels['Input Nodes'] = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=NODE_COLORS.get('INPUT_NODE', 'skyblue'), markersize=10)
    legend_labels['Intermediate Nodes'] = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=NODE_COLORS.get('INTERMEDIATE_NODE', 'lightgreen'), markersize=15)
    legend_labels['Target'] = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=NODE_COLORS.get('TARGET_NODE', 'salmon'), markersize=20)
    legend_labels['Input -> Intermediate'] = plt.Line2D([0], [0], color=style_input_orig['color'], linestyle=style_input_orig['style'], linewidth=style_input_orig['width'])
    legend_labels['Intermediate -> Target'] = plt.Line2D([0], [0], color=style_inter_target['color'], linestyle=style_inter_target['style'], linewidth=style_inter_target['width'])
    plt.legend(legend_labels.values(), legend_labels.keys(), loc='lower right', fontsize=14)
    plt.axis('off')
    os.makedirs('images', exist_ok=True) 
    output_viz_path = f"images/{dataset}.png" 
    plt.savefig(output_viz_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_lime_knowledge_graph(G_raw, dataset, node_groups, node_lime_weights, ensemble_lime_weights, output_base_dir="images", predicted_class_label="Predicted Class"):
    graph = nx.DiGraph()
    node_attributes_from_G = {
        n: {
            'entity_type': G_raw.nodes[n].get('entity_type', 'UNKNOWN'),
            **G_raw.nodes[n]
        }
        for n in G_raw if G_raw.has_node(n)
    }

    target_nodes_in_G = [n for n, d in G_raw.nodes(data=True) if d.get('entity_type') == 'TARGET_NODE']
    for tn in target_nodes_in_G:
        if tn in node_attributes_from_G: # Ensure target node exists in G_raw attributes
            graph.add_node(tn, **node_attributes_from_G[tn])

    for inter_node, final_features in node_groups.items():
        if inter_node in node_attributes_from_G: # Ensure intermediate node exists
            graph.add_node(inter_node, **node_attributes_from_G[inter_node])
            for input_feat in final_features:
                if input_feat in node_attributes_from_G and \
                   node_attributes_from_G[input_feat].get('entity_type') == 'INPUT_NODE':
                    if not graph.has_node(input_feat):
                        graph.add_node(input_feat, **node_attributes_from_G[input_feat])
                    graph.add_edge(input_feat, inter_node)
            
            for tn in target_nodes_in_G: # Connect intermediate node to all found target nodes
                if tn in graph: # Ensure target node was added to the graph
                     graph.add_edge(inter_node, tn)
    
    input_nodes = [n for n, d in graph.nodes(data=True) if d.get('entity_type') == 'INPUT_NODE']
    intermediate_nodes = [n for n, d in graph.nodes(data=True) if d.get('entity_type') == 'INTERMEDIATE_NODE']
    target_nodes = [n for n, d in graph.nodes(data=True) if d.get('entity_type') == 'TARGET_NODE']
    
    total_input_count = len(input_nodes)
    node_size_scale = max(0.3, min(1.0, 100.0 / max(1, total_input_count))) if total_input_count > 0 else 1.0
    font_size_scale = max(0.5, min(1.0, 60.0 / max(1, total_input_count))) if total_input_count > 0 else 1.0
    
    scaled_node_sizes = {
        'INPUT_NODE': int(NODE_SIZES.get('INPUT_NODE', 300) * node_size_scale),
        'INTERMEDIATE_NODE': NODE_SIZES.get('INTERMEDIATE_NODE', 1500),
        'TARGET_NODE': NODE_SIZES.get('TARGET_NODE', 2500),
    }
    
    max_nodes_per_circle = 30
    num_input_circles = max(1, int(np.ceil(total_input_count / max_nodes_per_circle))) if total_input_count > 0 else 1
    
    pos = {}
    if target_nodes: # Ensure target_nodes is not empty
        pos[target_nodes[0]] = (0, 0)
        for i, tn in enumerate(target_nodes[1:]):
            angle = 2 * np.pi * i / max(1, len(target_nodes)-1)
            pos[tn] = (0.5 * np.cos(angle), 0.5 * np.sin(angle))

    radius_intermediate = 2.0
    for i, node in enumerate(intermediate_nodes):
        angle = 2 * np.pi * i / max(1, len(intermediate_nodes))
        pos[node] = (radius_intermediate * np.cos(angle), radius_intermediate * np.sin(angle))

    base_radius = 4.0 
    radius_increment = 0.8  
    
    nodes_per_circle_dist = [min(max_nodes_per_circle, total_input_count - i * max_nodes_per_circle) 
                            for i in range(num_input_circles)]
    input_idx = 0
    for circle_idx in range(num_input_circles):
        radius = base_radius + circle_idx * radius_increment
        nodes_this_circle = nodes_per_circle_dist[circle_idx]
        if nodes_this_circle == 0: continue
        
        for j in range(nodes_this_circle):
            if input_idx < total_input_count: 
                node = input_nodes[input_idx]
                angle = 2 * np.pi * j / nodes_this_circle
                angle_offset = circle_idx * (np.pi / max(1, nodes_this_circle * 2))
                pos[node] = (radius * np.cos(angle + angle_offset), radius * np.sin(angle + angle_offset))
                input_idx += 1
    
    plt.figure(figsize=(28, 28))
    
    # Prepare edge lists based on LIME weights
    input_to_intermediate_pos_edges = []
    input_to_intermediate_neg_edges = []
    intermediate_to_target_pos_edges = []
    intermediate_to_target_neg_edges = []

    # Track which nodes should be included based on weight thresholds
    input_nodes_to_include = set()
    intermediate_nodes_to_include = set()
    
    # First, ensure all intermediate nodes with significant weights are included
    for node_name, weight in ensemble_lime_weights.items():
        if abs(weight) > 0.01 and node_name in graph.nodes():
            intermediate_nodes_to_include.add(node_name)
    
    # Process all intermediate nodes with significant weights
    for node_name in intermediate_nodes_to_include:
        weight = ensemble_lime_weights.get(node_name, 0)
        
        # Connect this intermediate node to input features
        if node_name in node_lime_weights:
            # Get all weights for this node
            node_features = list(node_lime_weights[node_name].items())
            
            # If no features have weight > 0.01, include at least the top 2 by absolute value
            significant_features = [f for f, w in node_features if abs(w) > 0.01]
            
            if not significant_features and node_features:
                # Sort by absolute weight value (highest first)
                sorted_features = sorted(node_features, key=lambda x: abs(x[1]), reverse=True)
                # Take top 2 features or all if less than 2
                top_features = sorted_features[:min(2, len(sorted_features))]
                
                # Add these top features
                for feat_name, feat_weight in top_features:
                    if graph.has_edge(feat_name, node_name):
                        input_nodes_to_include.add(feat_name)
                        # Determine if this feature supports the prediction
                        ensemble_contribution = ensemble_lime_weights.get(node_name, 0)
                        # If both input and intermediate have the same sign, it SUPPORTS the prediction
                        # (positive*positive or negative*negative)
                        # Otherwise, it OPPOSES the prediction
                        if (feat_weight > 0 and ensemble_contribution > 0) or (feat_weight < 0 and ensemble_contribution < 0):
                            input_to_intermediate_pos_edges.append((feat_name, node_name))
                        else:
                            input_to_intermediate_neg_edges.append((feat_name, node_name))
            
            # Include all features above threshold
            for feat_name, feat_weight in node_features:
                if abs(feat_weight) > 0.01 and graph.has_edge(feat_name, node_name):
                    input_nodes_to_include.add(feat_name)
                    # Determine if this feature supports the prediction
                    ensemble_contribution = ensemble_lime_weights.get(node_name, 0)
                    # Consider the sign combinations to determine support/opposition
                    if (feat_weight > 0 and ensemble_contribution > 0) or (feat_weight < 0 and ensemble_contribution < 0):
                        input_to_intermediate_pos_edges.append((feat_name, node_name))
                    else:
                        input_to_intermediate_neg_edges.append((feat_name, node_name))
        
        # Make sure this node is connected to the target node
        for tn in target_nodes:
            if graph.has_edge(node_name, tn):
                # For intermediate to target, the interpretation is more direct:
                # Positive LIME weight means it SUPPORTS the predicted class
                # Negative LIME weight means it OPPOSES the predicted class
                if weight > 0:
                    intermediate_to_target_pos_edges.append((node_name, tn))
                else:  # weight < 0
                    intermediate_to_target_neg_edges.append((node_name, tn))

    # Final check: ensure every intermediate node has at least one input node connection
    for node in intermediate_nodes_to_include.copy():
        # Find all connections to this node
        connected_inputs = [u for u, v in input_to_intermediate_pos_edges + input_to_intermediate_neg_edges if v == node]
        
        if not connected_inputs:
            # This intermediate node has no connected inputs, find the best features to connect
            all_edges = [(u, v) for u, v in graph.edges() 
                        if graph.nodes[u].get('entity_type') == 'INPUT_NODE' 
                        and v == node]
            
            if all_edges:
                # Take the first two available edges to this node
                for u, v in all_edges[:min(2, len(all_edges))]:
                    input_nodes_to_include.add(u)
                    # Determine connection type based on the ensemble weight
                    ensemble_contribution = ensemble_lime_weights.get(node, 0)
                    # For fallback connections, use positive if the intermediate node
                    # has a positive contribution to the target prediction
                    if ensemble_contribution > 0:
                        input_to_intermediate_pos_edges.append((u, v))
                    else:
                        input_to_intermediate_neg_edges.append((u, v))
                
    # Filter node lists to only include nodes with significant LIME contributions
    significant_input_nodes = [n for n in input_nodes if n in input_nodes_to_include]
    significant_intermediate_nodes = [n for n in intermediate_nodes if n in intermediate_nodes_to_include]
    
    # Calculate maximum weights before using them
    max_input_weight = 0.01  # Default minimum
    for _, weights_dict in node_lime_weights.items():
        for _, weight in weights_dict.items():
            if abs(weight) > max_input_weight:
                max_input_weight = abs(weight)
    
    max_ensemble_weight = 0.01  # Default minimum
    for _, weight in ensemble_lime_weights.items():
        if abs(weight) > max_ensemble_weight:
            max_ensemble_weight = abs(weight)
    
    # Draw only significant nodes with enhanced visibility
    # Use more saturated colors and add node borders for better visibility
    nx.draw_networkx_nodes(graph, pos, nodelist=significant_input_nodes, 
                          node_color=NODE_COLORS.get('INPUT_NODE', 'skyblue'), 
                          node_size=scaled_node_sizes['INPUT_NODE'], 
                          alpha=1.0,  # Full opacity
                          edgecolors='black', linewidths=1.0)  # Add borders
    
    # For intermediate nodes, adjust size based on importance
    intermediate_node_sizes = []
    intermediate_node_colors = []
    for node in significant_intermediate_nodes:
        weight = abs(ensemble_lime_weights.get(node, 0.01))
        weight_ratio = weight / max_ensemble_weight
        
        # Increase size for more important nodes
        size_factor = 1.0 + weight_ratio * 0.5  # Up to 50% larger
        node_size = scaled_node_sizes['INTERMEDIATE_NODE'] * size_factor
        intermediate_node_sizes.append(node_size)
        
        # Use the default color
        intermediate_node_colors.append(NODE_COLORS.get('INTERMEDIATE_NODE', 'lightgreen'))
    
    nx.draw_networkx_nodes(graph, pos, nodelist=significant_intermediate_nodes, 
                          node_color=intermediate_node_colors, 
                          node_size=intermediate_node_sizes, 
                          alpha=1.0,
                          edgecolors='black', linewidths=1.5)  # Slightly thicker border for intermediate nodes
    
    nx.draw_networkx_nodes(graph, pos, nodelist=target_nodes, 
                          node_color=NODE_COLORS.get('TARGET_NODE', 'salmon'), 
                          node_size=scaled_node_sizes['TARGET_NODE'], 
                          alpha=1.0,
                          edgecolors='black', linewidths=2.0)  # Thicker border for target
    
    # Only show labels for significant nodes
    significant_nodes = significant_input_nodes + significant_intermediate_nodes + target_nodes

    # Define colors for edges showing support vs opposition
    support_color = '#008000'  # Darker green for support
    oppose_color = '#CC0000'   # Darker red for opposition

    # Prepare node size dictionary for edge drawing
    node_size_dict = {}
    for n in graph.nodes():
        if n in significant_input_nodes:
            node_size_dict[n] = scaled_node_sizes['INPUT_NODE']
        elif n in significant_intermediate_nodes:
            idx = significant_intermediate_nodes.index(n)
            node_size_dict[n] = intermediate_node_sizes[idx]
        elif n in target_nodes:
            node_size_dict[n] = scaled_node_sizes['TARGET_NODE']
        else:
            node_size_dict[n] = 0  # Nodes not being drawn
    
    # Define edge width scaling with much greater contrast
    min_width = 1.0
    max_width = 8.0
    
    # Calculate edge widths using exponential scaling for better visual distinction
    input_pos_widths = []
    for u, v_node in input_to_intermediate_pos_edges:
        weight = abs(node_lime_weights.get(v_node, {}).get(u, 0.01))
        # Non-linear scaling to emphasize differences
        weight_ratio = weight / max_input_weight
        # Use power scaling for more dramatic effect
        scaled_width = min_width + (max_width - min_width) * (weight_ratio ** 0.7)
        input_pos_widths.append(scaled_width)
    
    input_neg_widths = []
    for u, v_node in input_to_intermediate_neg_edges:
        weight = abs(node_lime_weights.get(v_node, {}).get(u, 0.01))
        weight_ratio = weight / max_input_weight
        scaled_width = min_width + (max_width - min_width) * (weight_ratio ** 0.7)
        input_neg_widths.append(scaled_width)
    
    inter_pos_widths = []
    for u, v_node in intermediate_to_target_pos_edges:
        weight = abs(ensemble_lime_weights.get(u, 0.01))
        weight_ratio = weight / max_ensemble_weight
        scaled_width = min_width + (max_width - min_width) * (weight_ratio ** 0.7)
        inter_pos_widths.append(scaled_width)
    
    inter_neg_widths = []
    for u, v_node in intermediate_to_target_neg_edges:
        weight = abs(ensemble_lime_weights.get(u, 0.01))
        weight_ratio = weight / max_ensemble_weight
        scaled_width = min_width + (max_width - min_width) * (weight_ratio ** 0.7)
        inter_neg_widths.append(scaled_width)
    
    # Draw edges with variable widths
    # Draw positive input to intermediate edges
    if input_to_intermediate_pos_edges:
        nx.draw_networkx_edges(graph, pos, edgelist=input_to_intermediate_pos_edges, 
                              edge_color=support_color, style='solid', 
                              width=input_pos_widths, alpha=0.8, 
                              arrows=True, arrowsize=15, arrowstyle='-|>',
                              node_size=[node_size_dict[n] for n in graph.nodes()])
    
    # Draw negative input to intermediate edges
    if input_to_intermediate_neg_edges:
        nx.draw_networkx_edges(graph, pos, edgelist=input_to_intermediate_neg_edges, 
                              edge_color=oppose_color, style='solid', 
                              width=input_neg_widths, alpha=0.8, 
                              arrows=True, arrowsize=15, arrowstyle='-|>',
                              node_size=[node_size_dict[n] for n in graph.nodes()])
    
    # Draw positive intermediate to target edges
    if intermediate_to_target_pos_edges:
        nx.draw_networkx_edges(graph, pos, edgelist=intermediate_to_target_pos_edges, 
                              edge_color=support_color, style='solid', 
                              width=inter_pos_widths, alpha=0.9, 
                              arrows=True, arrowsize=20, arrowstyle='-|>',
                              node_size=[node_size_dict[n] for n in graph.nodes()])
    
    # Draw negative intermediate to target edges
    if intermediate_to_target_neg_edges:
        nx.draw_networkx_edges(graph, pos, edgelist=intermediate_to_target_neg_edges, 
                              edge_color=oppose_color, style='solid', 
                              width=inter_neg_widths, alpha=0.9, 
                              arrows=True, arrowsize=20, arrowstyle='-|>',
                              node_size=[node_size_dict[n] for n in graph.nodes()])
    
    # Only label significant nodes with larger, bold text
    label_nodes = [n for n in significant_nodes if pos.get(n) is not None]
    font_size = int(18 * font_size_scale)  # Increased from 16 for better readability
    nx.draw_networkx_labels(
        graph, 
        pos, 
        labels={n: n for n in label_nodes}, 
        font_size=font_size, 
        font_color='black',
        font_weight='bold'  # Make labels bold
    )
    
    # Add note about threshold to legend
    legend_note = f"* Only showing contributions > 0.01"
    
    # Legend
    legend_labels = {}
    legend_labels['Input Features'] = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=NODE_COLORS.get('INPUT_NODE', 'skyblue'), markersize=10)
    legend_labels['Intermediate Mechanisms'] = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=NODE_COLORS.get('INTERMEDIATE_NODE', 'lightgreen'), markersize=15)
    legend_labels[f'Target: {predicted_class_label}'] = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=NODE_COLORS.get('TARGET_NODE', 'salmon'), markersize=20)
    
    # Make legend more specific to the prediction - these represent actual support/opposition
    supports_text = f"Supports prediction of '{predicted_class_label}'"
    opposes_text = f"Opposes prediction of '{predicted_class_label}'"
    
    # Create a range of line thicknesses for the legend to show width variation
    supports_thick = plt.Line2D([0], [0], color=support_color, linestyle='solid', linewidth=max_width)
    supports_medium = plt.Line2D([0], [0], color=support_color, linestyle='solid', linewidth=max_width*0.5)
    supports_thin = plt.Line2D([0], [0], color=support_color, linestyle='solid', linewidth=min_width*2)
    
    opposes_thick = plt.Line2D([0], [0], color=oppose_color, linestyle='solid', linewidth=max_width)
    opposes_medium = plt.Line2D([0], [0], color=oppose_color, linestyle='solid', linewidth=max_width*0.5)
    opposes_thin = plt.Line2D([0], [0], color=oppose_color, linestyle='solid', linewidth=min_width*2)
    
    # Add them to the legend with explanations of importance
    legend_labels[supports_text + " (Major)"] = supports_thick
    legend_labels[supports_text + " (Medium)"] = supports_medium
    legend_labels[supports_text + " (Minor)"] = supports_thin
    
    legend_labels[opposes_text + " (Major)"] = opposes_thick
    legend_labels[opposes_text + " (Medium)"] = opposes_medium
    legend_labels[opposes_text + " (Minor)"] = opposes_thin
    
    # Add note about edge width and threshold
    legend_labels[legend_note] = plt.Line2D([0], [0], color='none')  # Invisible line for text-only legend item
    
    # Add the legend to the bottom right
    plt.legend(legend_labels.values(), legend_labels.keys(), loc='lower right', fontsize=16)
    
    plt.axis('off')
    os.makedirs(output_base_dir, exist_ok=True)
    # Use the exact dataset name that was passed in
    output_viz_path = f"{output_base_dir}/{dataset}_lime.png"
    plt.savefig(output_viz_path, dpi=300, bbox_inches='tight')
    plt.close()
