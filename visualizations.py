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
    
    input_edges = set()
    for u, v_node in graph.edges():
        if graph.nodes[u].get('entity_type') == 'INPUT_NODE' and graph.nodes[v_node].get('entity_type') == 'INTERMEDIATE_NODE':
            input_edges.add((u, v_node))

    intermediate_to_target = []
    for u, v_node in graph.edges():
        if graph.nodes[u].get('entity_type') == 'INTERMEDIATE_NODE' and graph.nodes[v_node].get('entity_type') == 'TARGET_NODE':
            intermediate_to_target.append((u,v_node))

    style_input_orig = EDGE_STYLES.get('input_to_intermediate', {'color': 'gray', 'style': 'solid', 'width': 1.5, 'alpha': 0.7})
    style_inter_target = EDGE_STYLES.get('intermediate_to_target', {'color': 'red', 'style': 'solid', 'width': 2.0, 'alpha': 0.8})

    nx.draw_networkx_edges(graph, pos, edgelist=input_edges, 
                            edge_color=style_input_orig['color'], style=style_input_orig['style'], 
                            width=style_input_orig['width'], alpha=style_input_orig['alpha'], 
                            arrows=True, arrowsize=20, arrowstyle='-|>')

    nx.draw_networkx_edges(graph, pos, edgelist=intermediate_to_target, 
                            edge_color=style_inter_target['color'], style=style_inter_target['style'], 
                            width=style_inter_target['width'], alpha=style_inter_target['alpha'], 
                            arrows=True, arrowsize=25, arrowstyle='-|>')
    
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