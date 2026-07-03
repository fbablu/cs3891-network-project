import networkx as nx

def calculate_network_metrics(G):
    """
    Calculate comprehensive network metrics including density, clustering, edge stats.
    
    Args:
        G (nx.Graph): NetworkX graph object
        
    Returns:
        dict: Dictionary containing network metrics
    """
    # Calculate basic network metrics
    density = nx.density(G)
    clustering_coeff = nx.average_clustering(G)
    num_edges = G.number_of_edges()
    
    # Calculate edge weight statistics
    edge_weights = [d['weight'] for (u, v, d) in G.edges(data=True)]
    avg_weight = sum(edge_weights) / len(edge_weights) if edge_weights else 0
    min_weight = min(edge_weights) if edge_weights else 0
    max_weight = max(edge_weights) if edge_weights else 0
    
    # Format the results
    metrics = {
        'Network Density': round(density, 3),
        'Average Clustering Coefficient': round(clustering_coeff, 3),
        'Number of Edges': num_edges,
        'Edge Weight Statistics': {
            'Average': round(avg_weight, 3),
            'Minimum': round(min_weight, 3),
            'Maximum': round(max_weight, 3)
        }
    }
    
    return metrics

def print_network_metrics(metrics, network_name):
    """
    Print network metrics in a formatted way.
    
    Args:
        metrics (dict): Dictionary of network metrics
        network_name (str): Name of the network being analyzed
    """
    print(f"\n{'-'*20} {network_name} Network Metrics {'-'*20}")
    print(f"Network Density: {metrics['Network Density']}")
    print(f"Average Clustering Coefficient: {metrics['Average Clustering Coefficient']}")
    print(f"Number of Edges: {metrics['Number of Edges']}")
    print("\nEdge Weight Statistics:")
    print(f"  Average Weight: {metrics['Edge Weight Statistics']['Average']}")
    print(f"  Minimum Weight: {metrics['Edge Weight Statistics']['Minimum']}")
    print(f"  Maximum Weight: {metrics['Edge Weight Statistics']['Maximum']}")
    print("-" * (42 + len(network_name)))