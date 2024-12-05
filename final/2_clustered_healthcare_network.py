"""
Clustered Healthcare Network Analysis

This module implements a clustered network analysis approach to understand healthcare
provider and spending patterns across US states. It uses community detection and
advanced visualization techniques to identify and display healthcare metric clusters.

The analysis reveals how healthcare systems across states form natural groupings based 
on their provider density and spending patterns. By examining these clusters, we can 
understand regional patterns and identify states with similar healthcare approaches.

Key Features:
- Implements community detection for healthcare metrics
- Provides enhanced visualization of network clusters
- Calculates detailed cluster statistics
- Analyzes inter-cluster relationships

Dependencies:
    - pandas: For data manipulation and analysis
    - networkx: For network creation and analysis
    - matplotlib: For visualization
    - numpy: For numerical computations
    - scipy: For statistical analysis
    - python-louvain (community): For community detection algorithms

Design Notes:
- Uses Louvain community detection to identify natural groupings of states
- Implements custom balance metrics to quantify healthcare similarity
- Provides detailed cluster analysis for policy insights
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap
import community.community_louvain as community_louvain
from network_info.network_metrics import calculate_network_metrics, print_network_metrics

# Dictionary defining the four main US census regions and their constituent states
# This grouping enables analysis of regional patterns and visualization color-coding
US_REGIONS = {
    'Northeast': ['ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA'],
    'Midwest': ['OH', 'IN', 'IL', 'MI', 'WI', 'MN', 'IA', 'MO', 'ND', 'SD', 'NE', 'KS'],
    'South': ['DE', 'MD', 'DC', 'VA', 'WV', 'NC', 'SC', 'GA', 'FL', 'KY', 'TN', 'AL', 'MS', 'AR', 'LA', 'OK', 'TX'],
    'West': ['MT', 'ID', 'WY', 'CO', 'NM', 'AZ', 'UT', 'NV', 'WA', 'OR', 'CA', 'AK', 'HI']
}

# Color scheme for visualizing regions
# Chosen for optimal contrast and colorblind accessibility
# Uses distinct, easily distinguishable colors for each region
REGION_COLORS = {
    'Northeast': '#D55E00',  # Dark orange - distinctive and warm
    'Midwest': '#009E73',    # Dark green - natural and balanced
    'South': '#0072B2',      # Dark blue - cool and professional
    'West': '#CC79A7'        # Dark pink - unique and visible
}

def get_state_region(state_code):
    """
    Determine which census region a state belongs to.
    
    This function maps state codes to their corresponding census regions, enabling
    regional analysis and visualization. It's fundamental for understanding geographic
    patterns in healthcare metrics.
    
    Args:
        state_code (str): Two-letter state code (e.g., 'CA' for California)
        
    Returns:
        str: Census region name ('Northeast', 'Midwest', 'South', 'West') or 'Unknown'
             if the state code isn't found
    """
    for region, states in US_REGIONS.items():
        if state_code in states:
            return region
    return 'Unknown'

def load_and_process_data():
    """
    Load and preprocess healthcare provider and spending data.
    
    This function handles data ingestion and initial processing, creating a clean,
    merged dataset for analysis. It performs several key steps:
    1. Loads raw provider and spending data
    2. Aggregates metrics by state
    3. Merges the datasets for combined analysis
    
    Returns:
        pd.DataFrame: Clean dataset containing:
            - region_code: State identifier
            - providers_per_100k: Provider density metric
            - hospital_expenditure: Spending metric
    """
    # Load raw datasets from CSV files
    providers_df = pd.read_csv('FINAL_providers.csv')
    spending_df = pd.read_csv('FINAL_spending.csv')
    
    # Aggregate metrics by state, taking means to handle multiple years
    providers_agg = providers_df.groupby('region_code')['providers_per_100k'].mean().reset_index()
    spending_agg = spending_df.groupby('region_code')['(E058) Total Hospital-Dir Exp'].mean().reset_index()
    
    # Merge the aggregated datasets and rename columns for clarity
    merged_df = pd.merge(providers_agg, spending_agg, on='region_code')
    merged_df.columns = ['region_code', 'providers_per_100k', 'hospital_expenditure']
    
    return merged_df

def calculate_balance_metric(df):
    """
    Calculate healthcare balance similarity between all pairs of states.
    
    This function quantifies how similar states are in terms of their healthcare
    metrics. It uses a sophisticated approach that:
    1. Normalizes both provider and spending metrics
    2. Calculates differences between states
    3. Converts differences into similarity scores
    
    The balance metric considers both how close states are in their provider density
    and spending levels, creating a comprehensive similarity measure.
    
    Args:
        df (pd.DataFrame): Processed healthcare data with provider and spending metrics
        
    Returns:
        tuple: (balance_matrix, states)
            - balance_matrix (np.array): Square matrix of similarity scores
            - states (np.array): State codes in same order as matrix dimensions
    """
    states = df['region_code'].values
    n_states = len(states)
    balance_matrix = np.zeros((n_states, n_states))
    
    # Normalize both metrics to make them comparable
    providers_norm = (df['providers_per_100k'] - df['providers_per_100k'].mean()) / df['providers_per_100k'].std()
    spending_norm = (df['hospital_expenditure'] - df['hospital_expenditure'].mean()) / df['hospital_expenditure'].std()
    
    # Calculate similarity scores between all pairs of states
    for i in range(n_states):
        for j in range(n_states):
            if i != j:  # Skip self-comparisons
                provider_diff = abs(providers_norm.iloc[i] - providers_norm.iloc[j])
                spending_diff = abs(spending_norm.iloc[i] - spending_norm.iloc[j])
                # Convert differences to similarity score (closer to 1 means more similar)
                balance = (1 / (1 + abs(provider_diff - spending_diff))) ** 2
                balance_matrix[i, j] = balance
    
    return balance_matrix, states

def create_clustered_network(balance_matrix, states, weak_threshold=0.5, strong_threshold=0.8):
    """
    Create a network representation of healthcare relationships between states.
    
    This function builds a sophisticated network model where:
    - States are nodes
    - Healthcare similarities are edges
    - Edge weights represent similarity strength
    - Multiple centrality measures are calculated
    
    The network uses two thresholds to distinguish between weak and strong relationships,
    enabling more nuanced analysis of healthcare patterns.
    
    Args:
        balance_matrix (np.array): Matrix of healthcare similarity scores
        states (np.array): Array of state codes
        weak_threshold (float): Minimum similarity for a weak connection (default: 0.5)
        strong_threshold (float): Threshold for strong connections (default: 0.8)
        
    Returns:
        nx.Graph: Network with states as nodes and healthcare similarities as edges,
                 including centrality measures and relationship strengths
    """
    G = nx.Graph()
    
    # Add states as nodes, including their regional information
    for state in states:
        G.add_node(state, region=get_state_region(state))
    
    # Add edges representing healthcare similarities
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            balance = balance_matrix[i, j]
            if balance > weak_threshold:
                G.add_edge(states[i], states[j],
                          weight=balance,
                          is_strong=balance > strong_threshold)
    
    # Calculate multiple centrality measures to understand state importance
    degree_cent = nx.degree_centrality(G)
    betweenness_cent = nx.betweenness_centrality(G)
    eigenvector_cent = nx.eigenvector_centrality(G, max_iter=1000)
    
    # Combine centrality measures for a comprehensive importance score
    for node in G.nodes():
        G.nodes[node]['centrality'] = (
            degree_cent[node] + 
            betweenness_cent[node] + 
            eigenvector_cent[node]
        ) / 3.0
    
    return G

def visualize_clustered_network(G):
    """
    Create a sophisticated visualization of the healthcare network.
    
    This function generates a comprehensive visual representation that shows:
    - States as nodes, sized by their network importance
    - Regional groupings through color-coding
    - Relationship strengths through edge thickness and color
    - Clear labels and legends for interpretation
    
    The visualization is designed to reveal patterns in:
    - Regional healthcare similarities
    - Strong vs weak relationships between states
    - Central vs peripheral states in the healthcare network
    
    Args:
        G (nx.Graph): The healthcare network graph
    """
    # Create figure with space for main network and colorbar
    fig = plt.figure(figsize=(20, 15))
    ax_network = fig.add_axes([0.1, 0.1, 0.7, 0.8])
    ax_colorbar = fig.add_axes([0.85, 0.1, 0.03, 0.8])
    
    # Create weighted adjacency matrix for better layout
    adj_matrix = nx.to_numpy_array(G)
    for i, j in np.ndindex(adj_matrix.shape):
        if adj_matrix[i, j] > 0:
            edge_data = G.get_edge_data(list(G.nodes())[i], list(G.nodes())[j])
            if edge_data['is_strong']:
                adj_matrix[i, j] *= 2  # Stronger edges pull nodes closer
    
    # Calculate optimal layout
    pos = nx.spring_layout(
        G,
        k=1.5,  # Node spacing
        iterations=100,
        seed=42,  # For reproducibility
        weight='weight'
    )
    
    # Size nodes based on their centrality
    node_sizes = [G.nodes[node]['centrality'] * 3000 + 500 for node in G.nodes()]
    
    # Draw nodes with regional colors
    node_colors = [REGION_COLORS[G.nodes[node]['region']] for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos,
                          node_color=node_colors,
                          node_size=node_sizes,
                          alpha=0.7,
                          ax=ax_network)
    
    # Create custom purple colormap for edges
    colors = [(1, 1, 1), (0.5, 0, 0.5)]  # White to purple
    custom_cmap = LinearSegmentedColormap.from_list("custom_purple", colors, N=100)
    
    # Draw edges with different styles for weak and strong relationships
    edges = G.edges(data=True)
    
    # Draw weak edges (thinner and more transparent)
    weak_edges = [(u, v) for u, v, d in edges if not d['is_strong']]
    weak_weights = [d['weight'] for u, v, d in edges if not d['is_strong']]
    if weak_edges:
        nx.draw_networkx_edges(G, pos,
                             edgelist=weak_edges,
                             edge_color=weak_weights,
                             edge_cmap=custom_cmap,
                             width=1,
                             alpha=0.3,
                             ax=ax_network)
    
    # Draw strong edges (thicker and more opaque)
    strong_edges = [(u, v) for u, v, d in edges if d['is_strong']]
    strong_weights = [d['weight'] for u, v, d in edges if d['is_strong']]
    if strong_edges:
        nx.draw_networkx_edges(G, pos,
                             edgelist=strong_edges,
                             edge_color=strong_weights,
                             edge_cmap=custom_cmap,
                             width=3,
                             alpha=0.8,
                             ax=ax_network)
    
    # Add state labels with slight offset for visibility
    label_pos = {node: (coord[0], coord[1] + 0.02) for node, coord in pos.items()}
    nx.draw_networkx_labels(G, label_pos, font_size=10, font_weight='bold', ax=ax_network)
    
    # Add colorbar to show edge weight scale
    all_weights = [d['weight'] for u, v, d in edges]
    norm = plt.Normalize(vmin=min(all_weights), vmax=max(all_weights))
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=ax_colorbar, label='Balance Score')
    
    # Create legend
    legend_elements = []
    for region, color in REGION_COLORS.items():
        legend_elements.append(plt.Line2D([], [], color=color, marker='o',
                                        linestyle='None', markersize=10,
                                        label=region, alpha=0.7))
    
    # Add node size explanation to legend
    legend_elements.extend([
        plt.Line2D([], [], color='gray', marker='o', linestyle='None',
                  markersize=8, label='Node size: Network centrality',
                  alpha=0.7)
    ])
    
    # Position legend and add title
    ax_network.legend(handles=legend_elements,
                     fontsize=12,
                     loc='upper left',
                     bbox_to_anchor=(1.1, 1))
    
    ax_network.set_title('Healthcare Network: Clustered Provider-Spending Balance\n' +
                        'Darker edges indicate stronger balance, node size shows centrality',
                        fontsize=16, pad=20)
    ax_network.axis('off')
    
    # Save high-resolution figure
    plt.savefig('healthcare_clustered_network.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_clusters(G):
    """
    Perform detailed analysis of network clusters.
    
    This function uses the Louvain community detection algorithm to identify natural
    groupings of states based on their healthcare similarities. It then analyzes
    these clusters to understand:
    - Which states group together
    - Regional representation in clusters
    - Strength of relationships within clusters
    
    The analysis helps identify meaningful patterns in how healthcare systems
    are organized across states.
    
    Args:
        G (nx.Graph): Healthcare network graph
        
    Returns:
        dict: Detailed statistics for each cluster:
            - states: List of states in the cluster
            - regions: Set of regions represented
            - avg_balance: Average similarity score within cluster
    """
    # Detect communities using Louvain algorithm
    communities = community_louvain.best_partition(G)
    
    # Initialize storage for cluster statistics
    cluster_stats = {}
    for state, cluster_id in communities.items():
        if cluster_id not in cluster_stats:
            cluster_stats[cluster_id] = {
                'states': [],
                'regions': set(),
                'avg_balance': []
            }
        cluster_stats[cluster_id]['states'].append(state)
        cluster_stats[cluster_id]['regions'].add(get_state_region(state))
    
    # Calculate average balance scores within each cluster
    for cluster_id in cluster_stats:
        cluster_states = cluster_stats[cluster_id]['states']
        balances = []
        # Examine all edges within the cluster to calculate average relationship strength
        for u, v, d in G.edges(data=True):
            if u in cluster_states and v in cluster_states:
                balances.append(d['weight'])
        # Calculate average balance, defaulting to 0 if no internal edges exist
        cluster_stats[cluster_id]['avg_balance'] = np.mean(balances) if balances else 0
    
    return cluster_stats

def main():
    """
    Execute the complete healthcare network clustering analysis.
    
    This function orchestrates the entire analysis pipeline, including:
    1. Data loading and preprocessing
    2. Network creation and analysis
    3. Cluster detection and visualization
    4. Results reporting and interpretation
    
    The analysis provides insights into:
    - Natural groupings of healthcare systems
    - Regional patterns and relationships
    - Key states that bridge different healthcare approaches
    - Overall network structure and connectivity
    
    Results are both visualized and printed as detailed statistics.
    """
    # Load and process the raw healthcare data
    merged_df = load_and_process_data()
    
    # Calculate the similarity matrix between states
    balance_matrix, states = calculate_balance_metric(merged_df)
    
    # Create and analyze the network structure
    G = create_clustered_network(balance_matrix, states)
    
    # Generate the network visualization
    visualize_clustered_network(G)
    
    # Perform cluster analysis
    cluster_stats = analyze_clusters(G)
    
    # Print detailed analysis results
    print("\nCluster Analysis:")
    for cluster_id, stats in cluster_stats.items():
        print(f"\nCluster {cluster_id}:")
        print(f"States: {', '.join(stats['states'])}")
        print(f"Regions represented: {', '.join(stats['regions'])}")
        print(f"Average balance score: {stats['avg_balance']:.3f}")
    
    # Identify and print information about central states
    print("\nTop 5 Most Central States:")
    centrality_scores = {node: G.nodes[node]['centrality'] for node in G.nodes()}
    top_central = sorted(centrality_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    for state, score in top_central:
        print(f"{state}: {score:.3f}")
    
    # Print overall network statistics
    print("\nNetwork Summary:")
    print(f"Total number of states: {G.number_of_nodes()}")
    print(f"Total number of connections: {G.number_of_edges()}")
    avg_balance = np.mean([d['weight'] for u, v, d in G.edges(data=True)])
    print(f"Average balance score across network: {avg_balance:.3f}")

    
    metrics = calculate_network_metrics(G)
    print_network_metrics(metrics, "Geographic Metrics")

if __name__ == "__main__":
    main()