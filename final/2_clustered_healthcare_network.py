"""
Clustered Healthcare Network Analysis

This module implements a clustered network analysis approach to understand healthcare
provider and spending patterns across US states. It uses community detection and
advanced visualization techniques to identify and display healthcare metric clusters.

Key Features:
- Implements community detection for healthcare metrics
- Provides enhanced visualization of network clusters
- Calculates detailed cluster statistics
- Analyzes inter-cluster relationships

Dependencies:
    - pandas
    - networkx 
    - matplotlib
    - numpy
    - scipy
    - python-louvain (community)

Design Notes:
- Uses Louvain community detection
- Implements custom balance metrics
- Provides detailed cluster analysis
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap
import community.community_louvain as community_louvain


US_REGIONS = {
    'Northeast': ['ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA'],
    'Midwest': ['OH', 'IN', 'IL', 'MI', 'WI', 'MN', 'IA', 'MO', 'ND', 'SD', 'NE', 'KS'],
    'South': ['DE', 'MD', 'DC', 'VA', 'WV', 'NC', 'SC', 'GA', 'FL', 'KY', 'TN', 'AL', 'MS', 'AR', 'LA', 'OK', 'TX'],
    'West': ['MT', 'ID', 'WY', 'CO', 'NM', 'AZ', 'UT', 'NV', 'WA', 'OR', 'CA', 'AK', 'HI']
}

REGION_COLORS = {
    'Northeast': '#D55E00',  # Dark orange
    'Midwest': '#009E73',    # Dark green
    'South': '#0072B2',      # Dark blue
    'West': '#CC79A7'        # Dark pink
}

def get_state_region(state_code):
    """Get the region for a given state code."""
    for region, states in US_REGIONS.items():
        if state_code in states:
            return region
    return 'Unknown'

def load_and_process_data():
    """Load and process the provider and spending data."""
    # Load datasets
    providers_df = pd.read_csv('FINAL_providers.csv')
    spending_df = pd.read_csv('FINAL_spending.csv')
    
    # Aggregate by state
    providers_agg = providers_df.groupby('region_code')['providers_per_100k'].mean().reset_index()
    spending_agg = spending_df.groupby('region_code')['(E058) Total Hospital-Dir Exp'].mean().reset_index()
    
    # Merge datasets
    merged_df = pd.merge(providers_agg, spending_agg, on='region_code')
    merged_df.columns = ['region_code', 'providers_per_100k', 'hospital_expenditure']
    
    return merged_df

def calculate_balance_metric(df):
    """Calculate the balance matrix between states."""
    states = df['region_code'].values
    n_states = len(states)
    balance_matrix = np.zeros((n_states, n_states))
    
    # Normalize the data
    providers_norm = (df['providers_per_100k'] - df['providers_per_100k'].mean()) / df['providers_per_100k'].std()
    spending_norm = (df['hospital_expenditure'] - df['hospital_expenditure'].mean()) / df['hospital_expenditure'].std()
    
    for i in range(n_states):
        for j in range(n_states):
            if i != j:
                provider_diff = abs(providers_norm.iloc[i] - providers_norm.iloc[j])
                spending_diff = abs(spending_norm.iloc[i] - spending_norm.iloc[j])
                balance = (1 / (1 + abs(provider_diff - spending_diff))) ** 2
                balance_matrix[i, j] = balance
    
    return balance_matrix, states

def create_clustered_network(balance_matrix, states, weak_threshold=0.5, strong_threshold=0.8):
    """Create a network with clustering based on balance scores."""
    G = nx.Graph()
    
    # Add nodes
    for state in states:
        G.add_node(state, region=get_state_region(state))
    
    # Add edges with weights and strength marking
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            balance = balance_matrix[i, j]
            if balance > weak_threshold:
                G.add_edge(states[i], states[j],
                          weight=balance,
                          is_strong=balance > strong_threshold)
    
    # Calculate centrality measures
    degree_cent = nx.degree_centrality(G)
    betweenness_cent = nx.betweenness_centrality(G)
    eigenvector_cent = nx.eigenvector_centrality(G, max_iter=1000)
    
    # Combine centrality measures
    for node in G.nodes():
        G.nodes[node]['centrality'] = (
            degree_cent[node] + 
            betweenness_cent[node] + 
            eigenvector_cent[node]
        ) / 3.0
    
    return G

def visualize_clustered_network(G):
    """Visualize the network with enhanced clustering and centrality measures."""
    fig = plt.figure(figsize=(20, 15))
    ax_network = fig.add_axes([0.1, 0.1, 0.7, 0.8])
    ax_colorbar = fig.add_axes([0.85, 0.1, 0.03, 0.8])
    
    # Create weighted adjacency matrix for layout
    adj_matrix = nx.to_numpy_array(G)
    for i, j in np.ndindex(adj_matrix.shape):
        if adj_matrix[i, j] > 0:
            edge_data = G.get_edge_data(list(G.nodes())[i], list(G.nodes())[j])
            if edge_data['is_strong']:
                adj_matrix[i, j] *= 2
    
    # Calculate layout
    pos = nx.spring_layout(
        G,
        k=1.5,
        iterations=100,
        seed=42,
        weight='weight'
    )
    
    # Calculate node sizes based on centrality
    node_sizes = [G.nodes[node]['centrality'] * 3000 + 500 for node in G.nodes()]
    
    # Draw nodes
    node_colors = [REGION_COLORS[G.nodes[node]['region']] for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos,
                          node_color=node_colors,
                          node_size=node_sizes,
                          alpha=0.7,
                          ax=ax_network)
    
    # Create custom colormap
    colors = [(1, 1, 1), (0.5, 0, 0.5)]
    custom_cmap = LinearSegmentedColormap.from_list("custom_purple", colors, N=100)
    
    # Draw edges
    edges = G.edges(data=True)
    
    # Draw weak edges
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
    
    # Draw strong edges
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
    
    # Add labels
    label_pos = {node: (coord[0], coord[1] + 0.02) for node, coord in pos.items()}
    nx.draw_networkx_labels(G, label_pos, font_size=10, font_weight='bold', ax=ax_network)
    
    # Add colorbar
    all_weights = [d['weight'] for u, v, d in edges]
    norm = plt.Normalize(vmin=min(all_weights), vmax=max(all_weights))
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=ax_colorbar, label='Balance Score')
    
    # Add legend
    legend_elements = []
    for region, color in REGION_COLORS.items():
        legend_elements.append(plt.Line2D([], [], color=color, marker='o',
                                        linestyle='None', markersize=10,
                                        label=region, alpha=0.7))
    
    legend_elements.extend([
        plt.Line2D([], [], color='gray', marker='o', linestyle='None',
                  markersize=8, label='Node size: Network centrality',
                  alpha=0.7)
    ])
    
    ax_network.legend(handles=legend_elements,
                     fontsize=12,
                     loc='upper left',
                     bbox_to_anchor=(1.1, 1))
    
    ax_network.set_title('Healthcare Network: Clustered Provider-Spending Balance\n' +
                        'Darker edges indicate stronger balance, node size shows centrality',
                        fontsize=16, pad=20)
    ax_network.axis('off')
    
    plt.savefig('healthcare_clustered_network.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_clusters(G):
    """Analyze the network clusters and their properties."""
    communities = community_louvain.best_partition(G)
    
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
    
    for cluster_id in cluster_stats:
        cluster_states = cluster_stats[cluster_id]['states']
        balances = []
        for u, v, d in G.edges(data=True):
            if u in cluster_states and v in cluster_states:
                balances.append(d['weight'])
        cluster_stats[cluster_id]['avg_balance'] = np.mean(balances) if balances else 0
    
    return cluster_stats

def main():
    """Main function to run the analysis."""
    # Load and process data
    merged_df = load_and_process_data()
    
    # Calculate balance matrix
    balance_matrix, states = calculate_balance_metric(merged_df)
    
    # Create and analyze network
    G = create_clustered_network(balance_matrix, states)
    
    # Visualize network
    visualize_clustered_network(G)
    
    # Analyze clusters
    cluster_stats = analyze_clusters(G)
    
    # Print results
    print("\nCluster Analysis:")
    for cluster_id, stats in cluster_stats.items():
        print(f"\nCluster {cluster_id}:")
        print(f"States: {', '.join(stats['states'])}")
        print(f"Regions represented: {', '.join(stats['regions'])}")
        print(f"Average balance score: {stats['avg_balance']:.3f}")
    
    print("\nTop 5 Most Central States:")
    centrality_scores = {node: G.nodes[node]['centrality'] for node in G.nodes()}
    top_central = sorted(centrality_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    for state, score in top_central:
        print(f"{state}: {score:.3f}")
    
    print("\nNetwork Summary:")
    print(f"Total number of states: {G.number_of_nodes()}")
    print(f"Total number of connections: {G.number_of_edges()}")
    avg_balance = np.mean([d['weight'] for u, v, d in G.edges(data=True)])
    print(f"Average balance score across network: {avg_balance:.3f}")

if __name__ == "__main__":
    main()