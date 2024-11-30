"""
Geographic Healthcare Network Analysis

This module creates a geographically-aware network visualization of healthcare metrics
across US states. It positions nodes (states) according to their approximate geographic
locations while showing relationships between healthcare metrics.

Key Features:
- Uses geographic positioning for nodes based on Albers projection
- Visualizes healthcare relationships with geographic context
- Provides regional analysis of healthcare patterns
- Includes detailed centrality metrics

Dependencies:
    - pandas
    - networkx
    - matplotlib
    - numpy
    - scipy

Notable Components:
    STATE_POSITIONS (dict): Geographic coordinates for each state
    US_REGIONS (dict): Regional grouping of states
    REGION_COLORS (dict): Color scheme for different regions
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# State geographical positions (approximate Albers projection coordinates)
STATE_POSITIONS = {
    'ME': (0.85, 0.85), 'NH': (0.82, 0.8), 'VT': (0.78, 0.8), 'MA': (0.82, 0.75),
    'RI': (0.85, 0.73), 'CT': (0.81, 0.71), 'NY': (0.75, 0.7), 'NJ': (0.78, 0.65),
    'PA': (0.72, 0.65), 'OH': (0.65, 0.62), 'IN': (0.58, 0.58), 'IL': (0.52, 0.55),
    'MI': (0.6, 0.7), 'WI': (0.5, 0.75), 'MN': (0.4, 0.8), 'IA': (0.45, 0.6),
    'MO': (0.48, 0.5), 'ND': (0.35, 0.85), 'SD': (0.35, 0.75), 'NE': (0.35, 0.6),
    'KS': (0.38, 0.5), 'DE': (0.8, 0.6), 'MD': (0.75, 0.58), 'VA': (0.7, 0.55),
    'WV': (0.68, 0.58), 'NC': (0.7, 0.5), 'SC': (0.68, 0.45), 'GA': (0.65, 0.4),
    'FL': (0.65, 0.25), 'KY': (0.62, 0.55), 'TN': (0.6, 0.5), 'AL': (0.58, 0.38),
    'MS': (0.52, 0.38), 'AR': (0.48, 0.42), 'LA': (0.48, 0.32), 'OK': (0.4, 0.42),
    'TX': (0.38, 0.3), 'MT': (0.25, 0.85), 'ID': (0.2, 0.75), 'WY': (0.25, 0.7),
    'CO': (0.28, 0.55), 'NM': (0.28, 0.4), 'AZ': (0.2, 0.4), 'UT': (0.22, 0.6),
    'NV': (0.15, 0.6), 'WA': (0.15, 0.9), 'OR': (0.12, 0.8), 'CA': (0.1, 0.5),
    'AK': (0.15, 0.15), 'HI': (0.25, 0.15)
}

# Define US regions
US_REGIONS = {
    'Northeast': ['ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA'],
    'Midwest': ['OH', 'IN', 'IL', 'MI', 'WI', 'MN', 'IA', 'MO', 'ND', 'SD', 'NE', 'KS'],
    'South': ['DE', 'MD', 'DC', 'VA', 'WV', 'NC', 'SC', 'GA', 'FL', 'KY', 'TN', 'AL', 'MS', 'AR', 'LA', 'OK', 'TX'],
    'West': ['MT', 'ID', 'WY', 'CO', 'NM', 'AZ', 'UT', 'NV', 'WA', 'OR', 'CA', 'AK', 'HI']
}

# Updated color scheme with more distinct colors
REGION_COLORS = {
    'Northeast': '#D55E00',  # Dark orange
    'Midwest': '#009E73',    # Dark green
    'South': '#0072B2',      # Dark blue
    'West': '#CC79A7'        # Dark pink
}

def get_state_region(state_code):
    for region, states in US_REGIONS.items():
        if state_code in states:
            return region
    return 'Unknown'

def load_and_process_data():
    providers_df = pd.read_csv('FINAL_providers.csv')
    spending_df = pd.read_csv('FINAL_spending.csv')
    
    # Aggregate provider data by state (taking mean across years)
    providers_agg = providers_df.groupby('region_code')['providers_per_100k'].mean().reset_index()
    
    # Aggregate spending data by state
    spending_agg = spending_df.groupby('region_code')['(E058) Total Hospital-Dir Exp'].mean().reset_index()
    
    # Merge datasets
    merged_df = pd.merge(providers_agg, spending_agg, on='region_code')
    merged_df.columns = ['region_code', 'providers_per_100k', 'hospital_expenditure']
    
    return merged_df

def calculate_balance_metric(df):
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
                # Enhanced contrast in balance score
                balance = (1 / (1 + abs(provider_diff - spending_diff))) ** 1.5  # Exponent increases contrast
                balance_matrix[i, j] = balance
    
    return balance_matrix, states

def create_network(balance_matrix, states, threshold=0.7):
    G = nx.Graph()
    
    # Add nodes with positions
    for state in states:
        G.add_node(state, 
                  region=get_state_region(state),
                  pos=STATE_POSITIONS[state])
    
    # Add edges with enhanced balance scores
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            balance = balance_matrix[i, j]
            if balance > threshold:
                G.add_edge(states[i], states[j], 
                          weight=balance,
                          balance_score=balance)
    
    return G

def calculate_node_sizes(G):
    # Calculate both degree and betweenness centrality
    degree_cent = nx.degree_centrality(G)
    between_cent = nx.betweenness_centrality(G)
    
    # Combine the centrality measures with weights
    node_importance = {}
    for node in G.nodes():
        # Weight degree centrality more heavily (0.7) than betweenness (0.3)
        node_importance[node] = 0.7 * degree_cent[node] + 0.3 * between_cent[node]
    
    # Scale the sizes more dramatically (base size 500, max size 2500)
    min_importance = min(node_importance.values())
    max_importance = max(node_importance.values())
    scaled_sizes = {node: 500 + (2000 * (imp - min_importance) / (max_importance - min_importance))
                   for node, imp in node_importance.items()}
    
    return scaled_sizes, degree_cent, between_cent

def visualize_network(G):
    fig = plt.figure(figsize=(20, 15))
    ax_network = fig.add_axes([0.1, 0.1, 0.7, 0.8])
    
    # Get node positions and calculate sizes
    pos = nx.get_node_attributes(G, 'pos')
    node_sizes, degree_cent, between_cent = calculate_node_sizes(G)
    
    # Draw nodes with varying sizes
    node_colors = [REGION_COLORS[G.nodes[node]['region']] for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, 
                          node_color=node_colors,
                          node_size=[node_sizes[node] for node in G.nodes()],
                          alpha=0.7,
                          ax=ax_network)
    
    # Draw edges with enhanced color scheme
    edges = G.edges(data=True)
    edge_colors = [e[2]['balance_score'] for e in edges]
    edge_weights = [3 * e[2]['balance_score'] for e in edges]
    
    custom_cmap = plt.cm.RdYlBu_r
    edges_drawn = nx.draw_networkx_edges(G, pos,
                                       edge_color=edge_colors,
                                       edge_cmap=custom_cmap,
                                       width=edge_weights,
                                       alpha=0.6,
                                       ax=ax_network)
    
    # Add labels with size adjustment based on node importance
    label_pos = {k: (v[0], v[1] + 0.02) for k, v in pos.items()}
    label_sizes = {node: 6 + 4 * (size - 500) / 2000 for node, size in node_sizes.items()}
    for node, (x, y) in label_pos.items():
        ax_network.text(x, y, node, 
                       fontsize=label_sizes[node],
                       fontweight='bold',
                       ha='center',
                       va='center')
    
    # Create colorbar
    ax_colorbar = fig.add_axes([0.85, 0.1, 0.03, 0.8])
    norm = plt.Normalize(min(edge_colors), max(edge_colors))
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, cax=ax_colorbar, label='Balance Score')
    
    # Add legend for regions and node sizes
    legend_elements = []
    for region, color in REGION_COLORS.items():
        legend_elements.append(plt.Line2D([], [], color=color, marker='o',
                                        linestyle='None', markersize=10,
                                        label=region, alpha=0.7))
    
    # Add size legend
    legend_elements.append(plt.Line2D([], [], color='gray', marker='o',
                                    linestyle='None', markersize=8,
                                    label='Node size: Network centrality', alpha=0.7))
    
    ax_network.legend(handles=legend_elements,
                     fontsize=12,
                     loc='upper left',
                     bbox_to_anchor=(1.1, 1))
    
    ax_network.set_title('Healthcare Network: Geographic Distribution of Provider-Spending Balance\n' +
                        'Node size indicates network centrality; Darker edges show stronger balance',
                        fontsize=16, pad=20)
    ax_network.set_xlim(-0.1, 1.1)
    ax_network.set_ylim(-0.1, 1.1)
    ax_network.axis('off')
    
    plt.savefig('healthcare_geo_network_with_centrality.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Return centrality metrics for analysis
    return degree_cent, between_cent

def analyze_balance_patterns(G):
    region_stats = {region: {'count': 0, 'avg_balance': 0.0} for region in US_REGIONS.keys()}
    
    for u, v, data in G.edges(data=True):
        region1 = G.nodes[u]['region']
        region2 = G.nodes[v]['region']
        
        if region1 == region2:
            region_stats[region1]['count'] += 1
            region_stats[region1]['avg_balance'] += data['balance_score']
    
    for region in region_stats:
        if region_stats[region]['count'] > 0:
            region_stats[region]['avg_balance'] /= region_stats[region]['count']
    
    return region_stats


def main():
    merged_df = load_and_process_data()
    balance_matrix, states = calculate_balance_metric(merged_df)
    G = create_network(balance_matrix, states)
    node_sizes, degree_cent, between_cent = calculate_node_sizes(G)
    visualize_network(G)
    
    # Calculate combined centrality scores
    combined_centrality = {}
    for node in G.nodes():
        combined_centrality[node] = 0.7 * degree_cent[node] + 0.3 * between_cent[node]
    
    # Sort states by centrality
    sorted_states = sorted(combined_centrality.items(), key=lambda x: x[1], reverse=True)
    
    # Print top 5 and bottom 5 central states
    print("\nTop 5 Most Central States:")
    for state, score in sorted_states[:5]:
        print(f"{state}: {score:.3f} (Region: {get_state_region(state)})")
    
    print("\nBottom 5 Least Central States:")
    for state, score in sorted_states[-5:]:
        print(f"{state}: {score:.3f} (Region: {get_state_region(state)})")
    
    region_stats = analyze_balance_patterns(G)
    
    print("\nRegional Balance Analysis:")
    for region, stats in region_stats.items():
        print(f"\n{region}:")
        print(f"Number of internal connections: {stats['count']}")
        print(f"Average balance score: {stats['avg_balance']:.3f}")
    
    print("\nNetwork Summary:")
    print(f"Total number of states: {G.number_of_nodes()}")
    print(f"Total number of connections: {G.number_of_edges()}")
    avg_balance = np.mean([d['balance_score'] for u, v, d in G.edges(data=True)])
    print(f"Average balance score across network: {avg_balance:.3f}")


if __name__ == "__main__":
    main()