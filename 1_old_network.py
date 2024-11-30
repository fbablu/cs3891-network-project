import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats


# Define US regions
US_REGIONS = {
    'Northeast': ['ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA'],
    'Midwest': ['OH', 'IN', 'IL', 'MI', 'WI', 'MN', 'IA', 'MO', 'ND', 'SD', 'NE', 'KS'],
    'South': ['DE', 'MD', 'DC', 'VA', 'WV', 'NC', 'SC', 'GA', 'FL', 'KY', 'TN', 'AL', 'MS', 'AR', 'LA', 'OK', 'TX'],
    'West': ['MT', 'ID', 'WY', 'CO', 'NM', 'AZ', 'UT', 'NV', 'WA', 'OR', 'CA', 'AK', 'HI']
}

# Define colors for each region (avoiding red and blue)
REGION_COLORS = {
    'Northeast': '#8B4513',  # Saddle Brown
    'Midwest': '#2E8B57',    # Sea Green
    'South': '#00FFFF',      # Cyan
    'West': '#DAA520'        # Goldenrod
}

def get_state_region(state_code):
    for region, states in US_REGIONS.items():
        if state_code in states:
            return region
    return 'Unknown'



def calculate_summary_statistics(providers_df, spending_df):
    """Calculate and save summary statistics"""
    
    def calculate_stats(df):
        # Remove region_code column
        data = df.drop('region_code', axis=1)
        
        # Initialize dictionary to store statistics
        statistics = {}
        
        # Calculate each statistic separately
        statistics['Mean'] = data.mean()
        statistics['Std Dev'] = data.std()
        statistics['Min'] = data.min()
        statistics['Q1'] = data.quantile(0.25)
        statistics['Median'] = data.median()
        statistics['Q3'] = data.quantile(0.75)
        statistics['Max'] = data.max()
        statistics['Variance'] = data.var()
        statistics['Range'] = data.max() - data.min()
        statistics['Sum of Squares'] = (data ** 2).sum()
        statistics['CV (%)'] = (data.std() / data.mean() * 100)
        statistics['Skewness'] = data.apply(lambda x: stats.skew(x))
        statistics['Kurtosis'] = data.apply(lambda x: stats.kurtosis(x))
        
        # Convert to DataFrame
        stats_df = pd.DataFrame(statistics).T
        
        return stats_df.round(2)
    
    # Calculate statistics for both datasets
    provider_stats = calculate_stats(providers_df)
    spending_stats = calculate_stats(spending_df)
    
    # Save statistics to CSV files
    provider_stats.to_csv('provider_statistics.csv')
    spending_stats.to_csv('spending_statistics.csv')
    
    return provider_stats, spending_stats


def load_and_process_data():
    # Load datasets
    providers_df = pd.read_csv('/Users/fardeenb/Documents/Projects/cs3891-network-project/data/ska_state.csv')
    spending_df = pd.read_csv('/Users/fardeenb/Documents/Projects/cs3891-network-project/data/spending.csv')
    
    # Sort both datasets by state and year first
    providers_df = providers_df.sort_values(['region_code', 'period'])
    spending_df = spending_df.sort_values(['region_code', 'period'])
    
    # Print the data shapes before aggregation
    print(f"Providers data shape before aggregation: {providers_df.shape}")
    print(f"Spending data shape before aggregation: {spending_df.shape}")
    
    # Aggregate provider data by state (taking mean across years)
    providers_agg = providers_df.groupby('region_code').agg({
        'all_providers': 'mean',
        'all_primary_care_providers': 'mean',
        'all_physicians': 'mean',
        'all_nurse_practitioners': 'mean',
        'all_physician_assistants': 'mean'
    }).reset_index()
    
    # Aggregate spending data by state
    spending_agg = spending_df.groupby('region_code').agg({
        '(E052) Health & Hosp-Dir Exp': 'mean',
        '(E055) Health-Direct Expend': 'mean'
    }).reset_index()
    
    # Print unique states in each dataset after aggregation
    print("\nStates in providers dataset:", sorted(providers_agg['region_code'].unique()))
    print("States in spending dataset:", sorted(spending_agg['region_code'].unique()))
    
    # Find common states
    common_states = list(set(providers_agg['region_code']) & set(spending_agg['region_code']))
    common_states.sort()  # Sort alphabetically
    
    # Filter to include only common states
    providers_agg = providers_agg[providers_agg['region_code'].isin(common_states)]
    spending_agg = spending_agg[spending_agg['region_code'].isin(common_states)]
    
    # Sort both dataframes by region_code to ensure alignment
    providers_agg = providers_agg.sort_values('region_code').reset_index(drop=True)
    spending_agg = spending_agg.sort_values('region_code').reset_index(drop=True)
    
    print(f"\nNumber of states in final analysis: {len(common_states)}")
    
    # Verify alignment
    alignment_check = all(providers_agg['region_code'] == spending_agg['region_code'])
    print(f"Data alignment verified: {alignment_check}")
    
    return providers_agg, spending_agg

def calculate_similarity_matrices(providers_df, spending_df):
    # Verify alignment again
    assert all(providers_df['region_code'] == spending_df['region_code']), "State mismatch between datasets"
    
    states = providers_df['region_code'].values
    n_states = len(states)
    provider_similarity = np.zeros((n_states, n_states))
    spending_similarity = np.zeros((n_states, n_states))
    
    # Normalize provider data
    provider_features = providers_df.drop('region_code', axis=1)
    provider_normalized = (provider_features - provider_features.mean()) / provider_features.std()
    
    # Normalize spending data
    spending_features = spending_df.drop('region_code', axis=1)
    spending_normalized = (spending_features - spending_features.mean()) / spending_features.std()
    
    # Convert to numpy arrays for faster computation
    provider_data = provider_normalized.values
    spending_data = spending_normalized.values
    
    # Calculate similarity matrices
    for i in range(n_states):
        for j in range(n_states):
            if i != j:
                # Provider similarity using cosine similarity
                provider_sim = np.dot(provider_data[i], provider_data[j]) / (
                    np.linalg.norm(provider_data[i]) * np.linalg.norm(provider_data[j]))
                provider_similarity[i, j] = provider_sim
                
                # Spending similarity using cosine similarity
                spending_sim = np.dot(spending_data[i], spending_data[j]) / (
                    np.linalg.norm(spending_data[i]) * np.linalg.norm(spending_data[j]))
                spending_similarity[i, j] = spending_sim
    
    return provider_similarity, spending_similarity, states

def create_network(provider_similarity, spending_similarity, states, provider_threshold=0.85, spending_threshold=0.85):
    G = nx.MultiGraph()
    
    # Add nodes
    for state in states:
        G.add_node(state)
    
    # Add edges for provider similarity (red edges)
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            if provider_similarity[i, j] > provider_threshold:
                G.add_edge(states[i], states[j], 
                          weight=provider_similarity[i, j],
                          edge_type='provider',
                          similarity=provider_similarity[i, j])
    
    # Add edges for spending similarity (blue edges)
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            if spending_similarity[i, j] > spending_threshold:
                G.add_edge(states[i], states[j],
                          weight=spending_similarity[i, j],
                          edge_type='spending',
                          similarity=spending_similarity[i, j])
    
    return G



def visualize_network(G):
    plt.figure(figsize=(20, 15))
    
    # Create layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Get node colors based on regions
    node_colors = [REGION_COLORS[get_state_region(node)] for node in G.nodes()]
    
    # Draw nodes with regional colors (changed from 'lightgray' to node_colors)
    nx.draw_networkx_nodes(G, pos, 
                          node_color=node_colors,  # Use the node_colors list here
                          node_size=700,
                          alpha=0.7)
    
    # Draw edges with different colors based on type
    edges_by_type = {
        'provider': [(u, v, k) for (u, v, k, d) in G.edges(data=True, keys=True) 
                    if d['edge_type'] == 'provider'],
        'spending': [(u, v, k) for (u, v, k, d) in G.edges(data=True, keys=True) 
                    if d['edge_type'] == 'spending']
    }
    
    # Draw provider edges (red)
    if edges_by_type['provider']:
        weights = [G[u][v][k]['similarity'] for (u, v, k) in edges_by_type['provider']]
        nx.draw_networkx_edges(G, pos,
                             edgelist=[(u, v) for (u, v, k) in edges_by_type['provider']],
                             edge_color=weights,
                             edge_cmap=plt.cm.Reds,
                             width=2,
                             alpha=0.6)
    
    # Draw spending edges (blue)
    if edges_by_type['spending']:
        weights = [G[u][v][k]['similarity'] for (u, v, k) in edges_by_type['spending']]
        nx.draw_networkx_edges(G, pos,
                             edgelist=[(u, v) for (u, v, k) in edges_by_type['spending']],
                             edge_color=weights,
                             edge_cmap=plt.cm.Blues,
                             width=2,
                             alpha=0.6)
    
    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    # Add legend
    legend_elements = [
        plt.Line2D([], [], color='red', label='Provider Similarity', alpha=0.6),
        plt.Line2D([], [], color='blue', label='Spending Similarity', alpha=0.6)
    ]
    
    # Add region colors to legend
    for region, color in REGION_COLORS.items():
        legend_elements.append(plt.Line2D([], [], color=color, marker='o',
                                        linestyle='None', markersize=10,
                                        label=region, alpha=0.7))
    
    plt.legend(handles=legend_elements, fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.title('Healthcare Network: State Similarities by Region\nRed = Provider Similarity, Blue = Spending Similarity', 
              fontsize=16, pad=20)
    plt.axis('off')
    
    # Save the plot
    plt.savefig('healthcare_network.png', dpi=300, bbox_inches='tight')
    plt.close()




def main():
    # Load and process data
    providers_df, spending_df = load_and_process_data()
    
    # Calculate and save summary statistics
    provider_stats, spending_stats = calculate_summary_statistics(providers_df, spending_df)
    
    # Print summary statistics
    print("\nProvider Statistics:")
    print(provider_stats)
    print("\nSpending Statistics:")
    print(spending_stats)
    
    # Calculate similarity matrices
    provider_similarity, spending_similarity, states = calculate_similarity_matrices(providers_df, spending_df)
    
    # Create network
    G = create_network(provider_similarity, spending_similarity, states)
    
    # Generate visualization
    visualize_network(G)
    
    # Print network statistics
    print("\nNetwork Analysis Summary:")
    print(f"Number of nodes (states): {G.number_of_nodes()}")
    
    provider_edges = len([(u, v) for (u, v, d) in G.edges(data=True) if d['edge_type'] == 'provider'])
    spending_edges = len([(u, v) for (u, v, d) in G.edges(data=True) if d['edge_type'] == 'spending'])
    
    print(f"Provider similarity edges: {provider_edges}")
    print(f"Spending similarity edges: {spending_edges}")
    print(f"Total edges: {G.number_of_edges()}")
    
    # Print regional statistics
    print("\nNetwork Analysis by Region:")
    for region in US_REGIONS.keys():
        region_states = [state for state in G.nodes() if get_state_region(state) == region]
        region_edges = [e for e in G.edges(data=True) 
                       if get_state_region(e[0]) == region and get_state_region(e[1]) == region]
        
        print(f"\n{region}:")
        print(f"Number of states: {len(region_states)}")
        print(f"Internal connections: {len(region_edges)}")

if __name__ == "__main__":
    main()