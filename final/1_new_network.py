from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from networkx.exception import NetworkXNotImplemented


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

def load_and_process_data():
    # Load datasets
    providers_df = pd.read_csv('FINAL_providers.csv')
    spending_df = pd.read_csv('FINAL_spending.csv')
    
    # Sort both datasets by state and year first
    providers_df = providers_df.sort_values(['region_code', 'period'])
    spending_df = spending_df.sort_values(['region_code', 'period'])
    
    # Print the data shapes before aggregation
    print(f"Providers data shape before aggregation: {providers_df.shape}")
    print(f"Spending data shape before aggregation: {spending_df.shape}")
    
    # Aggregate provider data by state (taking mean across years)
    providers_agg = providers_df.groupby('region_code').agg({
        'providers_per_100k': 'mean'
    }).reset_index()
    
    # Aggregate spending data by state
    spending_agg = spending_df.groupby('region_code').agg({
        '(E058) Total Hospital-Dir Exp': 'mean'
    }).reset_index()
    
    # Print unique states in each dataset after aggregation
    print("\nStates in providers dataset:", sorted(providers_agg['region_code'].unique()))
    print("States in spending dataset:", sorted(spending_agg['region_code'].unique()))
    
    # Find common states
    common_states = list(set(providers_agg['region_code']) & set(spending_agg['region_code']))
    common_states.sort()
    
    # Filter to include only common states
    providers_agg = providers_agg[providers_agg['region_code'].isin(common_states)]
    spending_agg = spending_agg[spending_agg['region_code'].isin(common_states)]
    
    # Sort both dataframes by region_code to ensure alignment
    providers_agg = providers_agg.sort_values('region_code').reset_index(drop=True)
    spending_agg = spending_agg.sort_values('region_code').reset_index(drop=True)
    
    print(f"\nNumber of states in final analysis: {len(common_states)}")
    
    return providers_agg, spending_agg

def calculate_similarity_matrices(providers_df, spending_df):
    states = providers_df['region_code'].values
    n_states = len(states)
    
    # Initialize arrays for similarity calculations
    provider_similarity = np.zeros((n_states, n_states))
    spending_similarity = np.zeros((n_states, n_states))
    combined_similarity = np.zeros((n_states, n_states))
    balance_ratios = np.zeros((n_states, n_states))
    
    # Normalize provider data
    provider_values = providers_df['providers_per_100k'].values
    provider_normalized = (provider_values - provider_values.mean()) / provider_values.std()
    
    # Normalize spending data
    spending_values = spending_df['(E058) Total Hospital-Dir Exp'].values
    spending_normalized = (spending_values - spending_values.mean()) / spending_values.std()
    
    # Calculate similarity matrices and balance ratio
    for i in range(n_states):
        for j in range(n_states):
            if i != j:
                # Provider similarity using normalized difference
                provider_diff = abs(provider_normalized[i] - provider_normalized[j])
                provider_sim = 1 / (1 + provider_diff)
                provider_similarity[i, j] = provider_sim
                
                # Spending similarity using normalized difference
                spending_diff = abs(spending_normalized[i] - spending_normalized[j])
                spending_sim = 1 / (1 + spending_diff)
                spending_similarity[i, j] = spending_sim
                
                # Calculate balance ratio (α) = provider_similarity / spending_similarity
                balance_ratio = provider_sim / spending_sim if spending_sim != 0 else float('inf')
                balance_ratios[i, j] = balance_ratio
                
                # Combined similarity metric
                # High when both similarities are high and their ratio is close to 1
                ratio_balance = 1 / (1 + abs(1 - balance_ratio))  # Peaks at ratio = 1
                combined_similarity[i, j] = (provider_sim * spending_sim * ratio_balance) ** (1/3)
    
    return combined_similarity, balance_ratios, states

def create_network(combined_similarity, balance_ratios, states, similarity_threshold=0.7):
    G = nx.Graph()
    
    # Add nodes
    for state in states:
        G.add_node(state)
    
    # Add edges with combined similarity and balance information
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            if combined_similarity[i, j] > similarity_threshold:
                G.add_edge(states[i], states[j],
                          weight=combined_similarity[i, j],
                          balance_ratio=balance_ratios[i, j])
    
    return G



def calculate_node_sizes(G):
    # Previous conversion code remains the same
    G_simple = nx.Graph()
    for u, v, data in G.edges(data=True):
        w = data['weight']
        if G_simple.has_edge(u, v):
            G_simple[u][v]['weight'] = max(w, G_simple[u][v]['weight'])
        else:
            G_simple.add_edge(u, v, weight=w)
    
    # Calculate centrality measures
    degree_cent = nx.degree_centrality(G_simple)
    betweenness_cent = nx.betweenness_centrality(G_simple, weight='weight')
    eigenvector_cent = nx.eigenvector_centrality_numpy(G_simple, weight='weight')
    
    # Combine centrality measures with non-linear scaling
    combined_centrality = {}
    for node in G.nodes():
        base_score = (
            0.4 * degree_cent[node] +
            0.3 * betweenness_cent[node] +
            0.3 * eigenvector_cent[node]
        )
        # Apply exponential scaling to emphasize differences
        combined_centrality[node] = base_score ** 0.5
    
    # More dramatic size range
    min_size = 200   # Smaller minimum
    max_size = 4000  # Larger maximum
    min_cent = min(combined_centrality.values())
    max_cent = max(combined_centrality.values())
    
    # Non-linear scaling for node sizes
    sizes = {node: min_size + (max_size - min_size) * 
            ((score - min_cent) / (max_cent - min_cent)) ** 0.5
            for node, score in combined_centrality.items()}
    
    return sizes, combined_centrality







def visualize_network(G):
    # Create figure and axes with proper spacing for colorbar
    fig = plt.figure(figsize=(24, 18))
    ax_network = fig.add_axes([0.1, 0.1, 0.7, 0.8])  # [left, bottom, width, height]
    ax_colorbar = fig.add_axes([0.85, 0.1, 0.03, 0.8])  # Position for colorbar
    
    # Calculate node sizes based on centrality
    node_sizes, centrality_scores = calculate_node_sizes(G)
    
    # Create layout with more space between nodes
    pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)
    
    # Get node colors based on regions
    node_colors = [REGION_COLORS[get_state_region(node)] for node in G.nodes()]
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                          node_color=node_colors,
                          node_size=[node_sizes[node] for node in G.nodes()],
                          alpha=0.85,
                          ax=ax_network)
    
    # Create custom colormap for balance ratio
    colors = [(0.8, 0, 0),      # Red for provider-dominated
             (0.4, 0.4, 0.4),  # Gray for balanced
             (0, 0, 0.8)]      # Blue for spending-dominated
    custom_cmap = LinearSegmentedColormap.from_list("custom", colors)
    
    # Draw edges with color based on balance ratio
    edges = G.edges(data=True)
    edge_colors = []
    edge_weights = []
    
    for (u, v, data) in edges:
        ratio = data['balance_ratio']
        # Convert ratio to color scale (-1 to 1)
        color_val = 2 * (1 / (1 + abs(np.log2(ratio)))) - 1
        edge_colors.append(color_val)
        edge_weights.append(3 * data['weight'])
    
    nx.draw_networkx_edges(G, pos,
                          edge_color=edge_colors,
                          edge_cmap=custom_cmap,
                          width=edge_weights,
                          alpha=0.75,
                          ax=ax_network)
    
    # Add labels with white background
    label_sizes = {}
    min_font, max_font = 6, 16
    min_size = min(node_sizes.values())
    max_size = max(node_sizes.values())
    
    for node in G.nodes():
        size_scale = ((node_sizes[node] - min_size) / (max_size - min_size)) ** 0.7
        label_sizes[node] = min_font + size_scale * (max_font - min_font)
    
    for node, (x, y) in pos.items():
        ax_network.text(x, y, node,
                       fontsize=label_sizes[node],
                       ha='center',
                       va='center',
                       fontweight='bold',
                       bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))
    
    # Add colorbar for edge colors
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, 
                              norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, cax=ax_colorbar)
    ax_colorbar.set_title('Balance Ratio\nRed=Provider\nBlue=Spending', 
                         fontsize=12, pad=20)
    
    # Add legend for regions
    legend_elements = []
    for region, color in REGION_COLORS.items():
        legend_elements.append(plt.Line2D([], [], color=color, marker='o',
                                        linestyle='None', markersize=12,
                                        label=region, alpha=0.85))
    
    legend_elements.append(plt.Line2D([], [], color='gray', marker='o',
                                    linestyle='None', markersize=12,
                                    label='Node size = Centrality', alpha=0.85))
    
    ax_network.legend(handles=legend_elements, fontsize=14, loc='center left', 
                     bbox_to_anchor=(1.1, 0.5))
    
    ax_network.set_title('Healthcare Network: Combined Provider-Spending Similarity\n' +
                        'Edge color indicates provider/spending balance\n' +
                        'Edge thickness shows overall similarity strength',
                        fontsize=20, pad=20)
    ax_network.axis('off')
    
    plt.savefig('healthcare_combined_network.png', dpi=400, bbox_inches='tight')
    plt.close()

def analyze_balance_patterns(G):
    print("\nBalance Analysis:")
    
    # Initialize lists for different categories
    balanced_pairs = []
    provider_dominated = []
    spending_dominated = []
    
    for (u, v, data) in G.edges(data=True):
        ratio = data['balance_ratio']
        combined_strength = data['weight']
        
        # Categorize based on balance ratio
        if 0.9 <= ratio <= 1.1:  # Balanced
            balanced_pairs.append((u, v, ratio, combined_strength))
        elif ratio > 1.1:  # Provider-dominated
            provider_dominated.append((u, v, ratio, combined_strength))
        else:  # Spending-dominated
            spending_dominated.append((u, v, ratio, combined_strength))
    
    print(f"\nTop 5 Most Balanced State Pairs (α ≈ 1):")
    for pair in sorted(balanced_pairs, key=lambda x: abs(1 - x[2]))[:5]:
        print(f"{pair[0]}-{pair[1]}: α={pair[2]:.2f}, strength={pair[3]:.3f}")
    
    print(f"\nBottom 5 Least Balanced State Pairs:")
    for pair in sorted(balanced_pairs, key=lambda x: abs(1 - x[2]), reverse=True)[:5]:
        print(f"{pair[0]}-{pair[1]}: α={pair[2]:.2f}, strength={pair[3]:.3f}")
    
    print(f"\nTop 5 Most Provider-Dominated Pairs (α >> 1):")
    for pair in sorted(provider_dominated, key=lambda x: x[2], reverse=True)[:5]:
        print(f"{pair[0]}-{pair[1]}: α={pair[2]:.2f}, strength={pair[3]:.3f}")
    
    print(f"\nBottom 5 Least Provider-Dominated Pairs (α ≈ 1.1):")
    for pair in sorted(provider_dominated, key=lambda x: x[2])[:5]:
        print(f"{pair[0]}-{pair[1]}: α={pair[2]:.2f}, strength={pair[3]:.3f}")
    
    print(f"\nTop 5 Most Spending-Dominated Pairs (α << 1):")
    for pair in sorted(spending_dominated, key=lambda x: x[2])[:5]:
        print(f"{pair[0]}-{pair[1]}: α={pair[2]:.2f}, strength={pair[3]:.3f}")
    
    print(f"\nBottom 5 Least Spending-Dominated Pairs (α ≈ 0.9):")
    for pair in sorted(spending_dominated, key=lambda x: x[2], reverse=True)[:5]:
        print(f"{pair[0]}-{pair[1]}: α={pair[2]:.2f}, strength={pair[3]:.3f}")


def analyze_centrality(G):
    _, centrality_scores = calculate_node_sizes(G)
    
    # Sort nodes by centrality
    sorted_nodes = sorted(centrality_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("\nCentrality Analysis:")
    print("\nTop 5 Most Central States:")
    for state, score in sorted_nodes[:5]:
        region = get_state_region(state)
        print(f"{state} ({region}): {score:.3f}")
    
    print("\nBottom 5 Least Central States:")
    for state, score in sorted_nodes[-5:]:
        region = get_state_region(state)
        print(f"{state} ({region}): {score:.3f}")

def main():
    providers_df, spending_df = load_and_process_data()
    combined_similarity, balance_ratios, states = calculate_similarity_matrices(providers_df, spending_df)
    G = create_network(combined_similarity, balance_ratios, states)
    visualize_network(G)
    analyze_balance_patterns(G)
    analyze_centrality(G)
    
    
    
    
if __name__ == "__main__":
    main()