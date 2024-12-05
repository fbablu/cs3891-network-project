"""
Network Analysis for Healthcare Provider and Spending Data

This module performs a sophisticated network analysis to understand relationships between 
healthcare systems across US states. By examining both provider density and spending 
patterns, it reveals how different states' healthcare approaches relate to each other.

The analysis transforms raw healthcare data into an intuitive network visualization where:
- States are represented as nodes in the network
- Relationships between states are shown as connecting edges
- Node sizes indicate a state's influence in the healthcare network
- Colors help identify geographic patterns
- Edge characteristics reveal the nature of healthcare relationships

Key Features:
- Creates a network based on provider and spending similarities between states
- Visualizes the network with node sizes based on centrality
- Colors nodes by geographic region for spatial pattern recognition
- Analyzes balance patterns and centrality metrics
- Provides detailed statistics about network relationships

Dependencies:
    - matplotlib: For creating sophisticated visualizations
    - pandas: For efficient data handling and analysis
    - networkx: For network creation and analysis
    - numpy: For numerical computations
    - scipy: For statistical operations

Constants:
    US_REGIONS (dict): Mapping of US regions to their constituent states
    REGION_COLORS (dict): Carefully selected color scheme for different US regions
"""

from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from networkx.exception import NetworkXNotImplemented
from network_info.network_metrics import calculate_network_metrics, print_network_metrics
# Define US regions based on Census Bureau classifications
# This grouping enables analysis of geographic patterns in healthcare approaches
US_REGIONS = {
    'Northeast': ['ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA'],
    'Midwest': ['OH', 'IN', 'IL', 'MI', 'WI', 'MN', 'IA', 'MO', 'ND', 'SD', 'NE', 'KS'],
    'South': ['DE', 'MD', 'DC', 'VA', 'WV', 'NC', 'SC', 'GA', 'FL', 'KY', 'TN', 'AL', 'MS', 'AR', 'LA', 'OK', 'TX'],
    'West': ['MT', 'ID', 'WY', 'CO', 'NM', 'AZ', 'UT', 'NV', 'WA', 'OR', 'CA', 'AK', 'HI']
}

# Color scheme chosen for optimal visual distinction while maintaining readability
# Colors are selected to be colorblind-friendly and visually pleasing
REGION_COLORS = {
    'Northeast': '#8B4513',  # Saddle Brown - warm and earthy
    'Midwest': '#2E8B57',    # Sea Green - natural and balanced
    'South': '#00FFFF',      # Cyan - bright and distinct
    'West': '#DAA520'        # Goldenrod - rich and visible
}

def get_state_region(state_code):
    """
    Identify which census region a state belongs to.
    
    This function serves as a mapping tool to connect individual states with their
    broader geographic regions, enabling regional pattern analysis in the healthcare
    network.
    
    Args:
        state_code (str): Two-letter state abbreviation (e.g., 'CA' for California)
        
    Returns:
        str: The census region name ('Northeast', 'Midwest', 'South', 'West') or
             'Unknown' if the state code isn't recognized
    """
    for region, states in US_REGIONS.items():
        if state_code in states:
            return region
    return 'Unknown'

def load_and_process_data():
    """
    Load and prepare healthcare data for network analysis.
    
    This function handles the critical first step of data preparation by:
    1. Loading raw provider and spending data
    2. Ensuring temporal consistency through sorting
    3. Aggregating data to state level
    4. Aligning datasets for comparison
    5. Validating data completeness
    
    The function includes detailed progress reporting to track the data
    preparation process and identify potential issues early.
    
    Returns:
        tuple: (providers_agg, spending_agg)
            - providers_agg (pd.DataFrame): State-level provider density data
            - spending_agg (pd.DataFrame): State-level healthcare spending data
    """
    # Load the raw datasets from CSV files
    providers_df = pd.read_csv('FINAL_providers.csv')
    spending_df = pd.read_csv('FINAL_spending.csv')
    
    # Ensure consistent ordering for reliable analysis
    providers_df = providers_df.sort_values(['region_code', 'period'])
    spending_df = spending_df.sort_values(['region_code', 'period'])
    
    # Report initial data dimensions for verification
    print(f"Providers data shape before aggregation: {providers_df.shape}")
    print(f"Spending data shape before aggregation: {spending_df.shape}")
    
    # Create state-level aggregates by averaging across years
    providers_agg = providers_df.groupby('region_code').agg({
        'providers_per_100k': 'mean'
    }).reset_index()
    
    spending_agg = spending_df.groupby('region_code').agg({
        '(E058) Total Hospital-Dir Exp': 'mean'
    }).reset_index()
    
    # Verify state coverage in each dataset
    print("\nStates in providers dataset:", sorted(providers_agg['region_code'].unique()))
    print("States in spending dataset:", sorted(spending_agg['region_code'].unique()))
    
    # Identify states present in both datasets for complete analysis
    common_states = list(set(providers_agg['region_code']) & set(spending_agg['region_code']))
    common_states.sort()
    
    # Filter to include only states with complete data
    providers_agg = providers_agg[providers_agg['region_code'].isin(common_states)]
    spending_agg = spending_agg[spending_agg['region_code'].isin(common_states)]
    
    # Ensure consistent state ordering
    providers_agg = providers_agg.sort_values('region_code').reset_index(drop=True)
    spending_agg = spending_agg.sort_values('region_code').reset_index(drop=True)
    
    print(f"\nNumber of states in final analysis: {len(common_states)}")
    
    return providers_agg, spending_agg

def calculate_similarity_matrices(providers_df, spending_df):
    """
    Calculate comprehensive similarity metrics between states' healthcare systems.
    
    This function implements a sophisticated approach to quantify how similar states
    are in their healthcare delivery, considering both provider density and spending.
    It creates several types of similarity measures:
    
    1. Provider similarity: How close states are in provider density
    2. Spending similarity: How close states are in healthcare spending
    3. Balance ratio: The relationship between provider and spending similarities
    4. Combined similarity: A unified measure incorporating all aspects
    
    The similarity calculations use normalized metrics to ensure fair comparisons
    between different scales of measurement.
    
    Args:
        providers_df (pd.DataFrame): Provider density data by state
        spending_df (pd.DataFrame): Healthcare spending data by state
        
    Returns:
        tuple: (combined_similarity, balance_ratios, states)
            - combined_similarity (np.array): Matrix of overall healthcare similarities
            - balance_ratios (np.array): Matrix of provider/spending balance ratios
            - states (np.array): State codes in same order as matrix dimensions
    """
    states = providers_df['region_code'].values
    n_states = len(states)
    
    # Initialize matrices for storing different similarity measures
    provider_similarity = np.zeros((n_states, n_states))
    spending_similarity = np.zeros((n_states, n_states))
    combined_similarity = np.zeros((n_states, n_states))
    balance_ratios = np.zeros((n_states, n_states))
    
    # Normalize data to make metrics comparable
    provider_values = providers_df['providers_per_100k'].values
    provider_normalized = (provider_values - provider_values.mean()) / provider_values.std()
    
    spending_values = spending_df['(E058) Total Hospital-Dir Exp'].values
    spending_normalized = (spending_values - spending_values.mean()) / spending_values.std()
    
    # Calculate similarities between each pair of states
    for i in range(n_states):
        for j in range(n_states):
            if i != j:  # Skip self-comparisons
                # Calculate provider similarity using inverse of normalized difference
                provider_diff = abs(provider_normalized[i] - provider_normalized[j])
                provider_sim = 1 / (1 + provider_diff)
                provider_similarity[i, j] = provider_sim
                
                # Calculate spending similarity similarly
                spending_diff = abs(spending_normalized[i] - spending_normalized[j])
                spending_sim = 1 / (1 + spending_diff)
                spending_similarity[i, j] = spending_sim
                
                # Calculate balance ratio (α) to understand provider/spending relationship
                balance_ratio = provider_sim / spending_sim if spending_sim != 0 else float('inf')
                balance_ratios[i, j] = balance_ratio
                
                # Create combined similarity that considers all aspects:
                # - High when both similarities are high
                # - Peaks when provider and spending patterns are balanced
                ratio_balance = 1 / (1 + abs(1 - balance_ratio))
                combined_similarity[i, j] = (provider_sim * spending_sim * ratio_balance) ** (1/3)
    
    return combined_similarity, balance_ratios, states

def create_network(combined_similarity, balance_ratios, states, similarity_threshold=0.7):
    """
    Create a network representation of healthcare relationships between states.
    
    This function transforms the similarity matrices into a network structure where:
    - Nodes represent states
    - Edges connect states with similar healthcare patterns
    - Edge weights indicate the strength of similarity
    - Edge attributes store details about the provider/spending balance
    
    The similarity threshold determines which relationships are strong enough to
    include in the network, helping to focus on the most significant patterns.
    
    Args:
        combined_similarity (np.array): Matrix of overall healthcare similarities
        balance_ratios (np.array): Matrix of provider/spending balance ratios
        states (np.array): Array of state codes
        similarity_threshold (float): Minimum similarity to create an edge (default: 0.7)
        
    Returns:
        nx.Graph: Network structure representing healthcare relationships
    """
    G = nx.Graph()
    
    # Add all states as nodes
    for state in states:
        G.add_node(state)
    
    # Create edges for sufficiently similar states
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            if combined_similarity[i, j] > similarity_threshold:
                G.add_edge(states[i], states[j],
                          weight=combined_similarity[i, j],
                          balance_ratio=balance_ratios[i, j])
    
    return G

def calculate_node_sizes(G):
    """
    Calculate node sizes based on sophisticated centrality analysis.
    
    This function determines how important each state is in the healthcare network by:
    1. Creating a simplified network for centrality calculations
    2. Computing multiple centrality measures (degree, betweenness, eigenvector)
    3. Combining these measures with careful weighting
    4. Applying non-linear scaling to create visually meaningful sizes
    
    The resulting sizes emphasize differences between central and peripheral states
    while maintaining readability of the visualization.
    
    Args:
        G (nx.Graph): The healthcare network
        
    Returns:
        tuple: (sizes, combined_centrality)
            - sizes (dict): Node size values for visualization
            - combined_centrality (dict): Raw centrality scores for analysis
    """
    # Create a simplified network for centrality calculations
    # This helps avoid issues with multiple edges between the same nodes
    G_simple = nx.Graph()
    for u, v, data in G.edges(data=True):
        w = data['weight']
        if G_simple.has_edge(u, v):
            # Keep the stronger connection if multiple exist
            G_simple[u][v]['weight'] = max(w, G_simple[u][v]['weight'])
        else:
            G_simple.add_edge(u, v, weight=w)
    
    # Calculate different types of centrality
    # Each measure captures a different aspect of importance
    degree_cent = nx.degree_centrality(G_simple)  # Local importance
    betweenness_cent = nx.betweenness_centrality(G_simple, weight='weight')  # Bridge importance
    eigenvector_cent = nx.eigenvector_centrality_numpy(G_simple, weight='weight')  # Global importance
    
    # Combine centrality measures with non-linear scaling
    # This creates more meaningful differentiation between states
    combined_centrality = {}
    for node in G.nodes():
        base_score = (
            0.4 * degree_cent[node] +      # Emphasize direct connections
            0.3 * betweenness_cent[node] + # Value bridge positions
            0.3 * eigenvector_cent[node]   # Consider system-wide influence
        )
        # Apply exponential scaling to emphasize differences
        combined_centrality[node] = base_score ** 0.5
    
    # Create visually appropriate node sizes
    min_size = 200   # Ensure smallest nodes are visible
    max_size = 4000  # Allow prominent display of important nodes
    min_cent = min(combined_centrality.values())
    max_cent = max(combined_centrality.values())
    
    # Apply non-linear scaling to node sizes
    # This creates clear visual hierarchy while maintaining readability
    sizes = {node: min_size + (max_size - min_size) * 
            ((score - min_cent) / (max_cent - min_cent)) ** 0.5
            for node, score in combined_centrality.items()}
    
    return sizes, combined_centrality

def visualize_network(G):
    """
    Create a sophisticated visualization of the healthcare network.
    
    This function produces a comprehensive visualization that shows:
    - States as nodes, with size indicating network importance
    - Geographic regions through color-coding
    - Healthcare relationship patterns through edge properties
    - Balance between providers and spending through edge colors
    
    The visualization includes multiple visual encodings:
    - Node size: Overall importance in the network
    - Node color: Geographic region
    - Edge thickness: Strength of relationship
    - Edge color: Provider vs spending balance
    - Labels: State identification with size-appropriate text
    
    Args:
        G (nx.Graph): The healthcare network to visualize
    """
    # Set up the figure with space for main plot and legends
    fig = plt.figure(figsize=(24, 18))
    ax_network = fig.add_axes([0.1, 0.1, 0.7, 0.8])  # Main network plot
    ax_colorbar = fig.add_axes([0.85, 0.1, 0.03, 0.8])  # Color scale
    
    # Calculate node sizes based on network importance
    node_sizes, centrality_scores = calculate_node_sizes(G)
    
    # Create network layout with good spacing
    pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)
    
    # Assign colors based on geographic regions
    node_colors = [REGION_COLORS[get_state_region(node)] for node in G.nodes()]
    
    # Draw the nodes
    nx.draw_networkx_nodes(G, pos, 
                          node_color=node_colors,
                          node_size=[node_sizes[node] for node in G.nodes()],
                          alpha=0.85,
                          ax=ax_network)
    
    # Create custom colormap for balance visualization
    # Red indicates provider dominance, blue indicates spending dominance
    colors = [(0.8, 0, 0),      # Red for provider-dominated
             (0.4, 0.4, 0.4),  # Gray for balanced
             (0, 0, 0.8)]      # Blue for spending-dominated
    custom_cmap = LinearSegmentedColormap.from_list("custom", colors)
    
    # Process and draw edges
    edges = G.edges(data=True)
    edge_colors = []
    edge_weights = []
    
    for (u, v, data) in edges:
        ratio = data['balance_ratio']
        # Convert ratio to color scale (-1 to 1)
        # Using log scale to handle ratios more effectively
        color_val = 2 * (1 / (1 + abs(np.log2(ratio)))) - 1
        edge_colors.append(color_val)
        edge_weights.append(3 * data['weight'])
    
    # Draw the edges
    nx.draw_networkx_edges(G, pos,
                          edge_color=edge_colors,
                          edge_cmap=custom_cmap,
                          width=edge_weights,
                          alpha=0.75,
                          ax=ax_network)
    
    # Add state labels with size-appropriate formatting
    label_sizes = {}
    min_font, max_font = 6, 16
    min_size = min(node_sizes.values())
    max_size = max(node_sizes.values())
    
    # Scale label sizes with node sizes
    for node in G.nodes():
        size_scale = ((node_sizes[node] - min_size) / (max_size - min_size)) ** 0.7
        label_sizes[node] = min_font + size_scale * (max_font - min_font)
    
    # Add labels with white background for readability
    for node, (x, y) in pos.items():
        ax_network.text(x, y, node,
                       fontsize=label_sizes[node],
                       ha='center',
                       va='center',
                       fontweight='bold',
                       bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))
    
    # Add colorbar for edge balance interpretation
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, 
                              norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, cax=ax_colorbar)
    ax_colorbar.set_title('Balance Ratio\nRed=Provider\nBlue=Spending', 
                         fontsize=12, pad=20)
    
    # Create legend for regions and node sizes
    legend_elements = []
    for region, color in REGION_COLORS.items():
        legend_elements.append(plt.Line2D([], [], color=color, marker='o',
                                        linestyle='None', markersize=12,
                                        label=region, alpha=0.85))
    
    legend_elements.append(plt.Line2D([], [], color='gray', marker='o',
                                    linestyle='None', markersize=12,
                                    label='Node size = Centrality', alpha=0.85))
    
    # Position legend and add title
    ax_network.legend(handles=legend_elements, fontsize=14, loc='center left', 
                     bbox_to_anchor=(1.1, 0.5))
    
    ax_network.set_title('Healthcare Network: Combined Provider-Spending Similarity\n' +
                        'Edge color indicates provider/spending balance\n' +
                        'Edge thickness shows overall similarity strength',
                        fontsize=20, pad=20)
    ax_network.axis('off')
    
    # Save high-resolution figure
    plt.savefig('healthcare_combined_network.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_balance_patterns(G):
    """
    Perform detailed analysis of healthcare balance patterns across states.
    
    This function examines how states relate to each other in terms of their
    provider-to-spending balance, revealing different patterns in healthcare delivery.
    It categorizes relationships into three key groups:
    1. Balanced pairs: States with similar provider-spending ratios
    2. Provider-dominated pairs: States where provider metrics show stronger similarity
    3. Spending-dominated pairs: States where spending patterns are more aligned
    
    The analysis helps understand how different states prioritize and balance
    their healthcare resources.
    
    Args:
        G (nx.Graph): The healthcare network with balance information
    """
    print("\nBalance Analysis:")
    
    # Initialize categories for different types of relationships
    balanced_pairs = []          # States with similar provider-spending ratios
    provider_dominated = []      # States more similar in provider patterns
    spending_dominated = []      # States more similar in spending patterns
    
    # Examine each relationship in the network
    for (u, v, data) in G.edges(data=True):
        ratio = data['balance_ratio']
        combined_strength = data['weight']
        
        # Categorize relationships based on balance ratio
        # Ratios near 1 indicate balanced relationships
        if 0.9 <= ratio <= 1.1:  # Balanced
            balanced_pairs.append((u, v, ratio, combined_strength))
        elif ratio > 1.1:  # Provider patterns dominate
            provider_dominated.append((u, v, ratio, combined_strength))
        else:  # Spending patterns dominate
            spending_dominated.append((u, v, ratio, combined_strength))
    
    # Report the most interesting cases in each category
    
    # Most balanced relationships show healthcare systems with similar approaches
    print(f"\nTop 5 Most Balanced State Pairs (α ≈ 1):")
    for pair in sorted(balanced_pairs, key=lambda x: abs(1 - x[2]))[:5]:
        print(f"{pair[0]}-{pair[1]}: α={pair[2]:.2f}, strength={pair[3]:.3f}")
    
    # Least balanced relationships might indicate divergent healthcare strategies
    print(f"\nBottom 5 Least Balanced State Pairs:")
    for pair in sorted(balanced_pairs, key=lambda x: abs(1 - x[2]), reverse=True)[:5]:
        print(f"{pair[0]}-{pair[1]}: α={pair[2]:.2f}, strength={pair[3]:.3f}")
    
    # Provider-dominated relationships suggest similar healthcare delivery approaches
    print(f"\nTop 5 Most Provider-Dominated Pairs (α >> 1):")
    for pair in sorted(provider_dominated, key=lambda x: x[2], reverse=True)[:5]:
        print(f"{pair[0]}-{pair[1]}: α={pair[2]:.2f}, strength={pair[3]:.3f}")
    
    # Weakly provider-dominated pairs show subtle differences in approach
    print(f"\nBottom 5 Least Provider-Dominated Pairs (α ≈ 1.1):")
    for pair in sorted(provider_dominated, key=lambda x: x[2])[:5]:
        print(f"{pair[0]}-{pair[1]}: α={pair[2]:.2f}, strength={pair[3]:.3f}")
    
    # Spending-dominated relationships indicate similar financial approaches
    print(f"\nTop 5 Most Spending-Dominated Pairs (α << 1):")
    for pair in sorted(spending_dominated, key=lambda x: x[2])[:5]:
        print(f"{pair[0]}-{pair[1]}: α={pair[2]:.2f}, strength={pair[3]:.3f}")
    
    # Weakly spending-dominated pairs show subtle financial alignment
    print(f"\nBottom 5 Least Spending-Dominated Pairs (α ≈ 0.9):")
    for pair in sorted(spending_dominated, key=lambda x: x[2], reverse=True)[:5]:
        print(f"{pair[0]}-{pair[1]}: α={pair[2]:.2f}, strength={pair[3]:.3f}")


def analyze_centrality(G):
    """
    Analyze the importance of different states in the healthcare network.
    
    This function identifies which states are most central or influential in the
    healthcare network and which are more peripheral. Centrality here represents
    how well-connected a state is to others with similar healthcare patterns.
    
    The analysis reveals:
    - States that share healthcare patterns with many others
    - States with unique or isolated healthcare approaches
    - Potential bridge states that connect different healthcare patterns
    
    Args:
        G (nx.Graph): The healthcare network
    """
    # Calculate centrality scores for all states
    _, centrality_scores = calculate_node_sizes(G)
    
    # Sort states by their centrality scores
    sorted_nodes = sorted(centrality_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Report the most central states - these often represent common healthcare patterns
    print("\nCentrality Analysis:")
    print("\nTop 5 Most Central States:")
    for state, score in sorted_nodes[:5]:
        region = get_state_region(state)
        print(f"{state} ({region}): {score:.3f}")
    
    # Report the least central states - these often have unique approaches
    print("\nBottom 5 Least Central States:")
    for state, score in sorted_nodes[-5:]:
        region = get_state_region(state)
        print(f"{state} ({region}): {score:.3f}")

def main():
    """
    Execute the complete healthcare network analysis pipeline.
    
    This function orchestrates the entire analysis process, from data loading
    through visualization and analysis. It follows a careful sequence:
    1. Load and preprocess the healthcare data
    2. Calculate similarity measures between states
    3. Create and visualize the network
    4. Analyze patterns in healthcare relationships
    5. Identify important states and relationships
    
    The function serves as both an entry point for the analysis and a
    demonstration of the proper sequence for using the module's components.
    """
    # Load and prepare the data
    providers_df, spending_df = load_and_process_data()
    
    # Calculate similarity metrics
    combined_similarity, balance_ratios, states = calculate_similarity_matrices(providers_df, spending_df)
    
    # Create and analyze the network
    G = create_network(combined_similarity, balance_ratios, states)
    
    # Generate visualizations and analysis
    visualize_network(G)
    analyze_balance_patterns(G)
    analyze_centrality(G)
    
    
    metrics = calculate_network_metrics(G)
    print_network_metrics(metrics, "Geographic Metrics")


# Entry point for running the analysis
if __name__ == "__main__":
    main()