"""
Healthcare Provider Density Calculator

This script calculates population-adjusted healthcare provider metrics by state.
It combines provider counts with population data to create meaningful comparisons
across states of different sizes.

Key Operations:
- Combines provider counts with population data
- Calculates providers per 100,000 residents
- Generates summary statistics and rankings
- Identifies extremes in provider density

The resulting metrics enable fair comparisons of healthcare provider availability
across states with varying populations.
"""

import pandas as pd

# Load both required datasets
# We need both provider counts and population figures to calculate density
population_df = pd.read_csv('/Users/fardeenb/Documents/Projects/cs3891-network-project-test/data/population_by_state.csv')
provider_df = pd.read_csv('/Users/fardeenb/Documents/Projects/cs3891-network-project-test/data/provider.csv')

# Reshape population data to match provider data structure
# Transform wide format (years as columns) to long format (years as rows)
pop_melted = pd.melt(
    population_df,
    id_vars=['region'],                                        # Keep region as identifier
    value_vars=['population_2011', 'population_2012', 'population_2013'],  # Years to transform
    var_name='year',                                          # Name for year column
    value_name='population'                                   # Name for population values
)

# Extract year from column names and convert to integer
# This standardizes the year format to match the provider data
pop_melted['period'] = pop_melted['year'].str.extract('(\d{4})').astype(int)
pop_melted = pop_melted.drop('year', axis=1)  # Remove original year column

# Combine provider and population data
# This brings together all the information needed for density calculations
merged_df = pd.merge(
    provider_df,
    pop_melted,
    on=['region', 'period']  # Match records by state and year
)

# Calculate provider density (providers per 100,000 population)
# This creates a standardized metric for comparing across states
merged_df['providers_per_100k'] = (merged_df['total_provider_types'] / merged_df['population']) * 100000

# Round the results for cleaner presentation
# Two decimal places provides sufficient precision while remaining readable
merged_df['providers_per_100k'] = merged_df['providers_per_100k'].round(2)

# Save the processed data
# This creates our final dataset for analysis
merged_df.to_csv('providers_per_capita.csv', index=False)

# Generate and display summary statistics
# This provides an overview of provider density distribution
print("\nSummary Statistics for Providers per 100k Population:")
print(merged_df.groupby('period')['providers_per_100k'].agg(['mean', 'min', 'max']))

# Identify and display extremes for each year
# This helps identify states with unusual provider density
for year in [2011, 2012, 2013]:
    # Filter data for the current year
    year_data = merged_df[merged_df['period'] == year]
    
    # Find highest and lowest provider density states
    highest = year_data.nlargest(1, 'providers_per_100k')
    lowest = year_data.nsmallest(1, 'providers_per_100k')
    
    # Display results with clear formatting
    print(f"\nYear {year}:")
    print(f"Highest: {highest['region'].values[0]} ({highest['providers_per_100k'].values[0]:.2f})")
    print(f"Lowest: {lowest['region'].values[0]} ({lowest['providers_per_100k'].values[0]:.2f})")