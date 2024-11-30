import pandas as pd

# Read the data
population_df = pd.read_csv('/Users/fardeenb/Documents/Projects/cs3891-network-project-test/data/population_by_state.csv')
provider_df = pd.read_csv('/Users/fardeenb/Documents/Projects/cs3891-network-project-test/data/provider.csv')

# Reshape population data to match provider data format
pop_melted = pd.melt(
    population_df,
    id_vars=['region'],
    value_vars=['population_2011', 'population_2012', 'population_2013'],
    var_name='year',
    value_name='population'
)
# Convert year column to match provider data
pop_melted['period'] = pop_melted['year'].str.extract('(\d{4})').astype(int)
pop_melted = pop_melted.drop('year', axis=1)

# Merge the datasets
merged_df = pd.merge(
    provider_df,
    pop_melted,
    on=['region', 'period']
)

# Calculate providers per 100,000 population
merged_df['providers_per_100k'] = (merged_df['total_provider_types'] / merged_df['population']) * 100000

# Round to 2 decimal places
merged_df['providers_per_100k'] = merged_df['providers_per_100k'].round(2)

# Save the results
merged_df.to_csv('providers_per_capita.csv', index=False)

# Print summary statistics
print("\nSummary Statistics for Providers per 100k Population:")
print(merged_df.groupby('period')['providers_per_100k'].agg(['mean', 'min', 'max']))

# Print states with highest and lowest rates for each year
for year in [2011, 2012, 2013]:
    year_data = merged_df[merged_df['period'] == year]
    highest = year_data.nlargest(1, 'providers_per_100k')
    lowest = year_data.nsmallest(1, 'providers_per_100k')
    
    print(f"\nYear {year}:")
    print(f"Highest: {highest['region'].values[0]} ({highest['providers_per_100k'].values[0]:.2f})")
    print(f"Lowest: {lowest['region'].values[0]} ({lowest['providers_per_100k'].values[0]:.2f})")