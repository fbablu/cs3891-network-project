import pandas as pd

# Read the provider CSV file
df = pd.read_csv('/Users/fardeenb/Documents/Projects/cs3891-network-project-test/data/raw_data/ska_state.csv')

# List of columns to sum
provider_columns = [
    'all_providers',
    'all_primary_care_providers',
    'all_physicians',
    'all_primary_care_physicians',
    'all_nurse_practitioners',
    'all_primary_care_nurse_practitioners',
    'all_physician_assistants',
    'all_primary_care_physician_assistants'
]

# Create a new dataframe with only the columns we want
transformed_df = df[['region', 'region_code', 'period']].copy()

# Add the summed column
transformed_df['total_provider_types'] = df[provider_columns].sum(axis=1)

# Sort the dataframe to match the format (by region and period)
transformed_df = transformed_df.sort_values(['region', 'period'])

# Save the transformed data
output_path = '/Users/fardeenb/Documents/Projects/cs3891-network-project-test/data/provider.csv'
transformed_df.to_csv(output_path, index=False)