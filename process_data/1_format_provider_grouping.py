"""
Healthcare Provider Data Aggregation Processor

This script processes raw healthcare provider data to create a unified dataset of total
provider counts across different categories. It handles the initial data transformation
step in our healthcare analysis pipeline.

Key Operations:
- Loads raw provider data by state and year
- Aggregates different types of healthcare providers
- Creates a standardized format for further analysis
- Ensures data consistency through sorting

The resulting dataset serves as a foundation for understanding healthcare provider
distribution across states over time.
"""

import pandas as pd

# Load the raw provider data from the source CSV file
# This file contains detailed breakdowns of different provider types by state and year
df = pd.read_csv('/Users/fardeenb/Documents/Projects/cs3891-network-project-test/data/ska_state.csv')

# Define the provider categories to be included in the analysis
# These categories represent different types of healthcare providers,
# including both general practitioners and specialists
provider_columns = [
    'all_providers',                           # Total providers of all types
    'all_primary_care_providers',              # Providers focused on primary care
    'all_physicians',                          # Medical doctors
    'all_primary_care_physicians',             # Primary care physicians
    'all_nurse_practitioners',                 # Advanced practice nurses
    'all_primary_care_nurse_practitioners',    # Primary care nurse practitioners
    'all_physician_assistants',                # Physician assistants
    'all_primary_care_physician_assistants'    # Primary care physician assistants
]

# Create a new dataframe with essential identifying columns
# This establishes the basic structure for our aggregated data
transformed_df = df[['region', 'region_code', 'period']].copy()

# Calculate total providers by summing across all provider types
# This gives us a comprehensive count of healthcare providers in each state
transformed_df['total_provider_types'] = df[provider_columns].sum(axis=1)

# Sort the data for consistency and easier analysis
# Organizing by region and period makes patterns easier to identify
transformed_df = transformed_df.sort_values(['region', 'period'])

# Save the processed data to a new CSV file
# This creates a clean, standardized dataset for subsequent analysis steps
output_path = '/Users/fardeenb/Documents/Projects/cs3891-network-project-test/data/provider.csv'
transformed_df.to_csv(output_path, index=False)