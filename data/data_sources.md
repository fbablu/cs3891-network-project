# Data Sources and Processing Pipeline

## Primary Data Sources

### 1. US Census Bureau Population Data
- Source: [Census Bureau Population Estimates](https://www2.census.gov/programs-surveys/popest/tables/2010-2013/state/totals/)
- File: `population_by_state.csv`
- Years: 2011-2013
- Purpose: Population denominators for per capita calculations

### 2. Healthcare Provider Data
- Source: [Office of the National Coordinator for Health IT](https://www.healthit.gov/data/datasets/office-based-health-care-providers-database)
- Raw files: `ska_all.csv`, `ska_state.csv`
- Purpose: Provider counts and distributions

### 3. Government Spending Data
- Source: [US Census Bureau Annual Survey of State and Local Government Finances (via Urban Institute)](state-local-finance-data.taxpolicycenter.org)
- Raw file: `spending_unprocessed.csv`
- Metric: Hospital Direct Expenditure (E058)
- Years: 2011-2013

## Data Processing Steps

### 1. Provider Data Processing
```
ska_all.csv → ska_state.csv → provider.csv
- Removed county-level data
- Aggregated provider types into single column
```

### 2. Final Dataset Creation

#### FINAL_providers.csv
- Merged `provider.csv` with population data
- Calculated providers per 100k using formula:
```python
providers_per_100k = (total_provider_types / population) * 100000
```

#### FINAL_spending.csv
- Cleaned spending data from Urban Institute
- Standardized format for integration with provider data

### 3. Statistical Analysis Files

#### combined_statistics.csv
- Summary statistics for:
  - Providers per 100k population
  - Hospital direct expenditure
- Includes: mean, standard deviation, quartiles, variance, etc.

#### aggregated_state_data.csv
- State-level provider type summaries
- Used for detailed statistical analysis

This pipeline creates a comprehensive dataset linking healthcare provider distribution with government healthcare spending across US states for the years 2011-2013.