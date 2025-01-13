# Mapping Health: Analyzing Provider Distribution and Government Investment Across U.S. States, 2011–2013
*CS 3891 / Network Analysis in Healthcare with Dr. You Chen*

## Overview
This project employs sophisticated network analysis techniques to examine the relationships between healthcare provider density and government spending across U.S. states. The analysis focuses on a critical period (2011-2013) coinciding with early Affordable Care Act (ACA) implementation, revealing complex patterns in healthcare resource distribution that transcend traditional geographic boundaries.

## Research Questions
1. How do patterns of healthcare provider density correlate with government healthcare spending across different U.S. states and regions?
2. Which states serve as central nodes or bridges in the healthcare resource network, and what characteristics do these states share?

## Key Findings
- Striking contrast between provider density consistency (CV = 13.44%) and spending variability (CV = 75.4%)
- Provider distribution approximates normal distribution (kurtosis = -0.18)
- Substantial variation in spending ($8.00 to $624.00 per capita)
- Northeast region shows highest provider density (mean = 787.6 providers per 100k)
- Massachusetts and Minnesota emerged as key bridge states with balanced provider-to-spending ratios

## Data Sources
1. **US Census Bureau Population Data** (2011-2013)
   - Population denominators for per capita calculations
   - Source: Census Bureau Population Estimates

2. **Healthcare Provider Data**
   - Provider counts and distributions
   - Source: Office of the National Coordinator for Health IT
   - Metrics: Provider density per 100,000 population

3. **Government Spending Data**
   - Hospital Direct Expenditure (E058)
   - Source: US Census Bureau Annual Survey of State and Local Government Finances
   - Years: 2011-2013

## Methodology

### Network Analysis Approaches

1. **Geographic Network Analysis** (`1_geographic_healthcare_network.py`)
   - Preserves spatial relationships between states
   - Uses Albers projection for accurate state positioning
   - Implements weighted edges based on healthcare similarities
   - Calculates multiple centrality measures:
     - Degree centrality
     - Betweenness centrality
     - Eigenvector centrality

2. **Clustered Network Analysis** (`2_clustered_healthcare_network.py`)
   - Employs Louvain algorithm for community detection
   - Identifies clusters of states with similar healthcare profiles
   - Analyzes intra-cluster and inter-cluster relationships
   - Quantifies cluster cohesion and separation

3. **Combined Metrics Network** (`3_combined_metrics_network.py`)
   - Integrates provider density and spending patterns
   - Implements sophisticated similarity scoring
   - Uses exponential scaling for relationship strength
   - Provides comprehensive network statistics

### Statistical Methods

1. **Distribution Analysis**
   - Provider density analysis showing normal distribution
   - Spending pattern variation analysis
   - Regional distribution comparisons
   - Outlier identification and analysis

2. **Similarity Metrics**
   - Provider-spending balance ratios
   - Normalized cross-state comparisons
   - Regional pattern quantification
   - Bridge state identification

3. **Network Metrics**
   - Network density calculations
   - Clustering coefficient analysis
   - Centrality measurements
   - Edge weight distribution analysis

## Project Components

### Visualization Outputs
1. Geographic network maps showing spatial relationships
2. Clustered network diagrams revealing healthcare communities
3. Balance ratio heatmaps displaying provider-spending relationships
4. Regional pattern visualizations
5. Statistical distribution plots

### Analysis Reports
- Network metric summaries
- Regional pattern analysis
- Bridge state identification
- Provider density statistics
- Spending pattern analysis
- Combined metrics reports

## Dependencies
```
matplotlib>=3.7.1
networkx>=3.1
numpy>=1.24.3
pandas>=2.0.0
pyproj>=3.5.0
scipy>=1.10.1
seaborn>=0.12.2
plotly>=5.13.0
jupyter>=1.0.0
ipykernel>=6.0.0
```

## Significance
This research provides crucial insights for policymakers and healthcare administrators by:
- Identifying effective resource allocation strategies
- Revealing patterns that transcend geographic boundaries
- Demonstrating the importance of balanced provider-to-spending ratios
- Highlighting successful state models for healthcare resource distribution

## Conclusions
The analysis reveals that effective resource allocation strategies, rather than absolute spending levels, are crucial for maintaining optimal healthcare coverage. The network analysis approach uncovers complex relationships between provider density and government spending that would not be apparent through conventional statistical methods.

## Author
Fardeen Bablu  
CS 3891-05 Final Project

*Note: This research coincides with significant healthcare policy changes during the early implementation phases of the Affordable Care Act (ACA), providing valuable insights into healthcare resource distribution during this pivotal period.*
