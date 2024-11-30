import pandas as pd
import numpy as np
from pathlib import Path

def load_data(file_path: str, type: str) -> pd.DataFrame:
    """
    Load and prepare data from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        type (str): Type of data ('providers' or 'spending')
        
    Returns:
        pd.DataFrame: Processed data
    """
    try:
        df = pd.read_csv(file_path)
        
        # For providers data, we want the providers_per_100k column
        if type == 'providers':
            return df[['region', 'region_code', 'period', 'providers_per_100k']]
        
        # For spending data, we want the spending column
        elif type == 'spending':
            return df[['region', 'region_code', 'period', '(E058) Total Hospital-Dir Exp']]
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at {file_path}")
    except Exception as e:
        raise Exception(f"Error processing data: {str(e)}")

def calculate_statistics(df: pd.DataFrame, value_column: str) -> pd.Series:
    """
    Calculate comprehensive summary statistics for the data.
    
    Args:
        df (pd.DataFrame): Input data
        value_column (str): Name of the column to analyze
        
    Returns:
        pd.Series: Summary statistics
    """
    stats = pd.Series({
        'Mean': df[value_column].mean(),
        'Std Dev': df[value_column].std(),
        'Min': df[value_column].min(),
        'Q1': df[value_column].quantile(0.25),
        'Median': df[value_column].median(),
        'Q3': df[value_column].quantile(0.75),
        'Max': df[value_column].max(),
        'Variance': df[value_column].var(),
        'Range': df[value_column].max() - df[value_column].min(),
        'Sum of Squares': (df[value_column] ** 2).sum(),
        'CV (%)': (df[value_column].std() / df[value_column].mean() * 100),
        'Skewness': df[value_column].skew(),
        'Kurtosis': df[value_column].kurtosis()
    })
    
    return stats

def main():
    try:
        # Load the data
        providers_df = load_data('/Users/fardeenb/Documents/Projects/cs3891-network-project-test/data/FINAL_providers.csv', 'providers')
        spending_df = load_data('/Users/fardeenb/Documents/Projects/cs3891-network-project-test/data/FINAL_spending.csv', 'spending')
        
        # Calculate statistics for both datasets
        provider_stats = calculate_statistics(providers_df, 'providers_per_100k')
        spending_stats = calculate_statistics(spending_df, '(E058) Total Hospital-Dir Exp')
        
        # Combine statistics into a single DataFrame
        combined_stats = pd.DataFrame({
            'Providers per 100k': provider_stats,
            'Hospital Direct Expenditure': spending_stats
        })
        
        # Round the results to 2 decimal places
        combined_stats = combined_stats.round(2)
        
        # Save the results
        combined_stats.to_csv('combined_statistics.csv')
        
        # Display the results
        print("\nCombined Summary Statistics:")
        print(combined_stats)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()