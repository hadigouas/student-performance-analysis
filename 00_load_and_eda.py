"""
Student Performance Analysis - EDA
This script loads and explores the StudentPerformanceFactors.csv dataset.
It examines the data structure, checks for missing values, and creates visualizations.
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# Set plot style
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Create output directory for plots
os.makedirs('output/eda_plots', exist_ok=True)

def load_data(file_path):
    """Load and return the dataset."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found. Please check the file path.")
    return pd.read_csv(file_path)

def detect_target_column(df):
    """Detect and return the target column name."""
    target_candidates = [col for col in df.columns if 'final' in col.lower() or 'score' in col.lower()or 'exam_score' in col.lower()]
    if target_candidates:
        return target_candidates[0]
    return None
# ----
def basic_data_analysis(df):
    """Perform basic data analysis and return summary."""
    # Basic info
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing_percentage': (df.isnull().mean() * 100).round(2).to_dict(),
        'numeric_columns': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist()
    }
    
    # Detect target column
    target_col = detect_target_column(df)
    if target_col:
        info['target_column'] = target_col
        
        # Calculate correlations with target if it's numeric
        if target_col in info['numeric_columns']:
            correlations = df[info['numeric_columns']].corr()[target_col].sort_values(ascending=False)
            info['correlations_with_target'] = correlations.to_dict()
    
    return info

def plot_distributions(df, numeric_cols, output_dir):
    """Plot distributions for numeric columns."""
    for col in numeric_cols:
        plt.figure(figsize=(10, 4))
        sns.histplot(data=df, x=col, kde=True)
        plt.title(f'Distribution of {col}')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/dist_{col}.png')
        plt.close()

def plot_categorical_counts(df, categorical_cols, output_dir, max_categories=10):
    """Plot value counts for categorical columns."""
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        if len(value_counts) > max_categories:
            print(f"Skipping {col} - too many categories ({len(value_counts)} > {max_categories})")
            continue
            
        plt.figure(figsize=(10, 4))
        sns.countplot(data=df, x=col, order=value_counts.index)
        plt.title(f'Count of {col}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/count_{col}.png')
        plt.close()

def plot_correlation_heatmap(df, numeric_cols, output_dir):
    """Plot correlation heatmap for numeric columns."""
    if len(numeric_cols) < 2:
        print("Not enough numeric columns for correlation heatmap.")
        return
        
    plt.figure(figsize=(12, 10))
    corr = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_heatmap.png')
    plt.close()

def main():
    """Main function to run the EDA process."""
    # Load data
    file_path = 'StudentPerformanceFactors.csv'
    df = load_data(file_path)
    
    # Perform basic analysis
    analysis = basic_data_analysis(df)
    
    # Save analysis summary
    with open('output/eda_summary.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Create EDA summary CSV
    eda_summary = pd.DataFrame({
        'column': list(analysis['dtypes'].keys()),
        'dtype': list(analysis['dtypes'].values()),
        'missing_percentage': [analysis['missing_percentage'].get(col, 0) for col in analysis['dtypes'].keys()]
    })
    
    # Add correlation with target if available
    if 'correlations_with_target' in analysis:
        eda_summary['correlation_with_target'] = eda_summary['column'].map(
            analysis['correlations_with_target']
        )
    
    eda_summary.to_csv('output/eda_summary.csv', index=False)
    
    # Create visualizations
    plot_distributions(df, analysis['numeric_columns'], 'output/eda_plots')
    plot_categorical_counts(df, analysis['categorical_columns'], 'output/eda_plots')
    
    if len(analysis['numeric_columns']) > 1:
        plot_correlation_heatmap(df, analysis['numeric_columns'], 'output/eda_plots')
    
    print("✅ EDA completed successfully! Check the 'output/eda_plots' and 'output/eda_summary.json' for results.")

if __name__ == "__main__":
    main()
