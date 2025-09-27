"""
Cluster Analysis Script
This script analyzes the clusters to identify common features and characteristics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

def analyze_clusters():
    """Analyze clusters and identify common features."""
    
    print("🔍 Starting Cluster Analysis...")
    
    # Load data
    try:
        df_original = pd.read_csv("StudentPerformanceFactors.csv")
        cluster_assignments = pd.read_csv("output/clustering/cluster_assignments.csv")
        df = df_original.copy()
        df['cluster'] = cluster_assignments['cluster']
    except FileNotFoundError as e:
        print(f"❌ Error loading data: {e}")
        return
    
    n_clusters = df['cluster'].nunique()
    print(f"\n📊 Found {n_clusters} clusters.")
    
    # Analyze features
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if 'cluster' in numeric_cols:
        numeric_cols.remove('cluster')
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    cluster_analysis = {}
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id]
        cluster_size = len(cluster_data)
        
        analysis = {
            'size': cluster_size,
            'percentage': (cluster_size / len(df)) * 100,
            'numeric_features': {},
            'categorical_features': {},
            'key_characteristics': [],
            'performance_profile': {}
        }
        
        # Performance metrics
        avg_exam = cluster_data['Exam_Score'].mean()
        overall_avg_exam = df['Exam_Score'].mean()
        analysis['performance_profile'] = {
            'avg_exam_score': avg_exam,
            'exam_score_vs_overall': avg_exam - overall_avg_exam,
        }
        
        # Determine cluster type
        if avg_exam > overall_avg_exam + 5:
            analysis['cluster_type'] = "High Performers"
        elif avg_exam < overall_avg_exam - 5:
            analysis['cluster_type'] = "Needs Support"
        else:
            analysis['cluster_type'] = "Average Performers"
        
        # Analyze numeric features
        cluster_means = cluster_data[numeric_cols].mean()
        overall_means = df[numeric_cols].mean()
        differences = (cluster_means - overall_means).abs().sort_values(ascending=False)
        
        for feature in differences.head(5).index:
            analysis['numeric_features'][feature] = {
                'cluster_mean': cluster_means[feature],
                'overall_mean': overall_means[feature],
                'difference': cluster_means[feature] - overall_means[feature]
            }
        
        cluster_analysis[f'cluster_{cluster_id}'] = analysis
    
    # Save detailed analysis
    output_dir = Path("output/clustering")
    with open(output_dir / "cluster_analysis_detailed.json", 'w') as f:
        json.dump(cluster_analysis, f, indent=2, default=str)
    
    # Create and save summary report
    summary_lines = ["STUDENT PERFORMANCE CLUSTER ANALYSIS REPORT", "="*50]
    for cluster_id in sorted(df['cluster'].unique()):
        analysis = cluster_analysis[f'cluster_{cluster_id}']
        summary_lines.append(f"\nCLUSTER {cluster_id}: {analysis['cluster_type']}")
        summary_lines.append(f"Size: {analysis['size']} students ({analysis['percentage']:.1f}%)")
        summary_lines.append(f"Avg Exam Score: {analysis['performance_profile']['avg_exam_score']:.1f}")
    
    with open(output_dir / "cluster_analysis_summary.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"\n✅ Analysis complete! Reports saved to: {output_dir}")

if __name__ == "__main__":
    analyze_clusters()