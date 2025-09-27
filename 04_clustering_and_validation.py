"""
Clustering and Validation
This script performs clustering on the student performance data and validates the clusters.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator
import matplotlib.pyplot as plt

# Constants
RANDOM_STATE = 42
K_RANGE = range(2, 11)  # Range of k values to try
OUTPUT_DIR = 'output/clustering'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """Load the preprocessed data."""
    X = np.load('output/features/X_fused.npy')
    
    # Standardize the data again for clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, X.shape[1]  # Return scaled data and original feature count

def find_optimal_k_elbow(X_scaled, max_k=10):
    """Find the optimal number of clusters using elbow method and silhouette score."""
    print("Finding optimal number of clusters using elbow method...")
    
    # Calculate WCSS (Within-Cluster Sum of Squares) for elbow method
    wcss = []
    silhouette_scores = []
    k_values = list(range(2, min(max_k + 1, len(X_scaled))))
    
    for k in k_values:
        print(f"  Testing k={k}...")
        
        # KMeans for elbow method (WCSS calculation)
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        kmeans_labels = kmeans.fit_predict(X_scaled)
        wcss.append(kmeans.inertia_)
        
        # Calculate silhouette scores for both methods
        kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
        
        # Agglomerative Clustering (silhouette only, no WCSS)
        agg = AgglomerativeClustering(n_clusters=k)
        agg_labels = agg.fit_predict(X_scaled)
        agg_silhouette = silhouette_score(X_scaled, agg_labels)
        
        silhouette_scores.append({
            'k': k,
            'wcss': kmeans.inertia_,
            'kmeans_silhouette': kmeans_silhouette,
            'agg_silhouette': agg_silhouette
        })
    
    # Find elbow using KneeLocator
    try:
        knee_locator = KneeLocator(k_values, wcss, curve="convex", direction="decreasing")
        elbow_k = knee_locator.elbow
        print(f"  Elbow method suggests k={elbow_k}")
    except Exception as e:
        print(f"  Warning: KneeLocator failed ({e}), using manual elbow detection...")
        elbow_k = find_elbow_manual(k_values, wcss)
    
    # If elbow detection fails, use silhouette score as fallback
    if elbow_k is None:
        print("  Elbow detection failed, using silhouette score as fallback...")
        df = pd.DataFrame(silhouette_scores)
        best_row = df.loc[df[['kmeans_silhouette', 'agg_silhouette']].max(axis=1).idxmax()]
        elbow_k = best_row['k']
        print(f"  Silhouette method suggests k={elbow_k}")
    
    # Create visualization
    create_elbow_plot(k_values, wcss, silhouette_scores, elbow_k)
    
    return pd.DataFrame(silhouette_scores), elbow_k

def find_elbow_manual(k_values, wcss):
    """Manual elbow detection using rate of change."""
    if len(wcss) < 3:
        return k_values[0] if k_values else 2
    
    # Calculate rate of change (second derivative)
    rates = []
    for i in range(1, len(wcss) - 1):
        rate = wcss[i-1] - 2*wcss[i] + wcss[i+1]
        rates.append(rate)
    
    if not rates:
        return k_values[0] if k_values else 2
    
    # Find the point with maximum rate of change
    elbow_idx = np.argmax(rates) + 1  # +1 because rates start from index 1
    return k_values[elbow_idx] if elbow_idx < len(k_values) else k_values[-1]

def create_elbow_plot(k_values, wcss, silhouette_data, optimal_k):
    """Create and save elbow plot with silhouette scores."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Elbow plot
    ax1.plot(k_values, wcss, 'bo-', linewidth=2, markersize=8)
    ax1.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, 
                label=f'Optimal k={optimal_k}')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Within-Cluster Sum of Squares (WCSS)')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Silhouette scores plot
    df = pd.DataFrame(silhouette_data)
    ax2.plot(df['k'], df['kmeans_silhouette'], 'go-', label='KMeans', linewidth=2, markersize=6)
    ax2.plot(df['k'], df['agg_silhouette'], 'ro-', label='Agglomerative', linewidth=2, markersize=6)
    ax2.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, 
                label=f'Optimal k={optimal_k}')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Scores by k')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/elbow_and_silhouette_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Elbow and silhouette analysis plot saved to: {OUTPUT_DIR}/elbow_and_silhouette_analysis.png")

def train_best_clusterer(X_scaled, k):
    """Train the best clustering model for the given k."""
    # Ensure k is an integer and within valid range
    k = max(2, min(int(round(k)), len(X_scaled) - 1))  # At least 2 clusters, at most n_samples-1
    
    print(f"Training with k={k} clusters...")
    
    # Try both KMeans and Agglomerative clustering
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    kmeans_score = silhouette_score(X_scaled, kmeans_labels)
    print(f"  KMeans silhouette score: {kmeans_score:.4f}")
    
    agg = AgglomerativeClustering(n_clusters=k)
    agg_labels = agg.fit_predict(X_scaled)
    agg_score = silhouette_score(X_scaled, agg_labels)
    print(f"  Agglomerative silhouette score: {agg_score:.4f}")
    
    # Return the better performing model
    if kmeans_score >= agg_score:
        print("  Selected KMeans as the better model")
        return kmeans, kmeans_labels, 'kmeans'
    else:
        print("  Selected Agglomerative as the better model")
        return agg, agg_labels, 'agglomerative'

def save_cluster_visualization(labels, method, output_file):
    """Save cluster information."""
    # Create a simple visualization of cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(10, 6))
    plt.bar(unique, counts)
    plt.title(f'Cluster Sizes ({method.__class__.__name__})')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Samples')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def main():
    """Main function to run the clustering and validation process."""
    print("Starting clustering analysis with elbow method...")
    
    # Load and prepare data
    X_scaled, _ = load_data()
    
    # Find optimal number of clusters using elbow method
    silhouette_df, optimal_k = find_optimal_k_elbow(X_scaled)
    silhouette_df.to_csv(f"{OUTPUT_DIR}/clustering_silhouettes.csv", index=False)
    
    # Also show silhouette score analysis for comparison
    best_silhouette_row = silhouette_df.loc[silhouette_df[['kmeans_silhouette', 'agg_silhouette']].max(axis=1).idxmax()]
    best_silhouette_k = best_silhouette_row['k']
    
    # Use elbow method result
    best_k = int(round(optimal_k))
    
    # Save best k and analysis
    with open(f"{OUTPUT_DIR}/best_k.txt", 'w') as f:
        f.write(str(best_k))
    
    with open(f"{OUTPUT_DIR}/k_selection_analysis.txt", 'w') as f:
        f.write(f"K Selection Analysis\n")
        f.write(f"==================\n")
        f.write(f"Elbow method selected: k={optimal_k}\n")
        f.write(f"Silhouette method suggests: k={best_silhouette_k}\n")
        f.write(f"Final choice: k={best_k} (using elbow method)\n")
    
    print(f"\nTraining best clustering model with k={best_k}...")
    clusterer, labels, method_name = train_best_clusterer(X_scaled, best_k)
    
    # Save the clusterer and assignments
    joblib.dump(clusterer, f"{OUTPUT_DIR}/best_clusterer.joblib")
    np.save(f"{OUTPUT_DIR}/cluster_assignments.npy", labels)
    pd.DataFrame({'cluster': labels}).to_csv(f"{OUTPUT_DIR}/cluster_assignments.csv", index=False)
    
    # Save cluster size visualization
    save_cluster_visualization(labels, clusterer, f"{OUTPUT_DIR}/cluster_sizes.png")
    
    print(f"\n✅ Clustering analysis complete. Best k={best_k} ({method_name}). Output saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
