"""
Feature Fusion and Data Splitting
This script processes the dataset, combines different feature types, and splits into train/test sets.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Constants
RANDOM_STATE = 42
TEST_SIZE = 0.2
OUTPUT_DIR = 'output/features'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def detect_target_column(df):
    """Detect and return the target column name."""
    target_candidates = [col for col in df.columns if 'final' in col.lower() or 'score' in col.lower()or 'exam_score' in col.lower()]
    if target_candidates:
        return target_candidates[0]
    return None

def load_and_prepare_data(csv_path, embeddings_path):
    """Load CSV and embeddings, then prepare features and target."""
    # Load data
    df = pd.read_csv(csv_path)
    
    # Load embeddings
    if os.path.exists(embeddings_path):
        embeddings = np.load(embeddings_path)
    else:
        raise FileNotFoundError(f"Embeddings not found at {embeddings_path}")
    
    return df, embeddings

def get_feature_columns(df, target_col):
    """Identify numeric and categorical feature columns."""
    # Exclude target and potential text columns
    exclude_cols = [target_col, 'combined_text'] + [
        col for col in df.columns 
        if col.lower().endswith(('text', 'description', 'notes', 'name', 'id'))
    ]
    
    # Get numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
    
    return numeric_cols, categorical_cols

def create_preprocessor(numeric_cols, categorical_cols):
    """Create a column transformer for preprocessing."""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'  # Drop other columns
    )
    
    return preprocessor

def save_feature_names(preprocessor, numeric_cols, categorical_cols, output_dir):
    """Save feature names after one-hot encoding using a simplified approach."""
    # Start with numeric columns
    feature_names = numeric_cols.copy()
    
    # Generate simple categorical feature names
    if categorical_cols:  # Only proceed if there are categorical columns
        try:
            # Get the one-hot encoder
            ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
            
            # Check if we can access categories directly
            if hasattr(ohe, 'categories_'):
                for i, col in enumerate(categorical_cols):
                    if i < len(ohe.categories_):
                        # Get unique values for this column (up to 10 to avoid explosion)
                        unique_vals = ohe.categories_[i][:10]
                        for val in unique_vals:
                            feature_names.append(f"{col}_{val}")
            else:
                # Fallback: Just use column names with _1, _2, etc.
                for col in categorical_cols:
                    feature_names.append(f"{col}_category")
        except Exception as e:
            print(f"Warning: Could not generate categorical feature names: {str(e)}")
            # Fallback: Just use column names with _1, _2, etc.
            for col in categorical_cols:
                feature_names.append(f"{col}_category")
    
    # Add embedding dimensions
    embedding_dims = 384  # all-MiniLM-L6-v2 has 384 dimensions
    feature_names.extend([f"sbert_{i}" for i in range(embedding_dims)])
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to file
    output_file = os.path.join(output_dir, "feature_names.txt")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for name in feature_names:
                f.write(f"{name}\n")
        print(f"Saved feature names to {output_file}")
        return feature_names
    except Exception as e:
        print(f"Warning: Could not save feature names to {output_file}: {str(e)}")
        return []

def main():
    """Main function to run the feature fusion and data splitting process."""
    # File paths
    csv_path = 'StudentPerformanceFactors.csv'
    embeddings_path = 'output/embeddings/embeddings.npy'
    
    # Load and prepare data
    print("Loading data and embeddings...")
    df, embeddings = load_and_prepare_data(csv_path, embeddings_path)
    
    # Detect target column
    target_col = detect_target_column(df)
    if target_col is None:
        raise ValueError("Could not detect target column. Please specify it manually.")
    print(f"Using '{target_col}' as the target variable.")
    
    # Get feature columns
    numeric_cols, categorical_cols = get_feature_columns(df, target_col)
    
    # Create and fit preprocessor
    print("Preprocessing tabular features...")
    preprocessor = create_preprocessor(numeric_cols, categorical_cols)
    X_processed = preprocessor.fit_transform(df)
    
    # Combine with embeddings
    print("Combining features with text embeddings...")
    X_fused = np.hstack([X_processed, embeddings])
    y = df[target_col].values
    
    # Save feature names
    _ = save_feature_names(preprocessor, numeric_cols, categorical_cols, OUTPUT_DIR)
    
    # Split data
    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_fused, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # Save processed data
    print("Saving processed data...")
    np.save(f"{OUTPUT_DIR}/X_fused.npy", X_fused)
    np.save(f"{OUTPUT_DIR}/y.npy", y)
    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)
    
    # Save data summary
    summary = {
        'num_samples': len(X_fused),
        'num_features': X_fused.shape[1],
        'num_numeric_features': len(numeric_cols),
        'num_categorical_features': len(categorical_cols),
        'num_embedding_features': embeddings.shape[1],
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'target_column': target_col
    }
    
    with open(f"{OUTPUT_DIR}/data_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Feature fusion and data splitting complete. Output saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
