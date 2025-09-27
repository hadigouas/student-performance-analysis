"""
Student Performance Analysis Dashboard
A Streamlit application for exploring student performance data and model results.
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import sys
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)  # Add this for newer NLTK versions
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)  # Open Multilingual WordNet
except Exception as e:
    print(f"Warning: Could not download NLTK data: {e}")

from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(
    page_title="Student Performance Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DATA_DIR = Path("output")
MODELS_DIR = DATA_DIR / "models"
FEATURES_DIR = DATA_DIR / "features"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
CLUSTERING_DIR = DATA_DIR / "clustering"

# Load data and models
@st.cache_resource
def load_data():
    """Load the dataset and preprocessed data."""
    try:
        # Load original data
        df = pd.read_csv("StudentPerformanceFactors.csv")
        
        # Load feature names
        with open(FEATURES_DIR / "feature_names.txt", 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        # Load embeddings
        embeddings = np.load(EMBEDDINGS_DIR / "embeddings.npy")
        
        # Load cluster data
        cluster_data = {}
        for method in ['pca', 'tsne', 'umap']:
            try:
                cluster_data[method] = pd.read_csv(CLUSTERING_DIR / f"clusters_{method}.csv")
            except FileNotFoundError:
                pass
        
        return {
            'df': df,
            'feature_names': feature_names,
            'embeddings': embeddings,
            'cluster_data': cluster_data
        }
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_resource
def load_models():
    """Load trained models."""
    try:
        models = {}
        
        # Load regression model
        if (MODELS_DIR / "best_regressor.joblib").exists():
            models['regressor'] = joblib.load(MODELS_DIR / "best_regressor.joblib")
        
        # Load clustering model
        if (CLUSTERING_DIR / "best_clusterer.joblib").exists():
            models['clusterer'] = joblib.load(CLUSTERING_DIR / "best_clusterer.joblib")
        
        # Load SBERT model
        models['sbert'] = SentenceTransformer('all-MiniLM-L6-v2')
        
        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = load_data()
    st.session_state.models = load_models()
    st.session_state.current_school = None

def show_data_explorer():
    """Display the data explorer tab."""
    st.header("📊 Data Explorer")
    
    if st.session_state.data is None:
        st.warning("Data not loaded. Please check if the data files exist.")
        return
    
    df = st.session_state.data['df']
    
    # School filter
    school_col = next((col for col in df.columns if 'school' in col.lower()), None)
    if school_col:
        schools = ['All'] + sorted(df[school_col].dropna().unique().tolist())
        selected_school = st.sidebar.selectbox("Select School", schools, index=0)
        
        if selected_school != 'All':
            df = df[df[school_col] == selected_school]
            st.session_state.current_school = selected_school
    
    # Display data
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    
    # Basic stats
    st.subheader("Basic Statistics")
    st.write(df.describe())
    
    # Column selector for visualization
    col1, col2 = st.columns(2)
    
    with col1:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        selected_num_col = st.selectbox("Select Numeric Column", numeric_cols)
        
        if selected_num_col:
            fig = px.histogram(df, x=selected_num_col, title=f"Distribution of {selected_num_col}")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            selected_cat_col = st.selectbox("Select Categorical Column", cat_cols)
            
            if selected_cat_col:
                value_counts = df[selected_cat_col].value_counts().reset_index()
                value_counts.columns = [selected_cat_col, 'Count']
                
                fig = px.bar(
                    value_counts, 
                    x=selected_cat_col, 
                    y='Count',
                    title=f"Count of {selected_cat_col}"
                )
                st.plotly_chart(fig, use_container_width=True)

def show_text_analysis():
    """Display the text analysis tab."""
    st.header("📝 Text Analysis")
    
    with st.expander("View Text Preprocessing Steps"):
        st.markdown("""
        The text goes through the following preprocessing steps before being converted into a numerical vector:
        1. **Lowercase Conversion**: All text is converted to lowercase.
        2. **Cleaning**: Special characters and numbers are removed.
        3. **Tokenization**: Text is split into individual words (tokens).
        4. **Stopword Removal**: Common words (e.g., 'the', 'is') are removed.
        5. **Lemmatization**: Words are reduced to their base form (e.g., 'running' -> 'run').
        6. **Embedding**: The cleaned text is converted into a 384-dimensional vector using a pre-trained SBERT model.
        """)
    
    if st.session_state.data is None or st.session_state.models is None:
        st.warning("Data or models not loaded.")
        return
    
    model = st.session_state.models['sbert']
    
    st.subheader("Interactive Text Analysis")
    sample_text = st.text_area(
        "Enter student profile text to analyze:",
        "This student shows great potential in mathematics but needs improvement in time management.",
        height=150
    )
    
    if st.button("Analyze Text"):
        with st.spinner('Processing text...'):
            # Step 1: Basic text statistics
            st.subheader("📊 Text Statistics")
            words = sample_text.split()
            sentences = sample_text.split('.')
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Characters", len(sample_text))
            col2.metric("Words", len(words))
            col3.metric("Sentences", len([s for s in sentences if s.strip()]))
            
            # Step 2: Generate and analyze embeddings
            st.subheader("Semantic Analysis")
            embedding = model.encode([sample_text], normalize_embeddings=True)[0]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Embedding Statistics**")
                st.json({
                    "Vector Dimension": len(embedding),
                    "Vector Norm": f"{np.linalg.norm(embedding):.4f}",
                    "Min/Max": f"{np.min(embedding):.4f} / {np.max(embedding):.4f}",
                    "Mean": f"{np.mean(embedding):.4f}"
                })
            
            with col2:
                st.markdown("**Embedding Distribution**")
                fig_hist = px.histogram(x=embedding, nbins=50)
                st.plotly_chart(fig_hist, use_container_width=True)

def make_prediction(inputs, profile_text):
    """Helper function to make predictions with the given inputs and text."""
    # Get the expected feature names from the model
    if hasattr(st.session_state.models['regressor'], 'feature_names_in_'):
        expected_features = list(st.session_state.models['regressor'].feature_names_in_)
    elif 'feature_names' in st.session_state.data:
        expected_features = st.session_state.data['feature_names']
    else:
        raise ValueError("Could not determine expected feature names")
    
    # Create a dictionary of all possible features with default 0
    feature_dict = {feature: 0 for feature in expected_features}
    
    # Fill in the values from the form inputs
    for col, value in inputs.items():
        # Handle numeric features
        if col in feature_dict:
            feature_dict[col] = float(value)
        # Handle one-hot encoded categorical features
        else:
            # Check if this is a one-hot encoded column
            for feature in expected_features:
                if feature.startswith(f"{col}_"):
                    # Set to 1 if this is the selected category, 0 otherwise
                    feature_dict[feature] = 1 if feature == f"{col}_{value}" else 0
    
    # Convert to numpy array in the correct order
    input_features = np.array([[feature_dict[feature] for feature in expected_features]])
    
    # Process text if provided and the model was trained with text
    if profile_text.strip() and 'sbert' in st.session_state.models:
        text_embedding = st.session_state.models['sbert'].encode(
            [profile_text], 
            normalize_embeddings=True
        )[0]
        # Only add text features if the model was trained with them
        if len(expected_features) > input_features.shape[1]:
            input_features = np.hstack([input_features, text_embedding.reshape(1, -1)])
    
    # Make prediction
    return st.session_state.models['regressor'].predict(input_features)[0]

def show_prediction():
    """Display the prediction tab."""
    st.header("🔮 Predict Final Mark")
    
    if st.session_state.data is None or st.session_state.models is None:
        st.warning("Data or models not loaded. Please check if the required files exist.")
        return
    
    if 'regressor' not in st.session_state.models:
        st.error("Regression model not found. Please train the model first.")
        return
    
    df = st.session_state.data['df']
    
    # Create tabs for different prediction types
    tab1, tab2, tab3 = st.tabs(["📊 Numeric/Categorical", "📝 Text Only", "📈 Combined"])
    
    # Store inputs in session state to persist across tabs
    if 'inputs' not in st.session_state:
        st.session_state.inputs = {}
    if 'profile_text' not in st.session_state:
        st.session_state.profile_text = ""
    
    # Get numeric and categorical features
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Identify target column
    target_col = next((col for col in df.columns if 'final' in col.lower() or 'score' in col.lower()), None)
    
    # Remove target column from features
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # Tab 1: Numeric/Categorical Inputs Only
    with tab1:
        st.subheader("Numeric & Categorical Features")
        col1, col2 = st.columns(2)
        
        # Numeric inputs
        for i, col in enumerate(numeric_cols):
            col_widget = col1 if i % 2 == 0 else col2
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            default_val = float(df[col].median())
            
            # Load from session state or use default
            current_value = st.session_state.inputs.get(col, default_val)
            
            # Update in session state when changed
            new_value = col_widget.slider(
                col,
                min_value=min_val,
                max_value=max_val,
                value=current_value,
                step=0.1 if df[col].dtype == 'float64' else 1.0,
                key=f"num_{col}"
            )
            st.session_state.inputs[col] = new_value
        
        # Categorical inputs
        for i, col in enumerate(categorical_cols):
            if col != target_col:
                unique_vals = df[col].dropna().unique().tolist()
                if len(unique_vals) > 1:
                    default_val = df[col].mode()[0] if not df[col].empty else unique_vals[0] if unique_vals else None
                    current_value = st.session_state.inputs.get(col, default_val)
                    
                    selected = col1.selectbox(
                        col, 
                        unique_vals, 
                        index=unique_vals.index(current_value) if current_value in unique_vals else 0,
                        key=f"cat_{col}"
                    )
                    st.session_state.inputs[col] = selected
        
        # Predict button for numeric/categorical only
        if st.button("Predict from Features", key="predict_features"):
            try:
                prediction = make_prediction(st.session_state.inputs, "")
                st.success(f"Predicted Final Mark (Features Only): {prediction:.2f}")
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    
    # Tab 2: Text Input Only
    with tab2:
        st.subheader("Text Analysis")
        
        # Text input
        st.session_state.profile_text = st.text_area(
            "Enter student profile:",
            st.session_state.profile_text,
            key="text_input",
            height=200,
            placeholder="Enter student's profile, strengths, weaknesses, and any relevant information..."
        )
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if st.button("Predict from Text", key="predict_text", use_container_width=True):
                try:
                    if not st.session_state.profile_text.strip():
                        st.warning("Please enter some text to analyze.")
                    else:
                        with st.spinner('Analyzing text and extracting features...'):
                            # Get the SBERT model
                            sbert_model = st.session_state.models.get('sbert')
                            if not sbert_model:
                                raise ValueError("SBERT model not found in session state")
                            
                            # Generate SBERT embeddings
                            embedding = sbert_model.encode(
                                [st.session_state.profile_text],
                                normalize_embeddings=True,
                                show_progress_bar=False
                            )[0]
                            
                            # Store the embedding for display
                            st.session_state.last_embedding = embedding
                            
                            # Make prediction
                            prediction = make_prediction({}, st.session_state.profile_text)
                            
                            # Store the prediction in session state
                            if 'text_predictions' not in st.session_state:
                                st.session_state.text_predictions = []
                            st.session_state.text_predictions.append(prediction)
                            
                            # Display results
                            st.success(f"Predicted Final Mark: {prediction:.2f}")
                            
                            # Show the extracted features (first 20 dimensions)
                            with st.expander("🔍 View Extracted NLP Features", expanded=True):
                                st.write("The following features were extracted from the text using SBERT:")
                                
                                # Create a DataFrame for the first 20 dimensions
                                feature_df = pd.DataFrame({
                                    'Feature': [f'Dim_{i+1}' for i in range(20)],
                                    'Value': embedding[:20]
                                })
                                
                                # Display as a bar chart
                                fig = px.bar(
                                    feature_df,
                                    x='Feature',
                                    y='Value',
                                    title='First 20 SBERT Embedding Dimensions',
                                    labels={'Value': 'Feature Value', 'Feature': 'Dimension'}
                                )
                                fig.update_layout(xaxis_tickangle=-45)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Show statistics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Vector Dimension", len(embedding))
                                with col2:
                                    st.metric("Min Value", f"{embedding.min():.4f}")
                                with col3:
                                    st.metric("Max Value", f"{embedding.max():.4f}")
                                
                                st.caption(f"Total dimensions: {len(embedding)} (showing first 20)")
                
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
        
        with col2:
            if st.button("Reset Predictions", key="reset_text_predictions", type="secondary", use_container_width=True):
                if 'text_predictions' in st.session_state:
                    st.session_state.text_predictions = []
                st.info("Prediction history has been reset.")
        
        # Show prediction history if available
        if 'text_predictions' in st.session_state and st.session_state.text_predictions:
            with st.expander("📊 View Prediction History"):
                history_df = pd.DataFrame({
                    'Prediction #': range(1, len(st.session_state.text_predictions) + 1),
                    'Value': st.session_state.text_predictions
                })
                st.dataframe(
                    history_df,
                    column_config={
                        'Prediction #': st.column_config.NumberColumn(format='%d'),
                        'Value': st.column_config.NumberColumn(format='%.2f')
                    },
                    hide_index=True,
                    use_container_width=True
                )
    
    # Tab 3: Combined Inputs
    with tab3:
        st.subheader("Combined Prediction")
        st.info("Uses both feature inputs and text analysis for prediction")
        
        # Show summary of current inputs
        with st.expander("🔍 Current Inputs"):
            st.write("**Numeric/Categorical Features:**")
            if st.session_state.inputs:
                st.json({k: v for k, v in st.session_state.inputs.items() if not isinstance(v, str) or not v.startswith("cat_")})
            else:
                st.write("No feature inputs yet.")
            
            st.write("\n**Text Input:**")
            st.text(st.session_state.profile_text if st.session_state.profile_text else "No text provided.")
        
        if st.button("Predict with Combined Inputs", key="predict_combined"):
            try:
                prediction = make_prediction(st.session_state.inputs, st.session_state.profile_text)
                st.success(f"Predicted Final Mark (Combined): {prediction:.2f}")
                
                # Show feature importance if available
                if hasattr(st.session_state.models['regressor'], 'feature_importances_'):
                    with st.expander("View Feature Importance"):
                        feature_importance = pd.DataFrame({
                            'Feature': st.session_state.models['regressor'].feature_names_in_,
                            'Importance': st.session_state.models['regressor'].feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        fig = px.bar(
                            feature_importance.head(10),  # Show top 10 features
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title='Top 10 Most Important Features'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.warning("Please make sure all required features are provided and models are loaded correctly.")
                st.info(f"If the error persists, please check the console for more details.")

def show_cluster_analysis():
    """Display the cluster analysis tab."""
    st.header("🔍 Cluster Analysis")
    
    # Load original data and cluster assignments
    try:
        df_original = pd.read_csv("StudentPerformanceFactors.csv")
        cluster_assignments = pd.read_csv("output/clustering/cluster_assignments.csv")
        
        # Combine original data with cluster assignments
        df_with_clusters = df_original.copy()
        df_with_clusters['cluster'] = cluster_assignments['cluster']
        
    except FileNotFoundError as e:
        st.error(f"Required files not found: {e}")
        st.info("Please run the clustering script (04_clustering_and_validation.py) first.")
        return
    
    # Show cluster visualization if projection data exists
    if st.session_state.data and 'cluster_data' in st.session_state.data:
        cluster_data = st.session_state.data['cluster_data']
        
        if cluster_data:
            # Projection selector
            projection_method = st.sidebar.selectbox(
                "Select Projection Method",
                list(cluster_data.keys())
            )
            
            # Get data for selected projection
            df_proj = cluster_data[projection_method]
            
            # Create cluster plot
            fig = px.scatter(
                df_proj,
                x='x',
                y='y',
                color='cluster',
                title=f"Student Clusters ({projection_method.upper()})",
                labels={'x': 'Dimension 1', 'y': 'Dimension 2'},
                hover_data=['cluster']
            )
            
            # Update layout
            fig.update_traces(
                marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')),
                selector=dict(mode='markers')
            )
            
            # Show plot
            st.plotly_chart(fig, use_container_width=True)
    
    # Cluster statistics and features analysis
    st.subheader("📊 Cluster Statistics")
    cluster_stats = df_with_clusters['cluster'].value_counts().reset_index()
    cluster_stats.columns = ['Cluster', 'Count']
    cluster_stats = cluster_stats.sort_values('Cluster')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(cluster_stats, use_container_width=True)
    
    with col2:
        fig_pie = px.pie(
            cluster_stats,
            values='Count',
            names='Cluster',
            title='Cluster Distribution'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Detailed cluster analysis
    st.subheader("🎯 Common Features in Each Cluster")
    
    # Get numeric and categorical columns
    numeric_cols = df_with_clusters.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df_with_clusters.select_dtypes(include=['object']).columns.tolist()
    
    # Remove cluster column from analysis
    if 'cluster' in numeric_cols:
        numeric_cols.remove('cluster')
    if 'cluster' in categorical_cols:
        categorical_cols.remove('cluster')
    
    # Calculate statistics for each cluster
    for cluster_id in sorted(df_with_clusters['cluster'].unique()):
        cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
        cluster_size = len(cluster_data)
        
        with st.expander(f"🔍 Cluster {cluster_id} - {cluster_size} students ({cluster_size/len(df_with_clusters)*100:.1f}%)"):
            
            # Create tabs for different types of analysis
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Key Metrics", "📈 Numeric Features", "📋 Categorical Features", "🎯 Performance Profile"])
            
            with tab1:
                st.markdown("### 🏆 Key Performance Indicators")
                
                # Key metrics
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    avg_exam = cluster_data['Exam_Score'].mean()
                    overall_avg_exam = df_with_clusters['Exam_Score'].mean()
                    st.metric(
                        "Average Exam Score", 
                        f"{avg_exam:.1f}",
                        f"{avg_exam - overall_avg_exam:+.1f}"
                    )
                
                with col_b:
                    avg_attendance = cluster_data['Attendance'].mean()
                    overall_avg_attendance = df_with_clusters['Attendance'].mean()
                    st.metric(
                        "Average Attendance", 
                        f"{avg_attendance:.1f}%",
                        f"{avg_attendance - overall_avg_attendance:+.1f}%"
                    )
                
                with col_c:
                    avg_hours = cluster_data['Hours_Studied'].mean()
                    overall_avg_hours = df_with_clusters['Hours_Studied'].mean()
                    st.metric(
                        "Hours Studied", 
                        f"{avg_hours:.1f}h",
                        f"{avg_hours - overall_avg_hours:+.1f}h"
                    )
            
            with tab2:
                st.markdown("### 📊 Numeric Feature Analysis")
                
                # Calculate cluster means vs overall means
                cluster_means = cluster_data[numeric_cols].mean()
                overall_means = df_with_clusters[numeric_cols].mean()
                feature_comparison = pd.DataFrame({
                    'Feature': numeric_cols,
                    'Cluster_Mean': cluster_means.values,
                    'Overall_Mean': overall_means.values,
                    'Difference': (cluster_means - overall_means).values,
                    'Abs_Difference': (cluster_means - overall_means).abs().values
                })
                
                # Sort by absolute difference to show most distinguishing features
                feature_comparison = feature_comparison.sort_values('Abs_Difference', ascending=False)
                
                # Show top distinguishing features
                st.markdown("**🎯 Most Distinguishing Features (Top 5):**")
                top_features = feature_comparison.head(5)
                
                for _, row in top_features.iterrows():
                    direction = "↑" if row['Difference'] > 0 else "↓"
                    color = "🟢" if row['Difference'] > 0 else "🔴"
                    st.write(f"{color} **{row['Feature']}**: {row['Cluster_Mean']:.2f} vs {row['Overall_Mean']:.2f} (diff: {direction}{abs(row['Difference']):.2f})")
                
                # Show all numeric features in a table
                st.markdown("**📋 All Numeric Features:**")
                display_df = feature_comparison[['Feature', 'Cluster_Mean', 'Overall_Mean', 'Difference']].copy()
                display_df['Cluster_Mean'] = display_df['Cluster_Mean'].round(2)
                display_df['Overall_Mean'] = display_df['Overall_Mean'].round(2)
                display_df['Difference'] = display_df['Difference'].round(2)
                st.dataframe(display_df, use_container_width=True)
            
            with tab3:
                st.markdown("### 📋 Categorical Feature Analysis")
                
                # Analyze categorical features
                for cat_col in categorical_cols:
                    st.markdown(f"**{cat_col}:**")
                    
                    # Get value counts for this cluster
                    cluster_counts = cluster_data[cat_col].value_counts()
                    cluster_percentages = (cluster_counts / len(cluster_data) * 100).round(1)
                    
                    # Get overall percentages
                    overall_counts = df_with_clusters[cat_col].value_counts()
                    overall_percentages = (overall_counts / len(df_with_clusters) * 100).round(1)
                    
                    # Create comparison dataframe
                    comparison_data = []
                    for value in cluster_counts.index:
                        cluster_pct = cluster_percentages.get(value, 0)
                        overall_pct = overall_percentages.get(value, 0)
                        diff = cluster_pct - overall_pct
                        comparison_data.append({
                            'Value': value,
                            'Cluster %': cluster_pct,
                            'Overall %': overall_pct,
                            'Difference': diff
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    comparison_df = comparison_df.sort_values('Difference', ascending=False)
                    
                    # Show the comparison
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Highlight most common values in this cluster
                    most_common = comparison_df.iloc[0]
                    if most_common['Difference'] > 10:  # If significantly higher than overall
                        st.success(f"👆 **Most characteristic**: {most_common['Value']} ({most_common['Cluster %']:.1f}% vs {most_common['Overall %']:.1f}% overall)")
                    
                    st.markdown("---")
            
            with tab4:
                st.markdown("### 🎯 Performance Profile Summary")
                
                # Create a performance profile
                performance_metrics = {
                    'Academic Performance': {
                        'Exam_Score': cluster_data['Exam_Score'].mean(),
                        'Previous_Scores': cluster_data['Previous_Scores'].mean(),
                        'Hours_Studied': cluster_data['Hours_Studied'].mean()
                    },
                    'Engagement': {
                        'Attendance': cluster_data['Attendance'].mean(),
                        'Motivation_Level': cluster_data['Motivation_Level'].map({'Low': 1, 'Medium': 2, 'High': 3}).mean() if 'Motivation_Level' in cluster_data.columns else 0,
                        'Extracurricular_Activities': (cluster_data['Extracurricular_Activities'] == 'Yes').mean() * 100 if 'Extracurricular_Activities' in cluster_data.columns else 0
                    },
                    'Support & Resources': {
                        'Parental_Involvement': cluster_data['Parental_Involvement'].map({'Low': 1, 'Medium': 2, 'High': 3}).mean() if 'Parental_Involvement' in cluster_data.columns else 0,
                        'Access_to_Resources': cluster_data['Access_to_Resources'].map({'Low': 1, 'Medium': 2, 'High': 3}).mean() if 'Access_to_Resources' in cluster_data.columns else 0,
                        'Tutoring_Sessions': cluster_data['Tutoring_Sessions'].mean()
                    }
                }
                
                # Create performance summary
                st.markdown("**🎯 Cluster Characteristics:**")
                
                # Determine cluster type based on exam scores
                avg_score = cluster_data['Exam_Score'].mean()
                overall_avg_score = df_with_clusters['Exam_Score'].mean()
                
                if avg_score > overall_avg_score + 5:
                    cluster_type = "🏆 High Performers"
                    color = "success"
                elif avg_score < overall_avg_score - 5:
                    cluster_type = "🔄 Needs Support"
                    color = "warning"
                else:
                    cluster_type = "📈 Average Performers"
                    color = "info"
                
                st.markdown(f"**Cluster Type:** {cluster_type}")
                
                # Show key characteristics
                st.markdown("**Key Characteristics:**")
                
                # Find top 3 distinguishing features
                feature_comparison = pd.DataFrame({
                    'Feature': numeric_cols,
                    'Difference': (cluster_data[numeric_cols].mean() - df_with_clusters[numeric_cols].mean()).abs().values
                })
                top_3_features = feature_comparison.nlargest(3, 'Difference')['Feature'].tolist()
                
                for feature in top_3_features:
                    cluster_val = cluster_data[feature].mean()
                    overall_val = df_with_clusters[feature].mean()
                    if cluster_val > overall_val:
                        st.write(f"✅ Higher {feature}: {cluster_val:.1f} (vs {overall_val:.1f} overall)")
                    else:
                        st.write(f"⚠️ Lower {feature}: {cluster_val:.1f} (vs {overall_val:.1f} overall)")
                
                # Recommendations based on cluster characteristics
                st.markdown("**💡 Potential Interventions:**")
                if avg_score < overall_avg_score - 5:
                    st.write("• Focus on foundational skills strengthening")
                    st.write("• Increase tutoring and support services")
                    st.write("• Improve study habits and time management")
                elif cluster_data['Attendance'].mean() < 80:
                    st.write("• Address attendance issues")
                    st.write("• Investigate barriers to school attendance")
                elif cluster_data['Hours_Studied'].mean() < df_with_clusters['Hours_Studied'].mean():
                    st.write("• Encourage more dedicated study time")
                    st.write("• Provide study skills training")
                else:
                    st.write("• Continue current successful practices")
                    st.write("• Consider advanced or enrichment opportunities")

def main():
    """Main function to run the Streamlit app."""
    st.title("🎓 Student Performance Analysis Dashboard")
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Go to",
        ["Data Explorer", "Text Analysis", "Predict Final Mark", "Cluster Analysis"]
    )
    
    # Display selected page
    if app_mode == "Data Explorer":
        show_data_explorer()
    elif app_mode == "Text Analysis":
        show_text_analysis()
    elif app_mode == "Predict Final Mark":
        show_prediction()
    elif app_mode == "Cluster Analysis":
        show_cluster_analysis()
    
    # Add footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This dashboard provides insights into student performance data. "
        "Use the navigation menu to explore different aspects of the data."
    )

if __name__ == "__main__":
    main()
