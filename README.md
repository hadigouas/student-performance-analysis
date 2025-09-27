# Student Performance Analysis and Prediction

This project provides a comprehensive framework for analyzing student performance based on a variety of factors, including demographics, study habits, and qualitative text profiles. It uses a machine learning pipeline to perform Exploratory Data Analysis (EDA), feature engineering, text embedding, regression modeling, and clustering. An interactive web-based GUI built with Streamlit allows for easy exploration and prediction.

## 🚀 Features

- **Automated EDA**: Generates summaries and visualizations of the dataset.
- **Hybrid Feature Engineering**: Combines tabular data (numeric/categorical) with NLP-based text features using SBERT embeddings.
- **Regression Modeling**: Trains multiple models (Linear Regression, Decision Tree, Random Forest) to predict final exam scores and selects the best one.
- **Clustering Analysis**: Uses the Elbow Method with KMeans to automatically determine the optimal number of student clusters and identify their key characteristics.
- **Interactive Dashboard**: A Streamlit application for:
  - Exploring the dataset.
  - Analyzing text profiles.
  - Predicting student performance using different combinations of features.
  - Visualizing and understanding student clusters.

## 📂 Project Structure

The project is organized into a sequential pipeline of Python scripts, where each script performs a specific task and saves its output to be used by the next script.

```
.
├── env/                  # Python virtual environment
├── output/               # All generated files (plots, models, features)
│   ├── clustering/       # Cluster models, assignments, and analysis reports
│   ├── eda_plots/        # Plots from Exploratory Data Analysis
│   ├── embeddings/       # Processed text and SBERT embeddings
│   ├── features/         # Fused features and train/test splits
│   └── models/           # Trained regression models
├── 00_load_and_eda.py
├── 01_text_preprocessing_and_embeddings.py
├── 02_feature_fusion_and_split.py
├── 03_regression_models.py
├── 04_clustering_and_validation.py
├── 05_gui_app.py
├── analyze_clusters.py   # Standalone script for cluster analysis
├── requirements.txt      # Project dependencies
└── StudentPerformanceFactors.csv # The raw dataset
```

### Pipeline Scripts

1.  **`00_load_and_eda.py`**: Loads the raw data and performs initial exploratory analysis.
2.  **`01_text_preprocessing_and_embeddings.py`**: Cleans text data and generates sentence embeddings using SBERT.
3.  **`02_feature_fusion_and_split.py`**: Merges tabular features with text embeddings and splits the data into training and testing sets.
4.  **`03_regression_models.py`**: Trains and evaluates regression models to predict student scores.
5.  **`04_clustering_and_validation.py`**: Performs clustering on the fused data to group students.
6.  **`05_gui_app.py`**: Launches the Streamlit dashboard for interactive analysis.
7.  **`analyze_clusters.py`**: A utility script to run an in-depth analysis of the generated clusters.

## 🛠️ Setup and Installation

Follow these steps to set up and run the project locally.

### 1. Prerequisites

- Python 3.9 or higher
- `pip` for package management

### 2. Clone the Repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### 3. Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

**On Windows:**

```bash
python -m venv env
.\env\Scripts\activate
```

**On macOS/Linux:**

```bash
python3 -m venv env
source env/bin/activate
```

### 4. Install Dependencies

Install all required packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## ⚙️ How to Run

You can either run the entire pipeline sequentially or run individual scripts.

### 1. Run the Full Pipeline

To run all data processing, model training, and clustering steps in order, execute the following commands:

```bash
python 00_load_and_eda.py
python 01_text_preprocessing_and_embeddings.py
python 02_feature_fusion_and_split.py
python 03_regression_models.py
python 04_clustering_and_validation.py
```

This will populate the `output/` directory with all the necessary artifacts.

### 2. Launch the Interactive Dashboard

After running the pipeline, you can start the Streamlit application to explore the results.

```bash
streamlit run 05_gui_app.py
```

This will open the dashboard in your default web browser.

### 3. (Optional) Run Standalone Cluster Analysis

If you want to regenerate the cluster analysis reports without running the full pipeline again, you can use the `analyze_clusters.py` script.

```bash
python analyze_clusters.py
```
This will update the analysis files in `output/clustering/`.
