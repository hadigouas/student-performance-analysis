"""
Utility functions for the Student Performance Analysis project.
"""

import re
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any, Union
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt', quiet=True)
# nltk.download('stopwords', quiet=True)
# nltk.download('wordnet', quiet=True)

# Hardcoded English stopwords for simplicity
STOP_WORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it',
    "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
    'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
    'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've",
    'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
    "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
    "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't",
    'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn',
    "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn',
    "wouldn't"
])

def detect_text_column(df: pd.DataFrame) -> Optional[str]:
    """
    Detect the most likely text column in the dataframe.
    
    Args:
        df: Input pandas DataFrame
        
    Returns:
        str: Name of the detected text column, or None if none found
    """
    # Check for common text column names
    text_like_columns = [
        col for col in df.columns 
        if any(keyword in col.lower() for keyword in ['text', 'profile', 'note', 'description', 'comment'])
    ]
    
    if text_like_columns:
        return text_like_columns[0]
    
    # If no obvious text columns, look for columns with mostly string data
    text_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if the column contains mostly text (more than 50% of values have spaces)
            text_ratio = (df[col].dropna().astype(str).str.contains(r'\s').mean())
            if text_ratio > 0.5:
                text_cols.append((col, text_ratio))
    
    if text_cols:
        # Return the column with highest text ratio
        return max(text_cols, key=lambda x: x[1])[0]
    
    return None

def simple_clean_text(text: str, stop_words: set = None) -> str:
    """
    Clean and preprocess a text string.
    
    Args:
        text: Input text to clean
        stop_words: Set of stopwords to remove (default: None, uses built-in stopwords)
        
    Returns:
        str: Cleaned and preprocessed text
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Use provided stopwords or default
    stop_words = stop_words or STOP_WORDS
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers (keep only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and short tokens
    tokens = [
        token for token in tokens 
        if token not in stop_words and len(token) > 2
    ]
    
    # Lemmatize (simple version)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

def compute_adjusted_r2(r2: float, n: int, p: int) -> float:
    """
    Calculate adjusted R-squared.
    
    Args:
        r2: R-squared value
        n: Number of samples
        p: Number of features
        
    Returns:
        float: Adjusted R-squared value
    """
    if n - p - 1 <= 0:
        return float('nan')
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def plot_distribution(data: pd.Series, title: str = None, xlabel: str = None, **kwargs) -> None:
    """
    Plot distribution of a numeric column.
    
    Args:
        data: Input data (pandas Series)
        title: Plot title
        xlabel: X-axis label
        **kwargs: Additional arguments to pass to seaborn's histplot
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 5))
    sns.histplot(data=data, kde=True, **kwargs)
    
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    
    plt.tight_layout()
    plt.show()

def plot_boxplot(data: pd.DataFrame, x: str, y: str, title: str = None, **kwargs) -> None:
    """
    Plot a boxplot.
    
    Args:
        data: Input DataFrame
        x: Column name for x-axis
        y: Column name for y-axis
        title: Plot title
        **kwargs: Additional arguments to pass to seaborn's boxplot
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=data, x=x, y=y, **kwargs)
    
    if title:
        plt.title(title)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def detect_target_column(df: pd.DataFrame) -> Optional[str]:
    """
    Detect the target column in the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        str: Name of the target column, or None if not found
    """
    # Look for common target column names
    target_candidates = [
        col for col in df.columns 
        if any(keyword in col.lower() for keyword in ['final', 'score', 'mark', 'grade'])
    ]
    
    # Prioritize numeric columns
    numeric_targets = [col for col in target_candidates if df[col].dtype in ['int64', 'float64']]
    
    if numeric_targets:
        return numeric_targets[0]
    elif target_candidates:
        return target_candidates[0]
    
    # If no obvious target found, try to find a numeric column that could be a target
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if len(numeric_cols) == 1:
        return numeric_cols[0]
    
    return None
