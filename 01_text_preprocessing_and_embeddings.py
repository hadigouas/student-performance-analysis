"""
Text Preprocessing and Embeddings Generation
This script processes text data from the student performance dataset and generates embeddings using SBERT.
"""

import os
import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Set
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer

def download_nltk_data():
    """Download required NLTK data with error handling."""
    try:
        nltk_data = {
            'punkt': 'tokenizers/punkt',
            'stopwords': 'corpora/stopwords',
            'wordnet': 'corpora/wordnet',
            'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
            'omw-1.4': 'corpora/omw-1.4'  # Open Multilingual WordNet
        }
        
        for resource, path in nltk_data.items():
            try:
                nltk.data.find(path)
            except LookupError:
                print(f"Downloading NLTK {resource}...")
                nltk.download(resource, quiet=True)
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
        raise

# Download required NLTK data
download_nltk_data()

# Constants
MODEL_NAME = 'all-MiniLM-L6-v2'
BATCH_SIZE = 32
OUTPUT_DIR = 'output/embeddings'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_stop_words() -> Set[str]:
    """Get English stop words from NLTK with additional common words."""
    try:
        # Get NLTK's English stopwords
        stop_words = set(stopwords.words('english'))
        
        # Add common contractions and other frequent terms that might appear in student text
        additional_stopwords = {
            'also', 'could', 'would', 'should', 'might', 'must', 'shall',
            'may', 'need', 'use', 'like', 'get', 'go', 'one', 'two', 'three',
            'first', 'second', 'third', 'many', 'much', 'well', 'even', 'still',
            'without', 'since', 'though', 'whether', 'another', 'however',
            'therefore', 'thus', 'hence', 'meanwhile', 'furthermore', 'moreover',
            'nevertheless', 'nonetheless', 'accordingly', 'consequently',
            'otherwise', 'similarly', 'thereby', 'whereas', 'yet', 'thus'
        }
        
        return stop_words.union(additional_stopwords)
    except Exception as e:
        print(f"Warning: Could not load stop words: {e}")
        return set()  # Return empty set if stopwords can't be loaded

class TextPreprocessor:
    """Handles text preprocessing including cleaning, tokenization, and lemmatization."""
    
    def __init__(self, stop_words: Set[str] = None):
        """Initialize the text preprocessor.
        
        Args:
            stop_words: Optional set of stop words to use. If None, will use default NLTK stop words.
        """
        self.stop_words = stop_words or get_stop_words()
        try:
            self.lemmatizer = WordNetLemmatizer()
        except Exception as e:
            print(f"Error initializing lemmatizer: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess a single text string.
        
        Args:
            text: Input text to process
            
        Returns:
            Processed text string
        """
        if not isinstance(text, str):
            return ""
            
        # Step 1: Convert to lowercase
        text_lower = text.lower()
        
        # Step 2: Remove special characters and numbers
        text_clean = re.sub(r'[^a-zA-Z\s]', ' ', text_lower)
        
        # Step 3: Tokenization
        tokens = word_tokenize(text_clean)
        
        # Step 4: Remove stopwords and short tokens
        filtered_tokens = [token for token in tokens 
                         if token not in self.stop_words and len(token) > 1]
        
        # Step 5: Lemmatization
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in filtered_tokens]
        
        # Join tokens back to string
        processed_text = ' '.join(lemmatized_tokens)
        
        return processed_text

def detect_text_column(df: pd.DataFrame) -> str:
    """Detect the most likely text column in the dataframe."""
    text_columns = []
    text_like_columns = []
    
    for col in df.columns:
        # Check if column is text-like
        if df[col].dtype == 'object' and df[col].str.contains('[a-zA-Z]').any():
            text_columns.append(col)
            
            # Check for common text column names
            if any(keyword in col.lower() for keyword in ['text', 'profile', 'note', 'description', 'comment']):
                text_like_columns.append(col)
    
    # Return the most likely text column
    if text_like_columns:
        return text_like_columns[0]
    elif text_columns:
        return text_columns[0]
    else:
        # If no text column found, create one by concatenating all string columns
        string_cols = [col for col in df.columns if df[col].dtype == 'object']
        if string_cols:
            df['combined_text'] = df[string_cols].astype(str).agg(' '.join, axis=1)
            return 'combined_text'
        else:
            raise ValueError("No suitable text column found in the dataset.")

def save_embeddings_stats(embeddings: np.ndarray, output_file: str) -> None:
    """Save basic statistics about the embeddings."""
    stats = {
        'num_embeddings': embeddings.shape[0],
        'embedding_dimension': embeddings.shape[1],
        'mean_norm': float(np.mean(np.linalg.norm(embeddings, axis=1))),
        'min_norm': float(np.min(np.linalg.norm(embeddings, axis=1))),
        'max_norm': float(np.max(np.linalg.norm(embeddings, axis=1))),
        'mean_values': [float(x) for x in np.mean(embeddings, axis=0)[:5]],  # First 5 dimensions
        'std_values': [float(x) for x in np.std(embeddings, axis=0)[:5]]     # First 5 dimensions
    }
    
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)

def main():
    """Main function to run the text preprocessing and embedding generation."""
    # Load the dataset
    input_file = 'StudentPerformanceFactors.csv'
    print(f"Loading dataset from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Detect text column
    text_column = detect_text_column(df)
    print(f"Detected text column: '{text_column}'")
    
    # Initialize preprocessor
    print("Initializing text preprocessor...")
    preprocessor = TextPreprocessor()
    
    # Process all text
    print("Processing all texts...")
    tqdm.pandas(desc="Text Cleaning")
    df['cleaned_text'] = df[text_column].progress_apply(preprocessor.clean_text)
    
    # Save original and cleaned text
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    text_df = df[[text_column, 'cleaned_text']].copy()
    text_df.columns = ['original_profile_text', 'cleaned_profile_text']
    text_df.to_csv(f'{OUTPUT_DIR}/text_data.csv', index=False)
    
    # Generate embeddings
    print(f"Loading SBERT model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    print("Generating embeddings...")
    texts = df['cleaned_text'].tolist()
    
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # Save embeddings and stats
    print("Saving embeddings and statistics...")
    np.save(f'{OUTPUT_DIR}/embeddings.npy', embeddings)
    save_embeddings_stats(embeddings, f'{OUTPUT_DIR}/embeddings_stats.json')
    
    print(f"\n✅ Text processing and embedding generation complete. Output saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
