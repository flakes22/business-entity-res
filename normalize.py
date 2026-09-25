import pandas as pd
import numpy as np
import re
import os
import unicodedata

# Define paths
TRAIN_DIR = "6ab10eb3b23ba_student_resource/student_resource/dataset/train/"
TEST_DIR = "6ab10eb3b23ba_student_resource/student_resource/dataset/test/"
OUTPUT_DIR = "6ab10eb3b23ba_student_resource/student_resource/dataset/normalized/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Standardize abbreviations (Add more as needed)
NAME_REPLACEMENTS = {
    r'\binc\b': 'incorporated',
    r'\bcorp\b': 'corporation',
    r'\bltd\b': 'limited',
    r'\bpvt\b': 'private',
    r'\bllc\b': 'limited liability company',
    r'\bco\b': 'company'
}

ADDRESS_REPLACEMENTS = {
    r'\bst\b': 'street',
    r'\brd\b': 'road',
    r'\bave\b': 'avenue',
    r'\bblvd\b': 'boulevard',
    r'\bdr\b': 'drive',
    r'\bln\b': 'lane',
    r'\bct\b': 'court',
    r'\bapt\b': 'apartment',
    r'\bste\b': 'suite',
    r'\bhwy\b': 'highway'
}

LEGAL_SUFFIXES_TO_REMOVE = r'\b(incorporated|corporation|limited|private|limited liability company|company)\b'

def clean_text(text, replacements):
    """
    Lowercases, normalizes unicode, handles '&', and standardizes.
    """
    if pd.isna(text):
        return ""
    
    # Lowercase
    text = str(text).lower()
    
    # Unicode normalization (e.g., handles French accents well without destroying Hindi characters)
    text = unicodedata.normalize('NFKC', text)
    
    # Explicitly handle '&' vs 'and' before removing punctuation
    text = text.replace('&', ' and ')
    
    # Apply standard abbreviation replacements
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
        
    # Remove punctuation except spaces (keep unicode characters for Hindi/French)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def remove_legal_suffixes(text):
    """Removes expanded legal suffixes from the normalized text."""
    text = re.sub(LEGAL_SUFFIXES_TO_REMOVE, '', text)
    return re.sub(r'\s+', ' ', text).strip()

def normalize_dataframe(df):
    """
    Normalizes the names and addresses in the dataframe, maintaining original representations.
    """
    print(f"Normalizing dataframe with {len(df)} rows...")
    
    # 1. Handle missing values (keeping original columns untouched mostly, just filling NaN)
    df['business_name'] = df['business_name'].fillna("")
    df['business_address'] = df['business_address'].fillna("")
    
    # 2. Clean Text (creates new columns)
    print("Cleaning business names...")
    df['norm_name'] = df['business_name'].apply(lambda x: clean_text(x, NAME_REPLACEMENTS))
    
    print("Creating name without legal suffixes...")
    df['norm_name_no_suffix'] = df['norm_name'].apply(remove_legal_suffixes)
    
    print("Cleaning business addresses...")
    df['norm_address'] = df['business_address'].apply(lambda x: clean_text(x, ADDRESS_REPLACEMENTS))
    
    # Note: Tokens and character representations will be handled dynamically during feature extraction
    # to save disk space in the TSV, but the original text is preserved!
    return df

def process_and_save(filename, source_dir):
    """Loads, normalizes, and saves the dataframe."""
    print(f"\n--- Processing {filename} ---")
    out_filepath = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(out_filepath):
        print(f"File {out_filepath} already exists. Skipping.")
        return
        
    filepath = os.path.join(source_dir, filename)
    
    # Load dataset
    df = pd.read_csv(filepath, sep="\t")
    
    # Normalize
    df_norm = normalize_dataframe(df)
    
    # Save to output directory
    df_norm.to_csv(out_filepath, sep="\t", index=False)
    print(f"Saved normalized data to {out_filepath}")

def main():
    print("Starting Normalization Process...")
    
    # Train files
    train_files = ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv"]
    for file in train_files:
        process_and_save(file, TRAIN_DIR)
        
    # Test files
    test_files = ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"]
    for file in test_files:
        process_and_save(file, TEST_DIR)

    print("\nNormalization Complete! All files saved in", OUTPUT_DIR)

if __name__ == "__main__":
    main()
