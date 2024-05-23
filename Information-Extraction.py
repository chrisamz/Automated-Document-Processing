# src/information_extraction.py

import os
import re
import pandas as pd
import spacy

# Define file paths
ocr_data_path = 'data/processed/ocr_results.csv'
extracted_info_path = 'data/processed/extracted_information.csv'

# Create directories if they don't exist
os.makedirs(os.path.dirname(extracted_info_path), exist_ok=True)

# Load OCR processed data
print("Loading OCR processed data...")
df = pd.read_csv(ocr_data_path)

# Display the first few rows of the dataset
print("OCR Data:")
print(df.head())

# Load Spacy model for Named Entity Recognition
print("Loading Spacy model...")
nlp = spacy.load('en_core_web_sm')

# Function to extract dates using regular expressions
def extract_dates(text):
    date_pattern = r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{2,4}[-/]\d{1,2}[-/]\d{1,2})\b'
    dates = re.findall(date_pattern, text)
    return dates

# Function to extract amounts using regular expressions
def extract_amounts(text):
    amount_pattern = r'\b(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\b'
    amounts = re.findall(amount_pattern, text)
    return amounts

# Function to extract named entities using Spacy
def extract_named_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Perform information extraction
print("Performing information extraction...")
extracted_info = []

for index, row in df.iterrows():
    text = row['text']
    dates = extract_dates(text)
    amounts = extract_amounts(text)
    entities = extract_named_entities(text)
    
    extracted_info.append({
        'filename': row['filename'],
        'dates': dates,
        'amounts': amounts,
        'entities': entities
    })

# Create a DataFrame from the extracted information
extracted_df = pd.DataFrame(extracted_info)

# Save the extracted information
print("Saving extracted information...")
extracted_df.to_csv(extracted_info_path, index=False)

print(f"Information extraction completed! Results saved to {extracted_info_path}")
