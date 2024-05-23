# src/data_visualization.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define file paths
ocr_data_path = 'data/processed/ocr_results.csv'
classification_results_path = 'data/processed/classification_evaluation_results.txt'
extracted_info_path = 'data/processed/extracted_information.csv'
figures_path = 'figures'

# Create directories if they don't exist
os.makedirs(figures_path, exist_ok=True)

# Load OCR processed data
print("Loading OCR processed data...")
ocr_df = pd.read_csv(ocr_data_path)

# Load classification results
print("Loading classification results...")
with open(classification_results_path, 'r') as file:
    classification_report = file.read()

# Load extracted information
print("Loading extracted information...")
extracted_df = pd.read_csv(extracted_info_path)

# Visualize OCR text length distribution
print("Visualizing OCR text length distribution...")
ocr_df['text_length'] = ocr_df['text'].apply(len)
plt.figure(figsize=(10, 6))
sns.histplot(ocr_df['text_length'], bins=50, kde=True, color='blue')
plt.title('Distribution of OCR Text Lengths')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.savefig(os.path.join(figures_path, 'ocr_text_length_distribution.png'))
plt.show()

# Visualize document classification results
print("Visualizing document classification results...")
# This is a placeholder visualization. Adjust it according to your classification evaluation results.
classification_metrics = {
    'Accuracy': float(classification_report.split("\n")[-2].split()[-1])
}

plt.figure(figsize=(10, 6))
sns.barplot(x=list(classification_metrics.keys()), y=list(classification_metrics.values()), palette='viridis')
plt.title('Document Classification Accuracy')
plt.xlabel('Metric')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.savefig(os.path.join(figures_path, 'classification_accuracy.png'))
plt.show()

# Visualize extracted dates
print("Visualizing extracted dates...")
extracted_df['dates'] = extracted_df['dates'].apply(eval)  # Convert string representation of list to list
all_dates = [date for sublist in extracted_df['dates'] for date in sublist]

plt.figure(figsize=(10, 6))
sns.histplot(all_dates, bins=50, kde=True, color='green')
plt.title('Distribution of Extracted Dates')
plt.xlabel('Date')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.savefig(os.path.join(figures_path, 'extracted_dates_distribution.png'))
plt.show()

# Visualize extracted amounts
print("Visualizing extracted amounts...")
extracted_df['amounts'] = extracted_df['amounts'].apply(eval)  # Convert string representation of list to list
all_amounts = [float(amount.replace(',', '')) for sublist in extracted_df['amounts'] for amount in sublist]

plt.figure(figsize=(10, 6))
sns.histplot(all_amounts, bins=50, kde=True, color='orange')
plt.title('Distribution of Extracted Amounts')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.savefig(os.path.join(figures_path, 'extracted_amounts_distribution.png'))
plt.show()

# Visualize extracted named entities
print("Visualizing extracted named entities...")
extracted_df['entities'] = extracted_df['entities'].apply(eval)  # Convert string representation of list to list
all_entities = [entity for sublist in extracted_df['entities'] for entity in sublist]
entity_types = [entity[1] for entity in all_entities]

plt.figure(figsize=(10, 6))
sns.countplot(y=entity_types, palette='magma')
plt.title('Distribution of Extracted Named Entities')
plt.xlabel('Count')
plt.ylabel('Entity Type')
plt.savefig(os.path.join(figures_path, 'extracted_entities_distribution.png'))
plt.show()

print("Data visualization completed!")
