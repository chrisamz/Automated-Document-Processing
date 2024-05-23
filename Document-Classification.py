# src/document_classification.py

import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Define file paths
ocr_data_path = 'data/processed/ocr_results.csv'
model_path = 'models/document_classification_model.pkl'
vectorizer_path = 'models/tfidf_vectorizer.pkl'

# Create directories if they don't exist
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# Load OCR processed data
print("Loading OCR processed data...")
df = pd.read_csv(ocr_data_path)

# Display the first few rows of the dataset
print("OCR Data:")
print(df.head())

# Preprocess data for classification
print("Preprocessing data for classification...")

# Assuming df contains a 'text' column with the extracted text and a 'label' column with the document type
X = df['text']
y = df['label']  # Adjust 'label' to the actual name of the column containing document types

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a TF-IDF Vectorizer and SVM pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', SVC(kernel='linear', probability=True))
])

# Train the model
print("Training the document classification model...")
pipeline.fit(X_train, y_train)

# Save the model and vectorizer
with open(model_path, 'wb') as f:
    pickle.dump(pipeline, f)

with open(vectorizer_path, 'wb') as f:
    pickle.dump(pipeline.named_steps['tfidf'], f)

# Evaluate the model
print("Evaluating the model...")
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

# Save evaluation results
evaluation_results_path = 'data/processed/classification_evaluation_results.txt'
with open(evaluation_results_path, 'w') as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write("Classification Report:\n")
    f.write(report)

print(f"Document classification completed! Model saved to {model_path} and results saved to {evaluation_results_path}")
