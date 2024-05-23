# src/information_extraction_model.py

import os
import spacy
import pandas as pd
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.util import minibatch, compounding
from spacy.training.example import Example

# Define file paths
ocr_data_path = 'data/processed/ocr_results.csv'
model_path = 'models/information_extraction_model'
evaluation_results_path = 'data/processed/information_extraction_evaluation_results.txt'
figures_path = 'figures'

# Create directories if they don't exist
os.makedirs(os.path.dirname(model_path), exist_ok=True)
os.makedirs(figures_path, exist_ok=True)

# Load OCR processed data
print("Loading OCR processed data...")
df = pd.read_csv(ocr_data_path)

# Display the first few rows of the dataset
print("OCR Data:")
print(df.head())

# Load spaCy model for Named Entity Recognition
print("Loading spaCy model...")
nlp = spacy.blank("en")

# Define a function to convert data to spaCy's training format
def convert_data_to_spacy(df):
    training_data = []
    for _, row in df.iterrows():
        text = row['text']
        entities = row['entities']  # Adjust 'entities' to the actual name of the column containing annotated entities
        ents = [(ent['start'], ent['end'], ent['label']) for ent in entities]
        training_data.append((text, {"entities": ents}))
    return training_data

# Convert the data to spaCy's training format
print("Converting data to spaCy's training format...")
TRAIN_DATA = convert_data_to_spacy(df)

# Create the DocBin object for training
doc_bin = DocBin()
for text, annotations in TRAIN_DATA:
    doc = nlp.make_doc(text)
    ents = []
    for start, end, label in annotations.get("entities"):
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is not None:
            ents.append(span)
    doc.ents = ents
    doc_bin.add(doc)

# Save the DocBin object
doc_bin.to_disk("train.spacy")

# Add the NER pipeline component
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner", last=True)
else:
    ner = nlp.get_pipe("ner")

# Add labels to the NER component
for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# Train the NER model
print("Training the NER model...")
optimizer = nlp.begin_training()

# Training loop
for itn in range(30):
    losses = {}
    batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
    for batch in batches:
        for text, annotations in batch:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], drop=0.5, losses=losses, sgd=optimizer)
    print(f"Iteration {itn}, Losses: {losses}")

# Save the trained model
print("Saving the trained model...")
nlp.to_disk(model_path)

# Evaluate the model
print("Evaluating the model...")
results = []
for text, annotations in TRAIN_DATA:
    doc = nlp(text)
    pred_ents = [(ent.text, ent.label_) for ent in doc.ents]
    true_ents = [(text[start:end], label) for start, end, label in annotations.get("entities")]
    results.append({
        "text": text,
        "true_entities": true_ents,
        "predicted_entities": pred_ents
    })

# Save evaluation results
with open(evaluation_results_path, 'w') as f:
    for result in results:
        f.write(f"Text: {result['text']}\n")
        f.write(f"True Entities: {result['true_entities']}\n")
        f.write(f"Predicted Entities: {result['predicted_entities']}\n\n")

print(f"Information extraction completed! Model saved to {model_path} and results saved to {evaluation_results_path}")
