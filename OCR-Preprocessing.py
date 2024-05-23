# src/ocr_preprocessing.py

import os
import cv2
import pytesseract
import pandas as pd
from pytesseract import Output
from pdf2image import convert_from_path

# Define file paths
raw_data_path = 'data/raw/'
processed_data_path = 'data/processed/ocr_results.csv'

# Create directories if they don't exist
os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)

# Function to preprocess image for OCR
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    return gray

# Function to extract text from image using Tesseract OCR
def extract_text_from_image(image_path):
    preprocessed_image = preprocess_image(image_path)
    text = pytesseract.image_to_string(preprocessed_image, lang='eng', config='--oem 1 --psm 6')
    return text

# Function to process PDF files
def process_pdf(pdf_path):
    images = convert_from_path(pdf_path)
    text = ""
    for i, image in enumerate(images):
        image_path = f'{os.path.splitext(pdf_path)[0]}_page_{i}.png'
        image.save(image_path, 'PNG')
        text += extract_text_from_image(image_path) + "\n"
        os.remove(image_path)  # Clean up temporary image file
    return text

# List to hold OCR results
ocr_results = []

# Process all files in the raw data directory
print("Processing raw documents for OCR...")
for filename in os.listdir(raw_data_path):
    file_path = os.path.join(raw_data_path, filename)
    if filename.endswith('.pdf'):
        print(f"Processing PDF file: {filename}")
        text = process_pdf(file_path)
    elif filename.endswith(('.png', '.jpg', '.jpeg')):
        print(f"Processing image file: {filename}")
        text = extract_text_from_image(file_path)
    else:
        print(f"Unsupported file type: {filename}")
        continue
    
    ocr_results.append({'filename': filename, 'text': text})

# Create a DataFrame from the OCR results
ocr_df = pd.DataFrame(ocr_results)

# Save the OCR results
print("Saving OCR results...")
ocr_df.to_csv(processed_data_path, index=False)

print(f"OCR preprocessing completed! Results saved to {processed_data_path}")
