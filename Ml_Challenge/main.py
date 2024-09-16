# Install necessary libraries for Google Colab


# Import necessary libraries
import easyocr
import re
import os
import urllib
import pandas as pd
from PIL import Image
from pathlib import Path
from urllib.parse import urlparse, unquote
from functools import partial
import multiprocessing
from tqdm import tqdm
import spacy
from spacy.tokens import DocBin
from sklearn.model_selection import train_test_split
import random
import time

# Step 1: Set to use GPU
def use_gpu():
    if spacy.prefer_gpu():
        print("Using GPU for training")
        spacy.require_gpu()  # Ensure Spacy uses the GPU
    else:
        print("GPU not available, using CPU")

# Step 2: Downloading images from URLs
def create_placeholder_image(image_save_path):
    try:
        placeholder_image = Image.new('RGB', (100, 100), color='black')
        placeholder_image.save(image_save_path)
    except Exception as e:
        return

def download_image(image_link, save_folder, retries=3, delay=3):
    if not isinstance(image_link, str):
        return

    filename = Path(image_link).name
    image_save_path = os.path.join(save_folder, filename)

    if os.path.exists(image_save_path):
        return

    for _ in range(retries):
        try:
            urllib.request.urlretrieve(image_link, image_save_path)
            return
        except:
            time.sleep(delay)

    create_placeholder_image(image_save_path)

def download_images(image_links, download_folder, allow_multiprocessing=True):
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    if allow_multiprocessing:
        download_image_partial = partial(
            download_image, save_folder=download_folder, retries=3, delay=3)

        with multiprocessing.Pool(16) as pool:  # Multiprocessing with fewer cores in Colab
            list(tqdm(pool.imap(download_image_partial, image_links), total=len(image_links)))
            pool.close()
            pool.join()
    else:
        for image_link in tqdm(image_links, total=len(image_links)):
            download_image(image_link, save_folder=download_folder, retries=3, delay=3)

# Step 3: Cleaning OCR text
def clean_text(text):
    text = text.replace('O', '0')
    text = text.replace('l', '1')
    text = text.replace('I', '1')
    text = text.replace('|', '1')
    text = text.replace('m1', 'ml')
    text = text.lower()

    text = re.sub(r'(\d+)\s*mG\b', r'\1 mg', text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+)\s*9\b', r'\1 g', text)
    return text

# Step 4: Processing OCR and feature extraction with easyOCR
def process(image_path):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image_path)
    extracted_texts = [text[1] for text in result]
    ocr_text = ' '.join(extracted_texts)
    ocr_text = clean_text(ocr_text)
    return ocr_text

# Step 5: Training the Named Entity Recognition (NER) Model
def train_ner(train_data):
    use_gpu()  # Ensure GPU usage if available
    nlp = spacy.blank("en")

    # Define the pipeline
    ner = nlp.create_pipe("ner")
    nlp.add_pipe(ner)

    # Add labels for each entity (for example: WEIGHT, DIMENSION, etc.)
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # Create a train/test split
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    # Convert data to Spacy format
    db = DocBin()
    for text, annot in train_data:
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annot["entities"]:
            span = doc.char_span(start, end, label=label)
            ents.append(span)
        doc.ents = ents
        db.add(doc)

    db.to_disk("./train.spacy")

    # Train the model
    optimizer = nlp.begin_training()
    for epoch in range(10):  # Define number of epochs
        random.shuffle(train_data)
        losses = {}
        batches = spacy.util.minibatch(train_data, size=8)
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, drop=0.5, losses=losses)
        print(f"Losses at iteration {epoch}: {losses}")

    # Save the model after training
    nlp.to_disk("/content/ner_model")  # Save the trained model to disk
    return nlp

# Step 6: Predicting for test data using the trained NER model
def predict_ner(nlp, test_data):
    results = []
    for image_path in test_data['image_path']:
        ocr_text = process(image_path)
        doc = nlp(ocr_text)
        entity_value = ""
        for ent in doc.ents:
            entity_value += f"{ent.text} "
        entity_value = entity_value.strip()
        results.append(entity_value)
    return results

# Helper function to convert URL to local path
def convert_url_to_local_path(image_link, images_folder):
    filename = Path(image_link).name
    return os.path.join(images_folder, filename)

# Step 7: Load the Saved Model
def load_model(model_path="/content/ner_model"):
    print("Loading model from", model_path)
    nlp = spacy.load(model_path)
    return nlp

# Step 8: Main Execution
def main():
    # Load training data
    train_df = pd.read_csv("/content/train.csv")  # Path to train.csv in Colab

    # Preprocess images for training and extract OCR text
    image_links = train_df['image_link'].dropna().unique()
    images_folder = '/content/train_images'  # Image folder in Colab
    download_images(image_links, download_folder=images_folder)

    # Extract and clean OCR text from training data
    train_data = []
    for _, row in train_df.iterrows():
        image_path = convert_url_to_local_path(row['image_link'], images_folder)
        ocr_text = process(image_path)
        entity_value = row['entity_value']

        # Define entities for NER
        start_idx = ocr_text.find(entity_value)
        if start_idx != -1:
            end_idx = start_idx + len(entity_value)
            train_data.append((ocr_text, {"entities": [(start_idx, end_idx, row['entity_name'])]}))

    # Train NER model
    nlp_model = train_ner(train_data)

    # Alternatively, load an already trained model
    # nlp_model = load_model()  # Uncomment to load a pre-trained model instead of training from scratch

    # Load test data and make predictions
    test_df = pd.read_csv("/content/test.csv")  # Path to test.csv in Colab
    test_images_folder = '/content/test_images'  # Test image folder in Colab
    download_images(test_df['image_link'], download_folder=test_images_folder)
    test_df['image_path'] = test_df['image_link'].apply(lambda x: convert_url_to_local_path(x, test_images_folder))

    predictions = predict_ner(nlp_model, test_df)

    # Format predictions into the required output format
    test_df['prediction'] = predictions
    test_df[['index', 'prediction']].to_csv("/content/result.csv", index=False)  # Save result.csv to Colab

    # Optional: Run the sanity check if provided
    os.system("python /content/sanity.py --test_filename /content/test.csv --output_filename /content/result.csv")

# Run the main function
main()
