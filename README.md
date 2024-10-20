# Amazon ML Challenge

## Team: Optimus_Prime  
**Members:**  
- Akshay Perison Davis J  
- Arulpathi A  
- Mithran M  
- Sharan S  

## Problem Statement
The goal of this project is to develop a machine learning model that extracts important information from product images, such as weight, volume, dimensions, and other physical attributes. This solution is essential in sectors like healthcare, e-commerce, and content moderation, where product details are often missing or incomplete. The model processes product images and predicts physical details in a structured format (e.g., "34 gram", "12.5 cm").

---

## Machine Learning Approach for Extracting Physical Attributes from Product Images

### 1. Problem Overview  
The task is to extract important physical attributes like weight, volume, and dimensions from product images. This is crucial for industries like e-commerce, healthcare, and logistics, where such details may not be easily accessible or structured. We use Optical Character Recognition (OCR) for text extraction from images and Named Entity Recognition (NER) to identify target attributes from the extracted text.

### 2. Machine Learning Approach

#### Step 1: Image Downloading
- Download the product images using the URLs provided in the dataset.
- In case of download failure, a placeholder image is generated after several retries.
- Multiprocessing is used to speed up the download process.

#### Step 2: Text Extraction with EasyOCR
- Extract text from product images using EasyOCR.  
- EasyOCR reads text from images and returns the recognized characters.

#### Step 3: Named Entity Recognition (NER)
- The extracted text is used to train an NER model that detects and classifies entities such as weight, volume, and dimensions.
- We use the Spacy library to train the NER model.

#### Step 4: Test Data Prediction
- Use the trained NER model to predict entities in test images.
- The extracted text from test images is processed, and the identified entities are saved in a CSV file.

### 3. Machine Learning Models Used

- **EasyOCR:**  
  - **Purpose:** Extract text from product images.  
  - **Why:** Lightweight, easy to use, supports multiple languages, and provides good accuracy.

- **Spacy NER:**  
  - **Purpose:** Identify specific entities like weight, volume, and dimensions from text.  
  - **Why:** Spacy provides a customizable NER pipeline, widely used in NLP tasks.

### 4. Experiments Conducted

- **Experiment 1: Basic OCR Text Extraction**  
  - **Objective:** Extract text using EasyOCR and clean the data.  
  - **Results:** Successful extraction with corrections for common OCR errors.

- **Experiment 2: NER Model Training**  
  - **Objective:** Train Spacy NER to recognize entities like weight and volume.  
  - **Results:** Trained with an 80/20 split over 10 epochs, showing progressive loss reduction.

- **Experiment 3: Test Data Prediction**  
  - **Objective:** Evaluate the NER model on unseen test data.  
  - **Results:** Successfully predicted entities and saved results for evaluation.

### 5. Conclusion
Our approach efficiently combines image processing, OCR, and NLP techniques to extract relevant physical attributes from product images. By using EasyOCR for text extraction and Spacy NER for entity identification, we achieved good performance in predicting attributes like weight and volume. Future improvements could include:
- Implementing advanced error correction in the OCR step.
- Exploring other OCR engines for higher accuracy.
- Tuning the NER model with more diverse and extensive training data to improve accuracy in real-world scenarios.

This approach demonstrates a practical application of machine learning techniques to extract structured data from images in industries like e-commerce and logistics.

---


