# Produce Freshness Classifier

A computer vision application that classifies images of produce as Fresh or Rotten using a fine-tuned ResNet50 model.

This project demonstrates a production-style ML inference pipeline, including:
- Model hosting via Hugging Face
- Clean preprocessing and inference pipeline
- CLI and batch prediction workflows
- Interactive Streamlit app
- Unit testing with pytest

---

## Project Overview

This application uses a deep learning model trained on a food freshness dataset to predict whether an image of produce is:

- Fresh (0)
- Rotten (1)

The model is stored externally and downloaded at runtime from Hugging Face Hub, keeping the repository lightweight and reproducible.

---

## Repository Structure

ProduceFreshnessClassifier/
├── src/                     # Core inference package
│   └── freshness_inference/
├── scripts/                 # Batch prediction scripts
├── sample_images/           # Placeholder for local test images
├── tests/                   # Unit tests (pytest)
├── app.py                   # Streamlit application
├── requirements.txt
├── pytest.ini
└── README.md

---

## Setup Instructions

### 1. Clone the Repository

git clone <your-repo-url>
cd ProduceFreshnessClassifier

### 2. Create a Virtual Environment

python -m venv .venv

Activate it:

Windows (PowerShell):
.venv\Scripts\Activate

### 3. Install Dependencies

pip install --upgrade pip
pip install -r requirements.txt

---

## How the Model Works

- Input: Image resized to 224×224 RGB
- Preprocessing:
  - Converted to float32
  - Scaled to [0, 1]
- Model:
  - ResNet50 backbone (transfer learning)
  - Sigmoid output
- Output:
  - Probability of Rotten
  - Threshold (default = 0.5) determines class

---

## Running the Application

### Option 1: Single Image Prediction (CLI)

python -m src.freshness_inference.predict --image-path sample_images/your_image.jpg

Example output:

{
  "image_path": "sample_images/apple.jpg",
  "predicted_class": 0,
  "predicted_label": "Fresh",
  "fresh_probability": 0.82,
  "rotten_probability": 0.18,
  "threshold": 0.5
}

---

### Option 2: Batch Predictions

python -m scripts.predict_folder --folder-path sample_images --output-csv predictions.csv

This creates:

predictions.csv

With columns such as:

image_path | predicted_label | fresh_probability | rotten_probability

---

### Option 3: Run the Streamlit App

streamlit run app.py

Features:
- Upload an image
- View prediction instantly
- See probabilities for each class

---

## Running Tests

pytest

Tests validate:
- Image preprocessing pipeline
- Input shape
- Data type consistency

---

## Adding Your Own Images

Place images inside:

sample_images/

Supported formats:
- .jpg
- .jpeg
- .png
- .bmp

---

## Model Hosting

The model is stored externally on Hugging Face Hub and downloaded at runtime using:

hf_hub_download(...)

This ensures:
- No large files in GitHub
- Reproducibility
- Easy model versioning

---

## Common Issues and Fixes

ModuleNotFoundError: 'src'

Run scripts using module syntax:

python -m src.freshness_inference.predict

Not:

python predict.py

Terminal frozen after running app:

Press:
Ctrl + C

Image errors:

Use standard .jpg or .png images. Avoid uncommon formats like .heic or .webp.

---

