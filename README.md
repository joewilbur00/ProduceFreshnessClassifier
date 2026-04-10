# Freshness Inference

This repository runs inference for a binary food freshness classifier.

## Model
The model is hosted on Hugging Face:

`pqhunter15/freshnessclassv1`

The code downloads this file automatically at runtime:

`fresh_rotten_resnet_tuned_conv4_conv5.keras`

## Classes
- 0 = Fresh
- 1 = Rotten

## Setup

```bash
pip install -r requirements.txt

### run single image inference
python -m src.freshness_inference.predict --image-path sample_images/example.jpg

### Run inference on a folder of images
python scripts/predict_folder.py --folder-path sample_images --output-csv predictions.csv

### Run streamlit app
streamlit run app.py
