from pathlib import Path
import tempfile

import streamlit as st
from PIL import Image

from src.freshness_inference.model import FreshnessClassifier

st.set_page_config(page_title="Freshness Classifier", layout="centered")
st.title("Fresh vs Rotten Image Classifier")

@st.cache_resource
def load_classifier() -> FreshnessClassifier:
    return FreshnessClassifier()

classifier = load_classifier()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    result = classifier.predict_image(temp_path)

    st.subheader("Prediction")
    st.write(f"**Label:** {result.predicted_label}")
    st.write(f"**Fresh probability:** {result.fresh_probability:.4f}")
    st.write(f"**Rotten probability:** {result.rotten_probability:.4f}")
