from dataclasses import dataclass
from typing import Dict, Any

import tensorflow as tf
from huggingface_hub import hf_hub_download

from .config import (
    MODEL_REPO_ID,
    MODEL_FILENAME,
    CLASS_NAMES,
    DEFAULT_THRESHOLD,
)
from .preprocess import make_batch_from_path


@dataclass
class PredictionResult:
    image_path: str
    predicted_class: int
    predicted_label: str
    fresh_probability: float
    rotten_probability: float
    threshold: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path,
            "predicted_class": self.predicted_class,
            "predicted_label": self.predicted_label,
            "fresh_probability": self.fresh_probability,
            "rotten_probability": self.rotten_probability,
            "threshold": self.threshold,
        }


class FreshnessClassifier:
    def __init__(
        self,
        repo_id: str = MODEL_REPO_ID,
        filename: str = MODEL_FILENAME,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self.repo_id = repo_id
        self.filename = filename
        self.threshold = threshold
        self.model = self._load_model()

    def _load_model(self) -> tf.keras.Model:
        model_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=self.filename,
        )
        return tf.keras.models.load_model(model_path)

    def predict_image(self, image_path: str) -> PredictionResult:
        batch = make_batch_from_path(image_path)

        # Model outputs sigmoid probability for class 1 = Rotten
        rotten_prob = float(self.model.predict(batch, verbose=0)[0][0])
        fresh_prob = 1.0 - rotten_prob

        pred_class = 1 if rotten_prob >= self.threshold else 0
        pred_label = CLASS_NAMES[pred_class]

        return PredictionResult(
            image_path=image_path,
            predicted_class=pred_class,
            predicted_label=pred_label,
            fresh_probability=round(fresh_prob, 6),
            rotten_probability=round(rotten_prob, 6),
            threshold=self.threshold,
        )
