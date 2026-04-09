import numpy as np
from PIL import Image
import tensorflow as tf

from src.freshness_inference.preprocess import load_and_preprocess_image


def test_load_and_preprocess_image(tmp_path):
    image_path = tmp_path / "test.jpg"

    arr = np.zeros((300, 300, 3), dtype=np.uint8)
    arr[:, :, 0] = 255
    Image.fromarray(arr).save(image_path)

    img = load_and_preprocess_image(str(image_path))

    assert isinstance(img, tf.Tensor)
    assert img.shape == (224, 224, 3)
    assert img.dtype == tf.float32
