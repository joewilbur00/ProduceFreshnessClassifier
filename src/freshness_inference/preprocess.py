from pathlib import Path
import tensorflow as tf

from .config import IMG_SIZE


def load_and_preprocess_image(image_path: str) -> tf.Tensor:
    """
    Load one image from disk and preprocess it for inference.

    Returns:
        tf.Tensor of shape (224, 224, 3), dtype float32, values in [0, 1].
    """
    image_path = str(Path(image_path))

    img_bytes = tf.io.read_file(image_path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0

    img.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
    return img


def make_batch_from_path(image_path: str) -> tf.Tensor:
    """
    Convert a single image path into a batch of size 1.
    """
    img = load_and_preprocess_image(image_path)
    return tf.expand_dims(img, axis=0)
