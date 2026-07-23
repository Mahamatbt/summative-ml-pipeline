"""
prediction.py

Thin inference layer used by the FastAPI /predict endpoint. Keeps the loaded
model cached in memory (loading a .h5 file per request would be far too slow
under Locust load) and exposes a simple predict_image() function.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf

from .preprocessing import CLASS_NAMES, load_and_preprocess_image

_MODEL_CACHE: dict = {"model": None, "path": None}


def get_model(model_path: Path):
    """Load the model once and cache it; reload only if the path on disk changed
    (e.g. after a retrain swapped in a new file)."""
    if _MODEL_CACHE["model"] is None or _MODEL_CACHE["path"] != str(model_path):
        _MODEL_CACHE["model"] = tf.keras.models.load_model(model_path)
        _MODEL_CACHE["path"] = str(model_path)
    return _MODEL_CACHE["model"]


def invalidate_model_cache():
    """Call after a successful retrain so the next prediction picks up the new weights."""
    _MODEL_CACHE["model"] = None
    _MODEL_CACHE["path"] = None


def predict_image(model_path: Path, image_path_or_file, class_names=CLASS_NAMES) -> dict:
    """Predict the class of a single image.

    Returns {"predicted_class": str, "confidence": float, "probabilities": {class: prob}}
    """
    model = get_model(model_path)
    batch = load_and_preprocess_image(image_path_or_file)
    probs = model.predict(batch, verbose=0)[0]

    idx = int(np.argmax(probs))
    return {
        "predicted_class": class_names[idx],
        "confidence": float(probs[idx]),
        "probabilities": {cls: float(p) for cls, p in zip(class_names, probs)},
    }
