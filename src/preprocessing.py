"""
preprocessing.py

Data preparation utilities for the garbage classification pipeline:
- building train/test directory splits from a raw, class-per-folder dataset
- Keras ImageDataGenerator factories for training, validation, and inference
- a single-image loader used by the prediction API

Kept dependency-free of TensorFlow model code (see model.py) so the API layer
can import only what it needs.
"""

import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
SEED = 42


def build_train_test_dirs(raw_dir: Path, train_dir: Path, test_dir: Path,
                           test_split: float = 0.15, seed: int = SEED,
                           class_names=CLASS_NAMES) -> None:
    """Split a raw/<class>/<image> dataset into train_dir/<class> and test_dir/<class>.

    Wipes and rebuilds train_dir and test_dir each call. Used both for the initial
    dataset split and, with a merged raw_dir, for folding newly uploaded bulk data
    into the training set before a retrain.
    """
    rng = random.Random(seed)
    for d in (train_dir, test_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    for cls in class_names:
        files = list((raw_dir / cls).glob("*"))
        rng.shuffle(files)
        n_test = max(1, int(len(files) * test_split))
        test_files, train_files = files[:n_test], files[n_test:]

        (train_dir / cls).mkdir(parents=True, exist_ok=True)
        (test_dir / cls).mkdir(parents=True, exist_ok=True)

        for f in train_files:
            shutil.copy(f, train_dir / cls / f.name)
        for f in test_files:
            shutil.copy(f, test_dir / cls / f.name)


def merge_uploads_into_raw(upload_dir: Path, raw_dir: Path, class_names=CLASS_NAMES) -> int:
    """Copy newly uploaded, class-labeled images into the raw dataset directory.

    Expects upload_dir to contain one subfolder per class name (same convention
    as raw_dir). Returns the number of images merged. Used by the API's bulk
    upload endpoint before triggering a retrain.
    """
    merged = 0
    for cls in class_names:
        src_dir = upload_dir / cls
        if not src_dir.exists():
            continue
        dst_dir = raw_dir / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        for f in src_dir.glob("*"):
            shutil.copy(f, dst_dir / f.name)
            merged += 1
    return merged


def get_train_val_generators(train_dir: Path, img_size=IMG_SIZE, batch_size=BATCH_SIZE,
                              validation_split: float = 0.15, seed: int = SEED,
                              class_names=CLASS_NAMES):
    """Return (train_gen, val_gen) with augmentation applied to the training split."""
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.15,
        validation_split=validation_split,
    )

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=img_size, batch_size=batch_size,
        class_mode="categorical", classes=class_names, subset="training", seed=seed,
    )
    val_gen = train_datagen.flow_from_directory(
        train_dir, target_size=img_size, batch_size=batch_size,
        class_mode="categorical", classes=class_names, subset="validation", seed=seed,
    )
    return train_gen, val_gen


def get_test_generator(test_dir: Path, img_size=IMG_SIZE, batch_size=BATCH_SIZE,
                        class_names=CLASS_NAMES):
    """Return an unshuffled generator for evaluation."""
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    return test_datagen.flow_from_directory(
        test_dir, target_size=img_size, batch_size=batch_size,
        class_mode="categorical", classes=class_names, shuffle=False,
    )


def load_and_preprocess_image(image_path_or_file, img_size=IMG_SIZE) -> np.ndarray:
    """Load a single image (path or file-like object) into a model-ready batch of 1.

    Used by the API's /predict endpoint for a single user-uploaded image.
    """
    img = Image.open(image_path_or_file).convert("RGB").resize(img_size)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)
