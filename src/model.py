"""
model.py

Model architecture, training, evaluation, and the guarded retraining routine
shared by the notebook and the API's /retrain endpoint.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from .preprocessing import (
    CLASS_NAMES, IMG_SIZE, BATCH_SIZE,
    build_train_test_dirs, get_train_val_generators, get_test_generator,
)


def build_model(num_classes: int = len(CLASS_NAMES), img_size=IMG_SIZE):
    """MobileNetV2 transfer-learning classifier with a frozen base."""
    base = MobileNetV2(input_shape=img_size + (3,), include_top=False, weights="imagenet")
    base.trainable = False

    inputs = layers.Input(shape=img_size + (3,))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base


def _class_weights_from_generator(gen):
    y = gen.classes
    weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
    return dict(enumerate(weights))


def train_model(model, train_gen, val_gen, epochs: int = 15, class_weight: dict = None):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]
    if class_weight is None:
        class_weight = _class_weights_from_generator(train_gen)
    return model.fit(
        train_gen, validation_data=val_gen, epochs=epochs,
        class_weight=class_weight, callbacks=callbacks,
    )


def fine_tune_model(model, base_model, train_gen, val_gen, epochs: int = 8,
                     unfreeze_last_n: int = 30, class_weight: dict = None):
    base_model.trainable = True
    for layer in base_model.layers[:-unfreeze_last_n]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return train_model(model, train_gen, val_gen, epochs=epochs, class_weight=class_weight)


def evaluate_model(model, test_gen, class_names=CLASS_NAMES) -> dict:
    """Run full evaluation and return every metric the assignment requires."""
    test_gen.reset()
    y_true = test_gen.classes
    y_pred_probs = model.predict(test_gen)
    y_pred = np.argmax(y_pred_probs, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    test_loss, test_acc = model.evaluate(test_gen, verbose=0)

    try:
        auc = tf.keras.metrics.AUC(multi_label=False)(
            tf.keras.utils.to_categorical(y_true, num_classes=len(class_names)), y_pred_probs
        ).numpy()
    except Exception:
        auc = None

    return {
        "loss": float(test_loss),
        "accuracy": float(test_acc),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "roc_auc": float(auc) if auc is not None else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True
        ),
    }


def retrain_model(existing_model_path: Path, train_dir: Path, test_dir: Path,
                   epochs: int = 6, min_f1_improvement: float = -0.01) -> tuple:
    """Retrain the currently deployed model on the current contents of train_dir.

    Only overwrites existing_model_path if the retrained model's weighted F1
    does not regress by more than `min_f1_improvement` versus the current
    model's F1 on test_dir. This is the retraining "trigger" guard used by
    the API — call it after new labeled data has been merged into train_dir.

    Returns (metrics: dict, was_deployed: bool).
    """
    class CustomDense(tf.keras.layers.Dense):
        def __init__(self, *args, **kwargs):
            kwargs.pop('quantization_config', None)
            super().__init__(*args, **kwargs)
            
    current_model = tf.keras.models.load_model(existing_model_path, custom_objects={'Dense': CustomDense})
    train_gen, val_gen = get_train_val_generators(train_dir)
    test_gen = get_test_generator(test_dir)

    baseline_metrics = evaluate_model(current_model, test_gen)
    baseline_f1 = baseline_metrics["f1_weighted"]

    train_model(current_model, train_gen, val_gen, epochs=epochs)

    new_metrics = evaluate_model(current_model, test_gen)
    new_f1 = new_metrics["f1_weighted"]

    was_deployed = (new_f1 - baseline_f1) >= min_f1_improvement
    if was_deployed:
        current_model.save(existing_model_path)

    return {"baseline_f1": baseline_f1, **new_metrics}, was_deployed
