"""Model definitions.

Two architectures are available:
  * "cnn"        - the original 3-block CNN from the notebook (trained from scratch)
  * "mobilenet"  - MobileNetV2 transfer learning (add-on; usually more accurate)
"""
import tensorflow as tf
from tensorflow.keras import layers

from . import config


def _augmentation():
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )


def build_cnn(num_classes: int, img_size=config.IMG_SIZE) -> tf.keras.Model:
    """Original notebook architecture (with BatchNorm added for stability)."""
    inputs = layers.Input(shape=(*img_size, 3))
    x = _augmentation()(inputs)
    x = layers.Rescaling(1.0 / 255)(x)
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="tomato_cnn")


def build_mobilenet(num_classes: int, img_size=config.IMG_SIZE,
                    weights="imagenet") -> tf.keras.Model:
    """MobileNetV2 transfer-learning model. Base is frozen; fine-tune with unfreeze()."""
    inputs = layers.Input(shape=(*img_size, 3))
    x = _augmentation()(inputs)
    # MobileNetV2 expects inputs scaled to [-1, 1]
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*img_size, 3), include_top=False, weights=weights)
    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="tomato_mobilenetv2")


def unfreeze(model: tf.keras.Model, last_n: int = 30) -> None:
    """Unfreeze the last `last_n` layers of the MobileNetV2 base for fine-tuning."""
    base = next(l for l in model.layers if isinstance(l, tf.keras.Model))
    base.trainable = True
    for layer in base.layers[:-last_n]:
        layer.trainable = False
    # keep BatchNorm layers frozen for stable fine-tuning
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def compile_model(model: tf.keras.Model, lr: float = 1e-3) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def build(arch: str, num_classes: int, **kwargs) -> tf.keras.Model:
    if arch == "cnn":
        return build_cnn(num_classes, **kwargs)
    if arch == "mobilenet":
        return build_mobilenet(num_classes, **kwargs)
    raise ValueError(f"Unknown architecture: {arch}")
