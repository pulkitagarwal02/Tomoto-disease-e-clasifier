"""Dataset loading utilities."""
import json
from pathlib import Path

import tensorflow as tf

from . import config


def load_datasets(split_dir: Path = config.SPLIT_DIR, img_size=config.IMG_SIZE,
                  batch_size: int = config.BATCH_SIZE):
    """Return (train_ds, val_ds, test_ds, class_names)."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        split_dir / "train", image_size=img_size, batch_size=batch_size, seed=config.SEED)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        split_dir / "validation", image_size=img_size, batch_size=batch_size, shuffle=False)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        split_dir / "test", image_size=img_size, batch_size=batch_size, shuffle=False)

    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000, seed=config.SEED).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    test_ds = test_ds.cache().prefetch(autotune)
    return train_ds, val_ds, test_ds, class_names


def save_class_names(class_names, path: Path = config.CLASS_NAMES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(class_names, indent=2))


def load_class_names(path: Path = config.CLASS_NAMES_PATH):
    return json.loads(Path(path).read_text())
