"""Single-image inference.

CLI:
    python -m src.predict path/to/leaf.jpg
"""
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from . import config, data


def load_model(model_path=config.MODEL_PATH):
    return tf.keras.models.load_model(model_path)


def preprocess(image: Image.Image, img_size=config.IMG_SIZE) -> np.ndarray:
    """RGB, resize, add batch dim. Rescaling happens inside the model."""
    image = image.convert("RGB").resize(img_size)
    return np.expand_dims(np.asarray(image, dtype="float32"), 0)


def predict_image(model, image: Image.Image, class_names, top_k: int = 4):
    probs = model.predict(preprocess(image), verbose=0)[0]
    order = np.argsort(probs)[::-1][:top_k]
    return [(class_names[i], float(probs[i])) for i in order]


def pretty(name: str) -> str:
    return name.replace("Tomato_", "").replace("_", " ").capitalize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--model", default=str(config.MODEL_PATH))
    ap.add_argument("--classes", default=str(config.CLASS_NAMES_PATH))
    args = ap.parse_args()

    model = load_model(args.model)
    class_names = data.load_class_names(Path(args.classes))
    for name, p in predict_image(model, Image.open(args.image), class_names):
        print(f"{pretty(name):<18} {p * 100:6.2f}%")


if __name__ == "__main__":
    main()
