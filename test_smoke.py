"""Fast smoke tests that need no dataset or internet."""
import numpy as np
from PIL import Image

from src import config
from src.model import build_cnn, build_mobilenet, compile_model
from src.predict import predict_image, preprocess

NUM_CLASSES = len(config.SELECTED_CLASSES)


def test_cnn_output_shape():
    m = build_cnn(NUM_CLASSES)
    out = m.predict(np.zeros((2, *config.IMG_SIZE, 3), dtype="float32"), verbose=0)
    assert out.shape == (2, NUM_CLASSES)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-4)


def test_mobilenet_builds_without_download():
    m = build_mobilenet(NUM_CLASSES, weights=None)
    compile_model(m)
    assert m.output_shape == (None, NUM_CLASSES)


def test_preprocess_shape_and_dtype():
    img = Image.new("RGB", (300, 200), color=(10, 200, 30))
    arr = preprocess(img)
    assert arr.shape == (1, *config.IMG_SIZE, 3)
    assert arr.dtype == np.float32


def test_predict_image_returns_sorted_probs():
    m = build_cnn(NUM_CLASSES)
    img = Image.new("RGB", (64, 64), color=(120, 120, 120))
    res = predict_image(m, img, config.SELECTED_CLASSES)
    probs = [p for _, p in res]
    assert probs == sorted(probs, reverse=True)
    assert abs(sum(probs) - 1.0) < 1e-3
