"""Evaluate a trained model on the test split.

Writes a classification report and confusion matrix to reports/.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from . import config, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(config.MODEL_PATH))
    ap.add_argument("--split-dir", default=str(config.SPLIT_DIR))
    args = ap.parse_args()

    model = tf.keras.models.load_model(args.model)
    _, _, test_ds, class_names = data.load_datasets(Path(args.split_dir))

    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"Test loss: {loss:.4f} | Test accuracy: {acc * 100:.2f}%")

    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "classification_report.txt").write_text(
        f"Test accuracy: {acc * 100:.2f}%\n\n{report}")

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "confusion_matrix.png", dpi=150)
    print(f"Reports saved to {config.REPORTS_DIR}")


if __name__ == "__main__":
    main()
