"""Train the classifier.

Examples:
    python -m src.train                       # original CNN
    python -m src.train --arch mobilenet      # transfer learning (+ optional fine-tune)
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from . import config, data, model as model_lib


def plot_history(histories, out_path):
    acc = sum((h.history["accuracy"] for h in histories), [])
    val_acc = sum((h.history["val_accuracy"] for h in histories), [])
    loss = sum((h.history["loss"] for h in histories), [])
    val_loss = sum((h.history["val_loss"] for h in histories), [])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(acc, label="train"); axes[0].plot(val_acc, label="validation")
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(loss, label="train"); axes[1].plot(val_loss, label="validation")
    axes[1].set_title("Loss"); axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["cnn", "mobilenet"], default="cnn")
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--fine-tune-epochs", type=int, default=0,
                    help="mobilenet only: extra epochs after unfreezing the top layers")
    ap.add_argument("--split-dir", type=str, default=str(config.SPLIT_DIR))
    ap.add_argument("--output", type=str, default=str(config.MODEL_PATH))
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(config.SEED)

    from pathlib import Path
    train_ds, val_ds, _, class_names = data.load_datasets(Path(args.split_dir))
    data.save_class_names(class_names)

    net = model_lib.build(args.arch, len(class_names))
    model_lib.compile_model(net)
    net.summary()

    def callbacks():
        return [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                             restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
        ]

    histories = [net.fit(train_ds, validation_data=val_ds,
                         epochs=args.epochs, callbacks=callbacks())]

    if args.arch == "mobilenet" and args.fine_tune_epochs > 0:
        print("Fine-tuning top MobileNetV2 layers...")
        model_lib.unfreeze(net)
        model_lib.compile_model(net, lr=1e-5)
        histories.append(net.fit(train_ds, validation_data=val_ds,
                                 epochs=args.fine_tune_epochs, callbacks=callbacks()))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    net.save(out)
    plot_history(histories, config.REPORTS_DIR / "training_curves.png")
    print(f"Model saved to {out}")


if __name__ == "__main__":
    main()
