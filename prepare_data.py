"""Download the PlantVillage dataset from Kaggle and build train/val/test splits.

Usage:
    python scripts/prepare_data.py            # download + split
    python scripts/prepare_data.py --skip-download --raw-dir path/to/PlantVillage

Requires a Kaggle API token (kaggle.json) in ~/.kaggle/ for the download step.
"""
import argparse
import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def download(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "plantdisease.zip"
    if not zip_path.exists():
        print("Downloading dataset from Kaggle...")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", config.KAGGLE_DATASET, "-p", str(raw_dir)],
            check=True,
        )
    print("Extracting...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)


def find_class_root(raw_dir: Path) -> Path:
    """Locate the folder that directly contains the class folders."""
    for candidate in [raw_dir, *raw_dir.rglob("*")]:
        if candidate.is_dir() and all((candidate / c).is_dir() for c in config.SELECTED_CLASSES):
            return candidate
    raise FileNotFoundError(
        f"Could not find all classes {config.SELECTED_CLASSES} under {raw_dir}"
    )


def split(class_root: Path, out_dir: Path) -> None:
    random.seed(config.SEED)
    shutil.rmtree(out_dir, ignore_errors=True)

    for cls in config.SELECTED_CLASSES:
        images = sorted(p for p in (class_root / cls).iterdir()
                        if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        random.shuffle(images)
        n = len(images)
        train_end = int(config.TRAIN_FRAC * n)
        val_end = int((config.TRAIN_FRAC + config.VAL_FRAC) * n)
        groups = {
            "train": images[:train_end],
            "validation": images[train_end:val_end],
            "test": images[val_end:],
        }
        for name, paths in groups.items():
            dest_dir = out_dir / name / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            for i, p in enumerate(paths):
                shutil.copy2(p, dest_dir / f"{i}_{p.name}")
        print(f"{cls}: {n} images -> "
              f"{len(groups['train'])}/{len(groups['validation'])}/{len(groups['test'])}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    ap.add_argument("--out-dir", type=Path, default=config.SPLIT_DIR)
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    if not args.skip_download:
        download(args.raw_dir)
    split(find_class_root(args.raw_dir), args.out_dir)
    print(f"Done. Splits written to {args.out_dir}")


if __name__ == "__main__":
    main()
