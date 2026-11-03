"""Central configuration for the Tomato Disease Classifier."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SPLIT_DIR = DATA_DIR / "split"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

MODEL_PATH = MODELS_DIR / "plant_disease_model.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"

# Same four classes used in the original notebook
SELECTED_CLASSES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_healthy",
]

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42
EPOCHS = 20
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15  # remaining 0.15 is test

KAGGLE_DATASET = "emmarex/plantdisease"

# Short advice shown in the web app
ADVICE = {
    "Tomato_Bacterial_spot": "Remove infected leaves, avoid overhead watering, and use a copper-based bactericide if it spreads.",
    "Tomato_Early_blight": "Prune lower infected leaves, rotate crops, mulch the soil, and apply a suitable fungicide.",
    "Tomato_Late_blight": "Act fast: remove and destroy infected plants, improve airflow, and apply a recommended fungicide.",
    "Tomato_healthy": "The leaf looks healthy. Keep monitoring regularly.",
}
