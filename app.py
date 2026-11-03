"""Streamlit web app: upload a tomato leaf photo and get a diagnosis.

Run:  streamlit run app.py
"""
import streamlit as st
from PIL import Image

from src import config, data
from src.predict import load_model, predict_image, pretty

st.set_page_config(page_title="Tomato Leaf Doctor", page_icon="🍅", layout="centered")


@st.cache_resource
def get_model():
    return load_model(), data.load_class_names()


st.title("🍅 Tomato Leaf Doctor")
st.caption("Upload a clear photo of a single tomato leaf to detect common diseases.")

if not config.MODEL_PATH.exists() or not config.CLASS_NAMES_PATH.exists():
    st.error(
        "No trained model found in `models/`. Train one with `python -m src.train` "
        "(see the README), then restart the app."
    )
    st.stop()

model, class_names = get_model()

file = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png"])
if file:
    image = Image.open(file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    results = predict_image(model, image, class_names)
    best, conf = results[0]

    if conf < 0.60:
        st.warning(f"Low confidence ({conf * 100:.1f}%). Try a clearer, closer photo of one leaf.")
    elif best == "Tomato_healthy":
        st.success(f"Healthy leaf ({conf * 100:.1f}% confidence)")
    else:
        st.error(f"Detected: {pretty(best)} ({conf * 100:.1f}% confidence)")

    st.info(config.ADVICE.get(best, ""))

    st.subheader("All probabilities")
    for name, p in results:
        st.write(pretty(name))
        st.progress(min(max(p, 0.0), 1.0), text=f"{p * 100:.1f}%")

st.divider()
st.caption("This tool is an educational aid and not a substitute for an agronomist's diagnosis. "
           "Trained on PlantVillage lab-style images, so field photos may be less accurate.")
