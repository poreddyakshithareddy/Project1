import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import os
from datetime import datetime

# ------------------------------------
# CONFIG
# ------------------------------------
st.set_page_config(
    page_title="Crop Nutrient Detector",
    page_icon="🌿",
    layout="wide"
)

MODEL_PATH = r"Models/best_mobilenet_model.h5"
LOG_FILE = "prediction_log.csv"
IMAGE_SIZE = (224, 224)

CLASS_NAMES = ["ALL Present", "ALLAB", "KAB", "NAB", "PAB", "ZNAB"]

FERTILIZERS = {
    "ALL Present": "Plant is healthy. Continue normal NPK schedule.",
    "ALLAB": "Apply Balanced NPK (17-17-17) + micronutrient spray.",
    "KAB": "Apply MOP (Potash) 5–10g per plant.",
    "NAB": "Apply Urea / DAP in small dose.",
    "PAB": "Apply SSP (Single Super Phosphate).",
    "ZNAB": "Apply Zinc Sulphate (ZnSO₄) 0.5–1g per plant."
}

ORGANIC = {
    "ALL Present": "Continue compost + mulching.",
    "ALLAB": "Vermicompost + seaweed extract spray.",
    "KAB": "Banana peel fertilizer or wood ash.",
    "NAB": "Cow dung compost / green manure.",
    "PAB": "Bone meal or rock phosphate.",
    "ZNAB": "Zinc-enriched compost or seaweed extract."
}

INFO = {
    "KAB": "Yellow leaf edges → Potassium deficiency.",
    "NAB": "Pale leaf → Nitrogen deficiency.",
    "PAB": "Purple tint → Phosphorus deficiency.",
    "ZNAB": "Interveinal chlorosis → Zinc deficiency.",
    "ALLAB": "Multiple nutrient imbalance.",
    "ALL Present": "Leaf appears healthy."
}

# ------------------------------------
# Load Model
# ------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ------------------------------------
# Prediction Function
# ------------------------------------
def preprocess(img):
    img = img.resize(IMAGE_SIZE)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, 0)

def predict(img):
    arr = preprocess(img)
    preds = model.predict(arr)[0]
    index = np.argmax(preds)
    return CLASS_NAMES[index], preds[index], preds

# ------------------------------------
# Log Prediction
# ------------------------------------
def log_result(img_name, result, confidence):
    row = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image": img_name,
        "prediction": result,
        "confidence": round(confidence * 100, 2)
    }
    df_row = pd.DataFrame([row])

    if os.path.exists(LOG_FILE):
        old = pd.read_csv(LOG_FILE)
        new = pd.concat([old, df_row], ignore_index=True)
    else:
        new = df_row

    new.to_csv(LOG_FILE, index=False)

# ------------------------------------
# UI
# ------------------------------------
st.title("🌿 Crop Nutrient Deficiency Detector")
st.write("Upload a leaf image to predict nutrient problems and get fertilizer suggestions.")

tab1, tab2 = st.tabs(["🔍 Prediction", "🤖 Chatbot"])

# ---------------------------------------------------------
# TAB 1 — PREDICTION
# ---------------------------------------------------------
with tab1:

    uploaded = st.file_uploader("Upload Leaf Image", type=["jpg","jpeg","png"])

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Leaf", width=350)

        with st.spinner("Analyzing leaf..."):
            label, conf, preds = predict(img)

        st.success(f"Prediction: {label} ({conf*100:.2f}% confidence)")

        st.subheader("💡 Fertilizer Recommendation")
        st.info(FERTILIZERS[label])

        st.subheader("🌱 Organic Option")
        st.warning(ORGANIC[label])

        st.subheader("🧪 Reason / Explanation")
        st.write(INFO[label])

        log_result(uploaded.name, label, conf)

        # Probability Chart
        st.subheader("📊 Prediction Probability")
        df_probs = pd.DataFrame({"Class": CLASS_NAMES, "Probability": preds})
        st.bar_chart(df_probs.set_index("Class"))

        st.markdown("---")
        st.info("Prediction completed successfully.")

# ---------------------------------------------------------

# ---------------------------------------------------------
# TAB 2 — CHATBOT
# ---------------------------------------------------------
with tab2:
    st.header("🤖 Crop Disease & Fertilizer Chatbot")

    st.write("Ask anything about crop diseases, nutrients, fertilizers, soil health, etc.")

    user_input = st.text_input("Ask your question:")

    if st.button("Chat"):
        if user_input.strip() == "":
            st.warning("Please enter a question.")
        else:
            if "nitrogen" in user_input.lower():
                reply = "Nitrogen deficiency causes pale leaves. Apply Urea or compost tea."
            elif "potassium" in user_input.lower():
                reply = "Potassium deficiency shows leaf edge burns. Apply MOP or banana peel fertilizer."
            elif "phosphorus" in user_input.lower():
                reply = "Phosphorus deficiency causes purple leaves. Apply SSP or bone meal."
            elif "zinc" in user_input.lower():
                reply = "Zinc deficiency shows interveinal chlorosis. Apply ZnSO₄ or seaweed extract."
            elif "fertilizer" in user_input.lower():
                reply = "Use NPK 17-17-17 for general growth. For organic, apply compost + seaweed extract."
            elif "hello" in user_input.lower() or "hi" in user_input.lower():
                reply = "Hello! Ask me anything related to plants 🌿."
            else:
                reply = "I can help with nutrient deficiency, fertilizers, soil issues, etc. Try asking about Nitrogen, Potassium, Zinc, etc."

            st.success(f"Chatbot: {reply}")