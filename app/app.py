import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

MODEL_PATH = "models/best_model_resnet.keras"
IMG_SIZE = (224, 224)

CLASS_INFO = {
    "akiec": "Actinic Keratoses / Intraepithelial Carcinoma",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis-like Lesion",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus (benign mole)",
    "vasc": "Vascular Lesion",
}

@st.cache_resource
def load_classifier():
    model = load_model(MODEL_PATH)
    with open("data/processed/label_classes.txt") as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names

def preprocess_image(image: Image.Image):
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(image).astype(np.float32)
    arr = tf.keras.applications.resnet50.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def main():
    st.set_page_config(page_title="Skin Lesion Classifier", page_icon="🔬", layout="centered")

    st.title("🔬 Skin Lesion Classifier")
    st.markdown(
        "Upload a dermatoscopic image to classify it into one of 7 skin lesion categories. "
        "Trained on the [HAM10000 dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) "
        "using a fine-tuned ResNet50 (transfer learning, macro F1: 0.676 on held-out test set)."
    )

    st.warning(
        "⚠️ **Disclaimer:** This is a portfolio/educational project, not a medical diagnostic tool. "
        "It should never be used to make real health decisions. Always consult a qualified "
        "dermatologist or doctor for any skin concerns."
    )

    model, class_names = load_classifier()

    uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1.3])

        with col1:
            st.image(image, caption="Uploaded image", use_container_width=True)

        with st.spinner("Analyzing..."):
            processed = preprocess_image(image)
            preds = model.predict(processed, verbose=0)[0]

        predicted_idx = int(np.argmax(preds))
        predicted_class = class_names[predicted_idx]
        confidence = preds[predicted_idx] * 100

        with col2:
            st.subheader("Prediction")
            st.markdown(f"**{CLASS_INFO.get(predicted_class, predicted_class)}** ({predicted_class})")
            st.metric("Confidence", f"{confidence:.1f}%")

            st.subheader("All class probabilities")
            sorted_idx = np.argsort(preds)[::-1]
            for idx in sorted_idx:
                cls = class_names[idx]
                prob = preds[idx] * 100
                label = f"{CLASS_INFO.get(cls, cls)} ({cls})"
                st.progress(float(preds[idx]), text=f"{label}: {prob:.1f}%")

    st.markdown("---")
    st.caption(
        "Built with TensorFlow/Keras, transfer learning (ResNet50), and Streamlit. "
        "[View source on GitHub](https://github.com/kriti3001/skin-lesion-classifier)"
    )

if __name__ == "__main__":
    main()