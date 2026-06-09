import streamlit as st
import tensorflow as tf
import torch
import numpy as np
import json
import re
import os
from PIL import Image
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from email_alert import send_disaster_alert


# ===============================
# PAGE CONFIG
# ===============================

st.set_page_config(
    page_title="Multimodal Disaster Alert System",
    page_icon="🚨",
    layout="wide"
)


# ===============================
# PATHS
# ===============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_resnet50_model.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "models", "class_names.json")
TEXT_MODEL_PATH = os.path.join(BASE_DIR, "models", "text_model_distilbert")


# ===============================
# LOAD MODELS
# ===============================

@st.cache_resource
def load_image_model():
    model = tf.keras.models.load_model(IMAGE_MODEL_PATH)

    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)

    return model, class_names


@st.cache_resource
def load_text_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = DistilBertTokenizerFast.from_pretrained(TEXT_MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(TEXT_MODEL_PATH)

    model.to(device)
    model.eval()

    return tokenizer, model, device


image_model, image_class_names = load_image_model()
tokenizer, text_model, device = load_text_model()


# ===============================
# DISPLAY NAMES
# ===============================

class_display_names = {
    "Damaged_Infrastructure": "Damaged Infrastructure",
    "Fire_Disaster": "Fire Disaster",
    "Human_Damage": "Human Damage / Casualty",
    "Land_Disaster": "Land Disaster / Landslide",
    "Non_Damage": "No Visible Disaster",
    "Water_Disaster": "Water Disaster / Flood",
    "Possible_Disaster_Text_Only": "Possible Disaster Based on Text Only",
    "Uncertain": "Uncertain / Needs Manual Verification"
}


# ===============================
# TEXT CLEANING
# ===============================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ===============================
# IMAGE PREDICTION
# ===============================

def predict_image(uploaded_image):
    image = Image.open(uploaded_image).convert("RGB")
    image = image.resize((224, 224))

    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)

    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

    preds = image_model.predict(img_array)

    predicted_index = np.argmax(preds[0])
    confidence = float(preds[0][predicted_index])
    predicted_class = image_class_names[predicted_index]

    return predicted_class, confidence


# ===============================
# TEXT PREDICTION
# ===============================

def predict_text(text):
    cleaned = clean_text(text)

    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = text_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    not_disaster_conf = float(probs[0][0])
    disaster_conf = float(probs[0][1])

    predicted_label = "Disaster" if disaster_conf >= not_disaster_conf else "Not Disaster"
    confidence = max(disaster_conf, not_disaster_conf)

    return predicted_label, confidence, disaster_conf


# ===============================
# FUSION LOGIC
# ===============================

def fusion_predict(uploaded_image, text):
    image_class, image_conf = predict_image(uploaded_image)
    text_label, text_conf, disaster_text_conf = predict_text(text)

    IMAGE_TRUST_THRESHOLD = 0.60
    ALERT_THRESHOLD = 0.75

    if image_conf < IMAGE_TRUST_THRESHOLD:
        if text_label == "Disaster" and disaster_text_conf >= 0.80:
            final_class = "Possible_Disaster_Text_Only"
            final_confidence = disaster_text_conf * 0.75
            alert_required = False
        else:
            final_class = "Uncertain"
            final_confidence = (image_conf + disaster_text_conf) / 2
            alert_required = False

    elif image_class == "Non_Damage" and text_label == "Not Disaster":
        final_class = "Non_Damage"
        final_confidence = (image_conf + text_conf) / 2
        alert_required = False

    elif image_class != "Non_Damage" and text_label == "Disaster":
        final_class = image_class
        final_confidence = (0.7 * image_conf) + (0.3 * disaster_text_conf)
        alert_required = final_confidence >= ALERT_THRESHOLD

    elif image_class != "Non_Damage" and text_label == "Not Disaster":
        final_class = image_class
        final_confidence = (0.8 * image_conf) + (0.2 * disaster_text_conf)
        alert_required = image_conf >= 0.85

    else:
        final_class = "Possible_Disaster_Text_Only"
        final_confidence = (0.3 * image_conf) + (0.7 * disaster_text_conf)
        alert_required = False

    result = {
        "image_prediction": image_class,
        "image_confidence": round(image_conf * 100, 2),

        "text_prediction": text_label,
        "text_confidence": round(text_conf * 100, 2),
        "text_disaster_confidence": round(disaster_text_conf * 100, 2),

        "final_prediction": final_class,
        "final_confidence": round(final_confidence * 100, 2),

        "alert_required": alert_required
    }

    return result


# ===============================
# STREAMLIT UI
# ===============================

st.title("🚨 Multimodal Disaster Alert System")
st.write(
    "Upload an image and enter disaster-related text. "
    "The system combines image and text predictions and sends an automated alert if disaster confidence is high."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Image Input")

    uploaded_image = st.file_uploader(
        "Upload disaster image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

with col2:
    st.subheader("📝 Text Input")

    text_input = st.text_area(
        "Enter text / tweet / emergency message",
        height=180,
        placeholder="Example: Heavy rainfall has caused flooding and people need urgent help."
    )

st.divider()


# ===============================
# ANALYZE BUTTON
# ===============================

if st.button("Analyze Disaster"):
    if uploaded_image is None:
        st.error("Please upload an image.")

    elif text_input.strip() == "":
        st.error("Please enter text.")

    else:
        with st.spinner("Analyzing image and text..."):
            result = fusion_predict(uploaded_image, text_input)

        st.subheader("✅ Prediction Results")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Image Prediction",
                class_display_names.get(result["image_prediction"], result["image_prediction"]),
                f'{result["image_confidence"]}%'
            )

        with c2:
            st.metric(
                "Text Prediction",
                result["text_prediction"],
                f'{result["text_confidence"]}%'
            )

        with c3:
            st.metric(
                "Final Prediction",
                class_display_names.get(result["final_prediction"], result["final_prediction"]),
                f'{result["final_confidence"]}%'
            )

        st.divider()

        # ===============================
        # ALERT + EMAIL SECTION
        # ===============================

        if result["alert_required"]:
            st.error("🚨 ALERT REQUIRED: Disaster detected with high confidence.")

            st.warning("📧 Sending automated email alert to authorities...")

            try:
                sender_email = st.secrets["email"]["sender_email"]
                sender_password = st.secrets["email"]["sender_password"]
                receiver_email = st.secrets["email"]["receiver_email"]

                success, message = send_disaster_alert(
                    sender_email=sender_email,
                    sender_password=sender_password,
                    receiver_email=receiver_email,
                    disaster_type=class_display_names.get(
                        result["final_prediction"],
                        result["final_prediction"]
                    ),
                    final_confidence=result["final_confidence"],
                    image_prediction=class_display_names.get(
                        result["image_prediction"],
                        result["image_prediction"]
                    ),
                    image_confidence=result["image_confidence"],
                    text_prediction=result["text_prediction"],
                    text_confidence=result["text_confidence"],
                    user_text=text_input
                )

                if success:
                    st.success("📧 Email alert sent successfully to authorities.")
                else:
                    st.error("❌ Email alert failed.")
                    st.code(message)

            except Exception as e:
                st.error("❌ Email configuration error.")
                st.write("Check `.streamlit/secrets.toml` file.")
                st.code(str(e))

        else:
            st.success("✅ No automatic alert required. Manual verification may still be needed.")

            if result["final_prediction"] == "Possible_Disaster_Text_Only":
                st.warning(
                    "⚠️ Text strongly indicates disaster, but image evidence is weak. "
                    "Manual verification is recommended."
                )

            elif result["final_prediction"] == "Uncertain":
                st.info(
                    "ℹ️ System confidence is low. This case should be manually reviewed."
                )

        with st.expander("View Detailed Output"):
            st.json(result)