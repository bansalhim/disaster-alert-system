# 🚨 Multimodal Disaster Alert System with Automated Emergency Notification

A deep learning based disaster alert system that uses both image input and text input to detect possible disaster situations and automatically send an email alert to authorities when high-confidence disaster evidence is found.

---

## 📌 Project Overview

This project combines Computer Vision and Natural Language Processing to build a multimodal disaster detection system.

The system takes:

- Disaster-related image input
- Text report / tweet / emergency message

Then it predicts:

- Image-based disaster category
- Text-based disaster probability
- Final fused disaster result
- Whether an automatic alert should be sent

If the final result shows a high-confidence disaster, the system sends an automated email notification to the configured authority email.

---

## 🔁 System Flow

```text
Image + Text Input
        ↓
Image Model: ResNet50
        ↓
Text Model: DistilBERT
        ↓
Fusion Logic
        ↓
Disaster Type + Confidence
        ↓
Streamlit Dashboard
        ↓
Email Alert to Authorities
```

---

## 🧠 Models Used

### 1. Image Model

The image model is based on ResNet50 and classifies disaster images into six categories:

```text
Damaged_Infrastructure
Fire_Disaster
Human_Damage
Land_Disaster
Non_Damage
Water_Disaster
```

The trained image model is saved as:

```text
models/best_resnet50_model.keras
```

Class labels are saved in:

```text
models/class_names.json
```

---

### 2. Text Model

The text model is based on DistilBERT and is trained on the Kaggle Disaster Tweets dataset.

It performs binary classification:

```text
1 → Disaster
0 → Not Disaster
```

The trained text model is saved inside:

```text
models/text_model_distilbert/
```

---

## 🔀 Fusion Logic

The fusion logic combines the output of both models.

Example:

```text
Image Model:
Water_Disaster = 92%

Text Model:
Disaster = 96%

Final Result:
Water Disaster / Flood

Alert Required:
YES
```

If the image confidence is low but the text strongly indicates disaster, the system marks it as:

```text
Possible Disaster Based on Text Only
```

and recommends manual verification instead of sending an automatic alert.

---

## 📧 Email Alert Feature

If the final confidence crosses the alert threshold, the system automatically sends an email alert containing:

- Final disaster prediction
- Final confidence score
- Image model prediction
- Image confidence
- Text model prediction
- Text confidence
- User-entered emergency message

Gmail SMTP is used for sending alerts.

> Gmail requires a Google App Password. Normal Gmail password will not work.

---

## 🗂️ Project Structure

```text
disaster-alert-system/
├── app.py
├── email_alert.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── secrets.toml
├── models/
│   ├── best_resnet50_model.keras
│   ├── class_names.json
│   └── text_model_distilbert/
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── vocab.txt
├── notebooks/
│   ├── image_model.ipynb
│   ├── text_model_distilbert.ipynb
│   └── fusion_model.ipynb
└── sample_inputs/
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/disaster-alert-system.git
cd disaster-alert-system
```

### 2. Create virtual environment

```bash
python -m venv .venv
```

### 3. Activate virtual environment

For Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

For CMD:

```bash
.venv\Scripts\activate
```

For macOS/Linux:

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

```text
streamlit==1.41.1
tensorflow==2.20.0
torch
transformers==4.44.2
tokenizers==0.19.1
huggingface-hub==0.24.6
safetensors
pillow
numpy
```

---

## 🔐 Email Configuration

Create a folder:

```text
.streamlit/
```

Inside it, create:

```text
secrets.toml
```

Add this:

```toml
[email]
sender_email = "your_email@gmail.com"
sender_password = "your_16_character_google_app_password"
receiver_email = "authority_email@gmail.com"
```

Important:

- Turn on Google 2-Step Verification.
- Generate a Gmail App Password.
- Use the App Password, not your normal Gmail password.
- Do not upload `secrets.toml` to GitHub.

---

## ▶️ Run the Application

Run this command:

```bash
streamlit run app.py
```

The app will open at:

```text
http://localhost:8501
```

---

## 🖥️ Streamlit Dashboard Features

The dashboard allows users to:

- Upload disaster image
- Enter text / tweet / emergency message
- View image model prediction
- View text model prediction
- View final fused prediction
- View confidence scores
- Check whether alert is required
- Send automatic email alert if disaster confidence is high

---

## 🧪 Example Inputs

### Example 1: Flood Case

Image:

```text
Flooded street / waterlogging image
```

Text:

```text
Heavy rainfall has caused flooding in the area and people need urgent help.
```

Expected result:

```text
Final Prediction: Water Disaster / Flood
Alert Required: YES
```

---

### Example 2: Normal Case

Image:

```text
Normal house / clean street image
```

Text:

```text
This is a beautiful house near the beach.
```

Expected result:

```text
Final Prediction: No Visible Disaster
Alert Required: NO
```

---

### Example 3: Text-only Disaster Suspicion

Image:

```text
Normal image
```

Text:

```text
Flood came in Delhi and people need help.
```

Expected result:

```text
Final Prediction: Possible Disaster Based on Text Only
Alert Required: NO
Manual Verification Recommended
```

---

## 📊 Model Results

### Image Model

The ResNet50 image model achieved approximately:

```text
Validation Accuracy: 93%
```

It performed best among tested image architectures such as:

- AlexNet-style model
- MobileNetV2
- EfficientNet
- Vision Transformer
- ResNet50

---

### Text Model

The DistilBERT text model achieved approximately:

```text
Test Accuracy: 85%
```

It was trained on the Kaggle Disaster Tweets dataset and detects whether a text input indicates a real disaster or not.

---

## 📁 Datasets Used

### Image Dataset

A comprehensive disaster image dataset containing categories such as:

```text
Damaged Infrastructure
Fire Disaster
Human Damage
Land Disaster
Non Damage
Water Disaster
```

### Text Dataset

Kaggle Disaster Tweets dataset containing tweets labelled as:

```text
1 → Real Disaster
0 → Not Disaster
```

---

## 🚨 Alert Decision Rule

An email alert is sent only when:

```text
Image model confidence is strong
+
Text model confirms disaster
+
Final fused confidence crosses threshold
```

This reduces false alerts caused by image-only or text-only uncertainty.

---

## 🔒 Security Note

Do not push sensitive files to GitHub.

Add this to `.gitignore`:

```gitignore
.streamlit/secrets.toml
.venv/
__pycache__/
*.pyc
```

---

## 🚀 Future Improvements

Possible future enhancements:

- Add GPS/location input
- Send SMS alerts using Twilio
- Add map-based disaster visualization
- Add real-time social media monitoring
- Add admin dashboard for emergency authorities
- Deploy on Streamlit Cloud
- Add multilingual disaster text support
- Improve image model with larger disaster datasets

---

## 👨‍💻 Tech Stack

```text
Python
Streamlit
TensorFlow / Keras
PyTorch
Transformers
DistilBERT
ResNet50
Gmail SMTP
```

---

## ✅ Current Status

```text
Image Model Training       ✅ Completed
Text Model Training        ✅ Completed
Fusion Logic               ✅ Completed
Streamlit Dashboard        ✅ Completed
Automated Email Alert      ✅ Completed
```

---

## 📌 Disclaimer

This system is built for academic and prototype purposes. It should not be used as the sole source for real-world emergency response decisions without human verification.
