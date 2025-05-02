import streamlit as st
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.preprocessing import image
from transformers import pipeline
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, LeakyReLU
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import uuid
import os

IMGWIDTH = 256

class Classifier:
    def __init__(self):
        self.model = None

    def predict(self, x):
        return self.model.predict(x)

    def fit(self, x, y):
        return self.model.train_on_batch(x, y)

    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)

    def load(self, path):
        self.model.load_weights(path)

class Meso4(Classifier):
    def __init__(self, learning_rate=0.001):
        self.model = self.init_model()
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

    def init_model(self):
        x = Input(shape=(IMGWIDTH, IMGWIDTH, 3))
        x1 = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
        x2 = Conv2D(8, (5, 5), padding='same', activation='relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)
        x3 = Conv2D(16, (5, 5), padding='same', activation='relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
        x4 = Conv2D(16, (5, 5), padding='same', activation='relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)
        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation='sigmoid')(y)
        return Model(inputs=x, outputs=y)

@st.cache_resource
def load_meso_classifier():
    classifier = Meso4()
    classifier.load("Meso4_DF.h5")
    return classifier

@st.cache_resource
def load_hf_pipeline():
    return pipeline('image-classification', model="prithivMLmods/Deep-Fake-Detector-Model", device=-1)

meso_classifier = load_meso_classifier()
hf_pipeline = load_hf_pipeline()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def load_and_preprocess_image(img_pil, target_size=(IMGWIDTH, IMGWIDTH)):
    img = img_pil.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array, img

st.markdown("""
    <style>
    @import url('https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css');
    .main {
        background: linear-gradient(to bottom, #1a202c, #2d3748);
        color: white;
        padding: 2rem;
        min-height: 100vh;
    }
    .sidebar .sidebar-content {
        background-color: #2d3748;
        color: white;
    }
    .stButton>button {
        background-color: #4a5568;
        color: white;
        border-radius: 0.375rem;
        padding: 0.5rem 1rem;
        transition: background-color 0.2s;
    }
    .stButton>button:hover {
        background-color: #718096;
    }
    .stFileUploader {
        background-color: #4a5568;
        border-radius: 0.375rem;
        padding: 1rem;
    }
    h1, h2, h3 {
        color: #e2e8f0;
        font-weight: bold;
    }
    .result-box {
        background-color: #2d3748;
        border-radius: 0.375rem;
        padding: 1rem;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("DeepFake Detector")
page = st.sidebar.selectbox("Select Mode", ["🖼️ Image Detection", "🎥 Video Detection"])

if page == "🖼️ Image Detection":
    st.title("🖼️ DeepFake Image Detector")
    st.markdown("Upload an image to detect if it's real or a deepfake using advanced AI models.")

    uploaded_img = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'], key="image_uploader")

    if uploaded_img:
        img = Image.open(uploaded_img).convert("RGB")
        img_np = np.array(img)
        processed_image, processed_image_pil = load_and_preprocess_image(img)

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            st.warning("No faces detected. Please upload an image with a visible face.")
        else:
            x, y, w, h = faces[0]
            face_img = img.crop((x, y, x + w, y + h))
            face_array, face_pil = load_and_preprocess_image(face_img)

            meso_pred = meso_classifier.predict(face_array)
            meso_prob = meso_pred[0][0]
            meso_label = 1 if meso_prob > 0.5 else 0

            prithiv_pred = hf_pipeline(face_pil)
            prithiv_prob = prithiv_pred[0]['score']
            prithiv_label = 1 if prithiv_pred[0]['label'].lower() == 'real' else 0

            combined_prob = (meso_prob + prithiv_prob) / 2
            final_label = 1 if combined_prob > 0.5 else 0

            num_to_label = {1: "Real", 0: "Fake"}

            st.image(img_np, caption="Uploaded Image with Detected Face", use_column_width=True)
            cv2.rectangle(img_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
            st.image(img_np, caption="Face Detection", use_column_width=True)

            st.markdown("### Results")
            st.markdown(f"<div class='result-box'>MesoNet: {num_to_label[meso_label]} (Probability: {meso_prob:.2f})</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-box'>Prithiv Model: {num_to_label[prithiv_label]} (Probability: {prithiv_prob:.2f})</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-box'>Combined: {num_to_label[final_label]} (Probability: {combined_prob:.2f})</div>", unsafe_allow_html=True)

elif page == "🎥 Video Detection":
    st.title("🎥 DeepFake Video Detector")
    st.markdown("Upload a video to analyze faces for deepfake detection frame by frame.")

    uploaded_vid = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'], key="video_uploader")

    if uploaded_vid:
        video_path = f"temp_video_{uuid.uuid4()}.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_vid.read())

        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()
        predictions = []
        confidences = []
        frame_count = 0

        st.info("Processing video. This may take some time...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            for (x, y, w, h) in faces:
                face_crop = frame[y:y + h, x:x + w]
                face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                face_array, face_pil_processed = load_and_preprocess_image(face_pil)

                meso_pred = meso_classifier.predict(face_array)
                meso_prob = meso_pred[0][0]
                meso_label = 1 if meso_prob > 0.5 else 0

                prithiv_pred = hf_pipeline(face_pil_processed)
                prithiv_prob = prithiv_pred[0]['score']
                prithiv_label = 1 if prithiv_pred[0]['label'].lower() == 'real' else 0

                combined_prob = (meso_prob + prithiv_prob) / 2
                final_label = 1 if combined_prob > 0.5 else 0

                predictions.append(1 if final_label == 0 else 0)
                confidences.append(combined_prob)

                display_label = "Fake" if final_label == 0 else "Real"
                color = (0, 0, 255) if final_label == 0 else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{display_label} ({combined_prob:.2f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            stframe.image(frame, channels="BGR", use_column_width=True)

        cap.release()
        os.remove(video_path)

        st.success("✅ Video Analysis Complete")

        if predictions:
            avg_prediction = np.mean(predictions)
            final_verdict = "🔴 Likely FAKE" if avg_prediction > 0.5 else "🟢 Most Likely REAL"

            fig, axs = plt.subplots(1, 2, figsize=(14, 5))
            axs[0].plot(predictions, marker='o', color='blue')
            axs[0].set_title("Frame-wise Prediction (0=Real, 1=Fake)")
            axs[0].set_xlabel("Frame")
            axs[0].set_ylabel("Prediction")
            axs[0].grid(True)

            axs[1].plot(confidences, marker='x', color='orange')
            axs[1].set_title("Confidence per Frame")
            axs[1].set_xlabel("Frame")
            axs[1].set_ylabel("Confidence")
            axs[1].grid(True)

            plt.tight_layout()
            st.pyplot(fig)

            st.markdown("### 🧠 Final Verdict")
            st.markdown(f"<div class='result-box'>**Result:** {final_verdict}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-box'>**Fake Probability (avg):** {avg_prediction:.2f}</div>", unsafe_allow_html=True)
        else:
            st.warning("No faces detected in the video.")