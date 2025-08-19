import streamlit as st
import cv2
from deepface import DeepFace
import numpy as np
from PIL import Image

def detect_emotion(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
        result = DeepFace.analyze(
    img_path="sample.jpg",
    actions=["emotion"],
    detector_backend="opencv",  # 👈 avoids RetinaFace/Keras issue
    enforce_detection=False
)

        if isinstance(result, list):
            result = result[0]
        emotion = result.get('dominant_emotion', 'Unknown')
        cv2.putText(frame_rgb, f"Emotion: {emotion}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    except:
        cv2.putText(frame_rgb, "No face detected", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    return frame_rgb

st.title("Real-time Emotion Detection")
run = st.checkbox("Start Camera")
FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture image. Make sure your camera is working.")
        break
    frame = detect_emotion(frame)
    FRAME_WINDOW.image(frame, channels='RGB')

cap.release()


