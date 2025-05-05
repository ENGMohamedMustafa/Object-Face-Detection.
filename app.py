import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.set_page_config(layout="wide")
st.title("👁️ Real-Time Face Detection using Mediapipe")

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

run = st.checkbox('Start Webcam')
FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while run:
        success, image = camera.read()
        if not success:
            st.write("Failed to capture image")
            break

        image = cv2.flip(image, 1)
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)

        FRAME_WINDOW.image(image, channels="BGR")

    else:
        st.write("Stopped.")
