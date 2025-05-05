import streamlit as st
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import numpy as np

st.set_page_config(page_title="Object & Face Detection App", layout="wide")
st.title("🧠 Object & Face Detection App (Streamlit Friendly)")

option = st.radio("Choose Input Type:", ("📷 Upload Image", "📹 Use Webcam"))

def process_image(img):
    bbox, label, conf = cv.detect_common_objects(img)
    output_img = draw_bbox(img, bbox, label, conf)
    return output_img, label

if option == "📷 Upload Image":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(image, channels="BGR", caption="Original Image", use_column_width=True)

        output_img, detected = process_image(image)
        st.image(output_img, channels="BGR", caption="Detected Objects", use_column_width=True)
        st.success(f"Detected: {', '.join(detected)}")

else:
    run = st.checkbox('Start Webcam')

    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        _, frame = camera.read()
        frame = cv2.flip(frame, 1)
        output_img, _ = process_image(frame)
        FRAME_WINDOW.image(output_img, channels="BGR")

    else:
        st.write('Webcam stopped.')
