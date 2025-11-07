# real_time_object_detection_app.py

import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import tempfile

# --------------------------------
# Page Configuration
# --------------------------------
st.set_page_config(page_title="Real-Time Object Detection", layout="wide")

# --------------------------------
# App Header
# --------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 40px;
        color: #ffffff;
        background: linear-gradient(90deg, #2a2a72, #009ffd);
        padding: 0.6em;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    </style>
    <div class="main-title">🕵️ Real-Time Object Detection with YOLOv8</div>
    """,
    unsafe_allow_html=True
)

st.write("Detect objects in **images, videos**, or even through your **webcam**, powered by the pre-trained YOLOv8 model!")

# --------------------------------
# Load YOLO Model
# --------------------------------
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")  # pre-trained on COCO dataset
    return model

model = load_model()

# --------------------------------
# Sidebar Controls
# --------------------------------
st.sidebar.header("⚙️ Detection Settings")
source_type = st.sidebar.radio("Select Input Source", ["Upload Image", "Upload Video", "Webcam"])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
st.sidebar.info("Adjust confidence to filter weaker detections.\n\nLower = more boxes, higher = stricter filtering.")

# --------------------------------
# Detection Function
# --------------------------------
def detect_objects(frame):
    results = model.predict(frame, conf=confidence_threshold)
    annotated_frame = results[0].plot()  # returns annotated numpy array
    return annotated_frame

# --------------------------------
# Image Upload Section
# --------------------------------
if source_type == "Upload Image":
    st.subheader("📷 Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        frame = np.array(img)
        annotated_img = detect_objects(frame)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original Image", use_container_width=True)
        with col2:
            st.image(annotated_img, caption="Detected Objects", use_container_width=True)

# --------------------------------
# Video Upload Section
# --------------------------------
elif source_type == "Upload Video":
    st.subheader("🎞️ Upload a Video")
    uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated_frame = detect_objects(frame_rgb)
            stframe.image(annotated_frame, channels="RGB", use_container_width=True)
        cap.release()

# --------------------------------
# Webcam Section
# --------------------------------
elif source_type == "Webcam":
    st.subheader("🎥 Live Webcam Detection")
    st.info("⚠️ Note: Webcam streaming works best when running Streamlit locally.")
    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Could not access webcam.")
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated_frame = detect_objects(frame_rgb)
        FRAME_WINDOW.image(annotated_frame, channels="RGB", use_container_width=True)

    cap.release()

# --------------------------------
# Footer
# --------------------------------
st.markdown(
    """
    ---
    <div style="text-align: center; color: gray; font-size: 0.9em;">
    Built with ❤️ using Streamlit and YOLOv8 • Powered by Ultralytics
    </div>
    """,
    unsafe_allow_html=True
)
