import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import time

# Page Configuration
st.set_page_config(page_title="AI Object Detector", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO("yolov5s.pt")

def main():
    st.title("AI Object Detector")
    st.sidebar.title("Configuration")
    
    mode = st.sidebar.selectbox(
        "Select Mode",
        ["About", "Real-time Webcam", "Upload Image", "Upload Video"]
    )
    
    detector = load_model()
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25)
    
    if mode == "About":
        st.info("Welcome to the AI Object Detector! Choose a mode from the sidebar to get started.")
        st.markdown("""
        ### Features:
        - **Real-time Webcam**: Live detection using your camera.
        - **Upload Image**: Structured detection results for static images.
        - **Upload Video**: Frame-by-frame processing for video files.
        """)

    elif mode == "Upload Image":
        st.subheader("🖼️ Image Detection")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            with st.spinner('Detecting...'):
                results = detector.predict(source=img_array, conf=conf_threshold)
                res_plotted = results[0].plot()
                
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", use_container_width=True)
            with col2:
                st.image(res_plotted, caption="Detected Objects", use_container_width=True)
            
            # Display Detections properly
            st.subheader("Detection Summary")
            detections = results[0].boxes
            if len(detections) > 0:
                for box in detections:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = detector.names[cls]
                    st.write(f"✅ Found **{name}** with **{conf:.2%}** confidence")
            else:
                st.write("No objects detected.")

    elif mode == "Upload Video":
        st.subheader("📽️ Video Detection")
        uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])
        
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            
            vf = cv2.VideoCapture(tfile.name)
            st_frame = st.empty()
            
            stop_btn = st.button("Stop Processing")
            
            while vf.isOpened():
                ret, frame = vf.read()
                if not ret or stop_btn:
                    break
                
                # Process frame
                results = detector.predict(source=frame, conf=conf_threshold, verbose=False)
                res_plotted = results[0].plot()
                
                # Display output
                st_frame.image(res_plotted, channels="BGR", use_container_width=True)
                
            vf.release()
            st.success("Video processing finished.")

    elif mode == "Real-time Webcam":
        st.subheader("🎥 Live Webcam Detection")
        run = st.checkbox('Start Webcam')
        st_frame = st.empty()
        
        if run:
            cap = cv2.VideoCapture(0)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("Could not access webcam.")
                    break
                
                # YOLOv5 inference
                results = detector.predict(source=frame, conf=conf_threshold, verbose=False)
                res_plotted = results[0].plot()
                
                st_frame.image(res_plotted, channels="BGR", use_container_width=True)
                
                if not run:
                    break
            cap.release()
        else:
            st.write("Click the checkbox to start the camera.")

if __name__ == "__main__":
    main()
