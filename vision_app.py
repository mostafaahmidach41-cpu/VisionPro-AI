import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
from ultralytics import YOLO
import numpy as np

# --- 1. Page Config ---
st.set_page_config(page_title="VisionPro AI - Live", page_icon="👁️")

# --- 2. Neural Engine Loading ---
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# --- 3. Video Processing Class ---
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")

        # Run YOLOv8 Inference
        results = model(img, conf=0.5)
        
        # Annotate frame with detection boxes
        annotated_frame = results[0].plot()

        return annotated_frame

# --- 4. User Interface ---
st.title("👁️ VisionPro AI - Live Video Scan")
st.markdown("Real-time Neural Terminal for Independent Detection")

webrtc_streamer(
    key="vision-live",
    video_processor_factory=VideoProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

st.divider()
st.caption("Architecture: YOLOv8 | Processing: Real-time Video Stream")
