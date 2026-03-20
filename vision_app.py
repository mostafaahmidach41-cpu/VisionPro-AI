import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
from ultralytics import YOLO
import numpy as np

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="VisionPro AI | Live Neural Terminal",
    page_icon="👁️",
    layout="wide"
)

# --- 2. Custom UI Styling ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { 
        background-color: #161b22; 
        padding: 15px; 
        border-radius: 10px; 
        border: 1px solid #30363d; 
    }
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 3. Model Initialization ---
@st.cache_resource
def load_yolo_model():
    # Automatically downloads yolov8n.pt if not present in the repository
    return YOLO("yolov8n.pt")

model = load_yolo_model()

# --- 4. Neural Processing Engine ---
class VideoTransformer(VideoProcessorBase):
    def __init__(self):
        self.model = model

    def recv(self, frame):
        # Convert incoming WebRTC frame to numpy array (BGR format)
        img = frame.to_ndarray(format="bgr24")

        # Run AI Inference (YOLOv8)
        results = self.model(img, conf=0.5, verbose=False)
        
        # Annotate Frame with detection boxes and labels
        annotated_frame = results[0].plot()

        return annotated_frame

# --- 5. Main Interface ---
st.title("👁️ VisionPro AI - Neural Terminal")
st.markdown("### Real-time Live Object Detection Pipeline")
st.divider()

col_stream, col_info = st.columns([2, 1])

with col_stream:
    st.subheader("📷 Live Neural Scan")
    
    # WebRTC configuration for STUN servers (Crucial for Cloud Deployment)
    RTC_CONFIG = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_streamer(
        key="vision-pro-live",
        video_processor_factory=VideoTransformer,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_info:
    st.subheader("🧠 System Analytics")
    st.info("System is ready. Click 'Start' to begin live neural processing.")
    
    # Expandable technical specifications based on your current build
    with st.expander("Technical Specifications"):
        st.write("**Architecture:** YOLOv8 Nano")
        st.write("**Framework:** Streamlit + WebRTC")
        st.write("**Environment:** Python 3.11")
        st.write("**Status:** Operational")

# --- 6. Footer ---
st.divider()
st.caption("VisionPro AI Enterprise | Architecture: YOLOv8 | Global Operational Status: ACTIVE")
