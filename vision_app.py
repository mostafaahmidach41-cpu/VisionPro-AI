import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
from ultralytics import YOLO
import numpy as np

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="VisionPro AI | Performance Terminal",
    page_icon="👁️",
    layout="wide"
)

# --- 2. Advanced CSS for Professional UI ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { 
        background-color: #161b22; 
        padding: 20px; 
        border-radius: 12px; 
        border: 1px solid #30363d;
    }
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 3. High-Performance Model Loading ---
@st.cache_resource
def load_optimized_model():
    # Using 'yolov8n.pt' (Nano) - The fastest YOLOv8 model for cloud deployment
    return YOLO("yolov8n.pt")

model = load_optimized_model()

# --- 4. Optimized Neural Engine ---
class VisionTransformer(VideoProcessorBase):
    def __init__(self):
        self.model = model

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # ENHANCEMENT: verbose=False reduces overhead and increases FPS
        # ENHANCEMENT: imgsz=320 speeds up processing while maintaining accuracy for people
        results = self.model(img, conf=0.55, verbose=False, imgsz=320)
        
        # Draw detection boxes
        annotated_frame = results[0].plot()

        return annotated_frame

# --- 5. Main Terminal UI ---
st.title("👁️ VisionPro AI - Neural Terminal")
st.markdown("Independent Real-time Object Detection v2.0")
st.divider()

col_stream, col_analytics = st.columns([2, 1])

with col_stream:
    st.subheader("📷 Visual Input Feed")
    
    # RTC configuration for low-latency streaming
    RTC_CONFIG = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_streamer(
        key="vision-v2",
        video_processor_factory=VisionTransformer,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True, # Critical for speed
    )

with col_analytics:
    st.subheader("🧠 Live Neural Analytics")
    
    # The image shows the new analytics dashboard
    st.info("System Standby: Click 'Start' to activate neural scan.")
    
    with st.expander("Technical Specifications"):
        st.write("**Engine:** YOLOv8 Nano (v8.1+)")
        st.write("**Optimization:** Asynchronous Frame Processing")
        st.write("**Deployment:** Streamlit Cloud (Python 3.11)")

# --- 6. Professional Footer ---
st.divider()
st.caption("VisionPro AI Enterprise | Architecture: YOLOv8 | Status: ACTIVE")
