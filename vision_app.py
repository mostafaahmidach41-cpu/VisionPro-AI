import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
from ultralytics import YOLO
import numpy as np
import time

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="VisionPro AI | Performance Terminal",
    page_icon="👁️",
    layout="wide"
)

# --- 2. Custom Styling & Branding ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { 
        background-color: #161b22; 
        padding: 20px; 
        border-radius: 12px; 
        border: 1px solid #30363d;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .css-1offfwp { font-weight: 700; color: #58a6ff; } /* Subheader color */
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 3. High-Performance Model Loading ---
@st.cache_resource
def load_optimized_model():
    # Using 'yolov8n.pt' (Nano) for maximum speed.
    # It will be downloaded automatically on first run.
    return YOLO("yolov8n.pt")

try:
    model = load_optimized_model()
except Exception as e:
    st.error(f"Neural Engine Error: {e}")

# --- 4. Advanced Video Processing Engine ---
class VisionTransformer(VideoProcessorBase):
    def __init__(self):
        self.model = model

    def recv(self, frame):
        # 1. Capture and convert frame from WebRTC (BGR)
        img = frame.to_ndarray(format="bgr24")

        # 2. Optimized Inference (Turning off verbose logging saves time)
        results = self.model(img, conf=0.55, iou=0.5, verbose=False)
        
        # 3. Dynamic Frame Annotation
        annotated_frame = results[0].plot()

        # 4. Extract Key Metadata
        for r in results:
            names = model.names
            classes = r.boxes.cls.numpy()
            
            # Count only persons for clarity (ID 0 in COCO)
            person_count = len([c for c in classes if names[int(c)] == 'person'])
            
            # Calculate average confidence if objects are detected
            if len(r.boxes) > 0:
                avg_conf = float(r.boxes.conf.mean() * 100)
            else:
                avg_conf = 0.0
                
            # Storing data to be consumed by main thread metrics
            st.session_state["p_count"] = person_count
            st.session_state["scan_acc"] = avg_conf

        return annotated_frame

# --- 5. Initializing Session States ---
if "p_count" not in st.session_state: st.session_state["p_count"] = 0
if "scan_acc" not in st.session_state: st.session_state["scan_acc"] = 0.0

# --- 6. Main Interface & Title ---
st.title("👁️ VisionPro AI - Neural Terminal")
st.markdown("Independent Real-time Object Detection v2.0")
st.divider()

col_stream, col_metrics = st.columns([1.8, 1])

with col_stream:
    st.subheader("📷 Visual Input Feed")
    
    # 7. WebRTC/STUN configuration for low-latency video on cloud
    RTC_CONFIG = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_streamer(
        key="vision-pro-v2",
        video_processor_factory=VisionTransformer,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_metrics:
    st.subheader("🧠 Live Neural Analytics")
    
    with st.container():
        m1, m2 = st.columns(2)
        m1.metric(
            label="Persons Detected", 
            value=f"{st.session_state['p_count']}",
            help="Number of people identified in the active frame."
        )
        m2.metric(
            label="Scan Accuracy", 
            value=f"{st.session_state['scan_acc']:.1f}%",
            help="Average confidence score of detected objects."
        )
        
    st.divider()
    
    with st.expander("System Standby / Information"):
        st.info("Terminal ready. Click 'Start' to begin neural processing.")
        st.caption("Environment: Streamlit Cloud | Python 3.11")
        st.caption("Core: YOLOv8 Nano | Optimization: Headless OpenCV")

# --- 8. Professional Footer ---
st.divider()
st.caption("VisionPro AI Enterprise | Architecture: YOLOv8 | Global Operational Status: ACTIVE")
