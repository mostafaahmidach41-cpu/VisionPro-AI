import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
from ultralytics import YOLO
import numpy as np
from datetime import datetime
import pandas as pd
import os

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="VisionPro AI | Neural Terminal", 
    page_icon="👁️", 
    layout="wide"
)

# --- 2. Directory & State Management ---
if not os.path.exists("captures"):
    os.makedirs("captures")

if "detection_log" not in st.session_state:
    st.session_state["detection_log"] = []
if "last_capture" not in st.session_state:
    st.session_state["last_capture"] = None

# --- 3. Neural Engine Model Loading ---
@st.cache_resource
def load_optimized_model():
    # Loading YOLOv8 Nano for high-speed inference
    return YOLO("yolov8n.pt")

model = load_optimized_model()

# --- 4. Video Processing Logic ---
class VisionTransformer(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.last_capture_time = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Inference with speed optimization (imgsz=320)
        results = self.model(img, conf=0.55, verbose=False, imgsz=320)
        
        # Identify person class
        persons = [r for r in results[0].boxes.cls if int(r) == 0]
        
        annotated_frame = results[0].plot()

        if len(persons) > 0:
            current_time = datetime.now()
            # 10 seconds interval between captures
            if (current_time.timestamp() - self.last_capture_time) > 10:
                self.last_capture_time = current_time.timestamp()
                
                # File handling
                timestamp_str = current_time.strftime("%Y%m%d_%H%M%S")
                file_path = f"captures/detect_{timestamp_str}.jpg"
                cv2.imwrite(file_path, annotated_frame)
                
                # Session State Updates
                st.session_state["last_capture"] = file_path
                log_entry = {
                    "Time": current_time.strftime("%H:%M:%S"), 
                    "Event": "Person Detected",
                    "Status": "System Alert"
                }
                st.session_state["detection_log"].append(log_entry)

        return frame.from_ndarray(annotated_frame, format="bgr24")

# --- 5. Sidebar - Configuration ---
st.sidebar.header("System Configuration")
stream_type = st.sidebar.radio("Select Source:", ("Webcam", "IP Camera (RTSP)"))

if stream_type == "IP Camera (RTSP)":
    rtsp_link = st.sidebar.text_input(
        "RTSP URL:", 
        placeholder="rtsp://admin:password@ip_address:554/stream"
    )
    st.sidebar.warning("Note: Cloud deployment requires public RTSP access.")

# --- 6. Main Interface Layout ---
st.title("👁️ VisionPro AI - Neural Terminal")
st.markdown("v2.4 | Enterprise Restaurant Monitoring System")
st.divider()

col_main, col_stats = st.columns([2, 1])

with col_main:
    st.subheader("📷 Neural Scan Feed")
    
    # Global RTC Configuration to fix STUN/TURN errors
    RTC_CONFIG = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
    )

    webrtc_streamer(
        key="vision-monitor",
        video_processor_factory=VisionTransformer,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

with col_stats:
    st.subheader("🖼️ Latest Capture")
    if st.session_state["last_capture"]:
        st.image(st.session_state["last_capture"], use_container_width=True)
    else:
        st.info("System Ready. Waiting for detection...")
    
    st.divider()
    st.subheader("📝 Activity Log")
    if st.session_state["detection_log"]:
        # Show latest 10 logs
        df = pd.DataFrame(st.session_state["detection_log"]).iloc[::-1]
        st.table(df.head(10))
    
    if st.button("Clear Logs"):
        st.session_state["detection_log"] = []
        st.session_state["last_capture"] = None
        st.rerun()

# --- 7. Footer ---
st.divider()
st.caption("VisionPro AI Enterprise | Engine: YOLOv8 | Status: OPERATIONAL")
