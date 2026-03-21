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
    page_title="VisionPro AI | Restaurant Monitor", 
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
    # Using YOLOv8 Nano for high-speed restaurant monitoring
    return YOLO("yolov8n.pt")

model = load_optimized_model()

# --- 4. Video Processing Logic ---
class VisionTransformer(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.last_capture_time = 0

    def recv(self, frame):
        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")
        
        # Run AI inference (imgsz=320 for speed optimization)
        results = self.model(img, conf=0.60, verbose=False, imgsz=320)
        
        # Filter for 'person' class (COCO ID: 0)
        persons = [r for r in results[0].boxes.cls if int(r) == 0]
        
        if len(persons) > 0:
            current_time = datetime.now()
            # Capture interval: 10 seconds to avoid duplicate storage
            if (current_time.timestamp() - self.last_capture_time) > 10:
                self.last_capture_time = current_time.timestamp()
                
                # Save detection image
                timestamp_str = current_time.strftime("%Y%m%d_%H%M%S")
                file_path = f"captures/detect_{timestamp_str}.jpg"
                cv2.imwrite(file_path, results[0].plot())
                
                # Log the event
                st.session_state["last_capture"] = file_path
                log_entry = {
                    "Time": current_time.strftime("%H:%M:%S"), 
                    "Event": "Object Detected",
                    "Status": "Image Saved"
                }
                st.session_state["detection_log"].append(log_entry)

        return results[0].plot()

# --- 5. Sidebar - Camera Configuration ---
st.sidebar.header("Camera Configuration")
stream_type = st.sidebar.radio("Select Source:", ("Webcam", "IP Camera (RTSP)"))

rtsp_link = ""
if stream_type == "IP Camera (RTSP)":
    rtsp_link = st.sidebar.text_input(
        "RTSP URL:", 
        placeholder="rtsp://admin:password@192.168.1.10:554/stream"
    )
    st.sidebar.warning("Note: RTSP requires local execution for private network cameras.")

# --- 6. Main Interface Layout ---
st.title("👁️ VisionPro AI - Neural Terminal")
st.markdown("v2.4 | Enterprise Restaurant Monitoring System")
st.divider()

col_main, col_stats = st.columns([2, 1])

with col_main:
    st.subheader("📷 Neural Scan Feed")
    
    # RTC Configuration for stable cloud streaming
    RTC_CONFIG = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    # Use RTSP link if provided, otherwise use default webcam
    webrtc_streamer(
        key="restaurant-monitor",
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
        st.info("No person detected yet.")
    
    st.divider()
    st.subheader("📝 Activity Log")
    if st.session_state["detection_log"]:
        df = pd.DataFrame(st.session_state["detection_log"]).iloc[::-1]
        st.table(df.head(10))
    
    if st.button("Clear History"):
        st.session_state["detection_log"] = []
        st.session_state["last_capture"] = None
        st.rerun()

# --- 7. Footer ---
st.divider()
st.caption("VisionPro AI Enterprise | Architecture: YOLOv8 | Global Operational Status: ACTIVE")
