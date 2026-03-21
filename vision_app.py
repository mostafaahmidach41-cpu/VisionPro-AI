import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
from ultralytics import YOLO
import numpy as np
from datetime import datetime
import pandas as pd
import os

# --- 1. Page Configuration ---
st.set_page_config(page_title="VisionPro AI | Auto-Capture", page_icon="📸", layout="wide")

# --- 2. Initialize Directories & Session State ---
if not os.path.exists("captures"):
    os.makedirs("captures")

if "detection_log" not in st.session_state:
    st.session_state["detection_log"] = []
if "last_capture" not in st.session_state:
    st.session_state["last_capture"] = None

# --- 3. Optimized Model Loading ---
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# --- 4. Neural Engine with Auto-Capture Logic ---
class VisionTransformer(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.last_capture_time = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.model(img, conf=0.60, verbose=False, imgsz=320)
        
        persons = [r for r in results[0].boxes.cls if int(r) == 0]
        
        if len(persons) > 0:
            current_time = datetime.now()
            # Capture logic: once every 10 seconds to save storage
            if (current_time.timestamp() - self.last_capture_time) > 10:
                self.last_capture_time = current_time.timestamp()
                
                # Save the frame as an image
                timestamp_str = current_time.strftime("%Y%m%d_%H%M%S")
                file_path = f"captures/detect_{timestamp_str}.jpg"
                cv2.imwrite(file_path, results[0].plot())
                
                # Update UI logs
                st.session_state["last_capture"] = file_path
                log_entry = {
                    "Time": current_time.strftime("%H:%M:%S"), 
                    "Event": "📸 Image Captured",
                    "File": file_path
                }
                st.session_state["detection_log"].append(log_entry)

        return results[0].plot()

# --- 5. User Interface Layout ---
st.title("👁️ VisionPro AI - Neural Terminal")
st.markdown("v2.3 | Live Neural Scan & Auto-Capture Environment")
st.divider()

col_main, col_sidebar = st.columns([2, 1])

with col_main:
    st.subheader("📷 Live Neural Stream")
    webrtc_streamer(
        key="vision-capture",
        video_processor_factory=VisionTransformer,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

with col_sidebar:
    st.subheader("🖼️ Latest Detection")
    if st.session_state["last_capture"]:
        st.image(st.session_state["last_capture"], caption="Last Recorded Person")
    else:
        st.info("No detections captured yet.")
    
    st.divider()
    st.subheader("📝 Activity Log")
    if st.session_state["detection_log"]:
        df = pd.DataFrame(st.session_state["detection_log"]).iloc[::-1]
        st.dataframe(df, hide_index=True)
    
    if st.button("Reset Terminal"):
        st.session_state["detection_log"] = []
        st.session_state["last_capture"] = None
        st.rerun()

st.divider()
st.caption("VisionPro AI Enterprise | Architecture: YOLOv8 | Status: OPERATIONAL")
