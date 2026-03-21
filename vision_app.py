import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
from ultralytics import YOLO
import numpy as np
from streamlit_js_eval import streamlit_js_eval
from datetime import datetime
import pandas as pd

# --- 1. Page Configuration ---
st.set_page_config(page_title="VisionPro AI | Activity Log", page_icon="📝", layout="wide")

# --- 2. Session State Initialization ---
if "detection_log" not in st.session_state:
    st.session_state["detection_log"] = []
if "person_found" not in st.session_state:
    st.session_state["person_found"] = False

# --- 3. Optimized Model Loading ---
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# --- 4. Neural Engine with Logging Logic ---
class VisionTransformer(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.last_log_time = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.model(img, conf=0.55, verbose=False, imgsz=320)
        
        persons = [r for r in results[0].boxes.cls if int(r) == 0]
        
        if len(persons) > 0:
            st.session_state["person_found"] = True
            # Log detection every 5 seconds to avoid flooding the log
            current_time = datetime.now()
            if (current_time.second % 5 == 0): 
                log_entry = {"Timestamp": current_time.strftime("%H:%M:%S"), "Status": "🚨 Person Detected"}
                if log_entry not in st.session_state["detection_log"][-1:]:
                    st.session_state["detection_log"].append(log_entry)
        else:
            st.session_state["person_found"] = False

        return results[0].plot()

# --- 5. User Interface ---
st.title("👁️ VisionPro AI - Neural Terminal")
st.markdown("v2.2 | Real-time Detection & Activity Logging")
st.divider()

col_stream, col_side = st.columns([2, 1])

with col_stream:
    RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    webrtc_streamer(
        key="vision-log",
        video_processor_factory=VisionTransformer,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

with col_side:
    st.subheader("📝 Live Activity Log")
    if st.session_state["detection_log"]:
        df = pd.DataFrame(st.session_state["detection_log"]).iloc[::-1] # Show latest first
        st.table(df.head(10)) # Display last 10 detections
    else:
        st.write("No activity recorded yet.")
    
    if st.button("Clear Log"):
        st.session_state["detection_log"] = []
        st.rerun()

st.divider()
st.caption("VisionPro AI Enterprise | Architecture: YOLOv8 | Global Operational Status: ACTIVE")
