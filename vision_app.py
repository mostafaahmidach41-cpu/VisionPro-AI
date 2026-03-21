import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
from ultralytics import YOLO
import numpy as np
from streamlit_js_eval import streamlit_js_eval

# --- 1. Page Configuration ---
st.set_page_config(page_title="VisionPro AI | Audio Alert", page_icon="🔔", layout="wide")

# --- 2. Audio Alert Logic ---
# Link to a short notification sound (Beep)
BEEP_URL = "https://www.soundjay.com/buttons/beep-01a.mp3"

def play_alert():
    # Execute JS to play sound in the user's browser
    streamlit_js_eval(code=f"new Audio('{BEEP_URL}').play();")

# --- 3. Optimized Model Loading ---
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# --- 4. Neural Engine with Detection Logic ---
class VisionTransformer(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.alert_triggered = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Inference with speed optimizations
        results = self.model(img, conf=0.55, verbose=False, imgsz=320)
        
        # Check for 'person' class (ID 0 in COCO)
        persons = [r for r in results[0].boxes.cls if int(r) == 0]
        
        if len(persons) > 0 and not self.alert_triggered:
            # Mark that a person was found to trigger the sound in the main thread
            st.session_state["person_found"] = True
            self.alert_triggered = True
        elif len(persons) == 0:
            self.alert_triggered = False
            st.session_state["person_found"] = False

        return results[0].plot()

# --- 5. User Interface ---
st.title("👁️ VisionPro AI - Neural Terminal")
st.markdown("v2.1 | Real-time Detection with Audio Alerts")
st.divider()

if st.session_state.get("person_found", False):
    st.warning("⚠️ Person Detected!")
    play_alert() # Trigger the audio via JS

col_stream, col_analytics = st.columns([2, 1])

with col_stream:
    RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    webrtc_streamer(
        key="vision-audio",
        video_processor_factory=VisionTransformer,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

with col_analytics:
    st.subheader("🧠 Live Neural Analytics")
    st.info("System Standby: Audio alerts are ACTIVE when a person is detected.")
    
    with st.expander("Technical Specs"):
        st.write("**Audio Engine:** Client-side JavaScript")
        st.write("**Detection Engine:** YOLOv8 Nano")

st.divider()
st.caption("VisionPro AI Enterprise | Global Operational Status: ACTIVE")
