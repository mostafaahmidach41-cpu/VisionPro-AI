import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="VisionPro AI | Performance Terminal",
    page_icon="👁️",
    layout="wide"
)

# --- 2. Custom Styling ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. Optimized Model Loading ---
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")

try:
    model = load_yolo_model()
except Exception as e:
    st.error(f"Neural Engine Error: {e}")

# --- 4. Header ---
st.title("👁️ VisionPro AI - Neural Terminal")
st.markdown("Independent Real-time Object Detection v1.0")
st.divider()

# --- 5. Logic & Interface ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📷 Visual Input")
    cam_active = st.toggle("Activate Neural Scan")
    # Using a unique key to prevent JavaScript errors
    input_img = st.camera_input("Neural Scanner", disabled=not cam_active, key="vision_scanner")

with col2:
    st.subheader("🧠 Neural Analysis")
    if input_img is not None:
        bytes_data = input_img.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        # AI Inference
        results = model(cv2_img, conf=0.5) 
        
        for r in results:
            annotated_frame = r.plot()
            st.image(annotated_frame, channels="BGR", use_container_width=True)
            
            # Analytics
            names = model.names
            classes = r.boxes.cls.numpy()
            person_count = len([c for c in classes if names[int(c)] == 'person'])
            
            # Live Dashboard Metrics
            m1, m2 = st.columns(2)
            m1.metric("Persons Detected", person_count)
            m2.metric("Scan Accuracy", f"{float(r.boxes.conf.mean()*100) if len(r.boxes) > 0 else 0:.1f}%")
    else:
        st.info("System Standby. Activate scanner to proceed.")

st.divider()
st.caption("VisionPro AI Enterprise | Architecture: YOLOv8 | Global Operational Status: OK")
