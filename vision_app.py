import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="VisionPro AI | Neural Terminal",
    page_icon="👁️",
    layout="wide"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

# --- Load Neural Engine ---
@st.cache_resource
def load_yolo_model():
    # This will automatically download yolov8n.pt on first run
    return YOLO("yolov8n.pt")

model = load_yolo_model()

# --- Header Section ---
st.title("👁️ VisionPro AI - Neural Vision Terminal")
st.markdown("Independent Real-time Object Detection Engine v1.0")
st.divider()

# --- Interface Layout ---
col_input, col_output = st.columns([1, 1])

with col_input:
    st.subheader("📷 Visual Input")
    enable_cam = st.toggle("Activate System Camera")
    img_file_buffer = st.camera_input("Neural Scan", disabled=not enable_cam)

with col_output:
    st.subheader("🧠 Neural Analysis")
    if img_file_buffer is not None:
        # Convert buffer to OpenCV format
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        # Run YOLO Inference
        results = model(cv2_img, conf=0.5) # 50% confidence threshold
        
        # Process Results
        for r in results:
            annotated_frame = r.plot()
            st.image(annotated_frame, channels="BGR", use_container_width=True)
            
            # Extract detection stats
            names = model.names
            classes = r.boxes.cls.numpy()
            person_count = len([c for c in classes if names[int(c)] == 'person'])
            
            # Display Metrics
            m1, m2 = st.columns(2)
            m1.metric("Persons Detected", person_count)
            m2.metric("Neural Confidence", f"{float(r.boxes.conf.mean()*100) if len(r.boxes) > 0 else 0:.1f}%")
    else:
        st.info("Awaiting visual input for analysis...")

# --- Footer ---
st.divider()
st.caption("VisionPro AI Enterprise | Powered by Ultralytics YOLOv8 | 2026")
