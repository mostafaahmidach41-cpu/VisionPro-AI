import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="VisionPro AI | Neural Engine",
    page_icon="👁️",
    layout="wide"
)

# --- 2. Custom CSS Styling ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. Optimized Model Loading ---
@st.cache_resource
def load_yolo_model():
    # This downloads the lightweight YOLOv8n model automatically
    return YOLO("yolov8n.pt")

try:
    model = load_yolo_model()
except Exception as e:
    st.error(f"Error loading neural engine: {e}")

# --- 4. Header Section ---
st.title("👁️ VisionPro AI - Neural Terminal")
st.markdown("Independent Real-time Object Detection v1.0")
st.divider()

# --- 5. Main Interface Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📷 System Camera")
    cam_toggle = st.toggle("Power On Neural Scan")
    input_img = st.camera_input("Scanner", disabled=not cam_toggle)

with col2:
    st.subheader("🧠 Neural Output")
    if input_img is not None:
        # Process the captured image
        bytes_data = input_img.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        # Run AI Inference
        results = model(cv2_img, conf=0.5) 
        
        for r in results:
            # Draw boxes and labels
            annotated_frame = r.plot()
            st.image(annotated_frame, channels="BGR", use_container_width=True)
            
            # Analytics Logic
            names = model.names
            detected_classes = r.boxes.cls.numpy()
            person_count = len([c for c in detected_classes if names[int(c)] == 'person'])
            
            # Display Real-time Metrics
            m1, m2 = st.columns(2)
            m1.metric("Persons Detected", person_count)
            m2.metric("Scan Accuracy", f"{float(r.boxes.conf.mean()*100) if len(r.boxes) > 0 else 0:.1f}%")
    else:
        st.info("System Ready. Please activate camera to begin scanning.")

# --- 6. Footer Information ---
st.divider()
st.caption("VisionPro AI Enterprise | Architecture: YOLOv8 | Status: Operational")
