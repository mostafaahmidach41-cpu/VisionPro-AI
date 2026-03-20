import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
from ultralytics import YOLO
import numpy as np

# Load Model
model = YOLO("yolov8n.pt")

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # AI Inference on each frame
        results = model(img, conf=0.5)
        
        # Draw results on the frame
        annotated_frame = results[0].plot()

        return annotated_frame

st.title("👁️ VisionPro AI - Live Video Scan")

# Start Video Stream
webrtc_streamer(
    key="vision-stream", 
    video_processor_factory=VideoProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)
