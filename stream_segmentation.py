import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import tempfile

st.set_page_config(
    page_title="Cityscapes Analytics",
    layout="wide"
)

st.title("Autonomous Driving Scene Analytics System")
st.write("Real-time Semantic Segmentation with Confidence Metrics.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('cityscapes_unet_model.keras')

try:
    model = load_model()
    st.sidebar.success("System Ready")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

CLASS_NAMES = [
    "Void/Background", "Road", "Construction", "Object", 
    "Nature", "Sky", "Human", "Vehicle"
]

COLORS = np.array([
    [0, 0, 0], [128, 64, 128], [70, 70, 70], [250, 170, 30],
    [107, 142, 35], [70, 130, 180], [220, 20, 60], [0, 0, 142]
])

def process_image(image_pil):
    target_size = (256, 256)
    img_resized = image_pil.resize(target_size)
    
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_input, verbose=0)
    

    confidence_map = np.max(pred[0], axis=-1)
    mean_confidence = np.mean(confidence_map) * 100
    
    pred_mask = np.argmax(pred, axis=-1)[0]
    colored_mask = COLORS[pred_mask].astype(np.uint8)

    original_arr = np.array(img_resized)
    overlay = cv2.addWeighted(original_arr, 0.6, colored_mask, 0.4, 0)
    
    return colored_mask, overlay, mean_confidence, pred_mask

def get_class_statistics(pred_mask):

    total_pixels = pred_mask.size
    unique, counts = np.unique(pred_mask, return_counts=True)
    stats = {}
    for class_id, count in zip(unique, counts):
        percentage = (count / total_pixels) * 100
        stats[CLASS_NAMES[class_id]] = round(percentage, 2)
    return stats

st.sidebar.title("Control Panel")
mode = st.sidebar.selectbox("Select Input Mode", ["Image Analysis", "Video Analysis", "Live Camera"])

if mode == "Image Analysis":
    st.subheader("Single Image Analytics")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Input", use_container_width=True)
            
        if st.button("Analyze Scene"):
            with st.spinner("Analyzing..."):
                _, overlay, conf, pred_mask = process_image(image)
                stats = get_class_statistics(pred_mask)
            
            with col2:
                st.image(overlay, caption=f"Output (Confidence: {conf:.2f}%)", use_container_width=True)
            
            st.markdown("### Scene Statistics")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Model Confidence", f"{conf:.2f}%")
            if "Road" in stats: m2.metric("Road Coverage", f"{stats['Road']}%")
            if "Vehicle" in stats: m3.metric("Vehicle Coverage", f"{stats['Vehicle']}%")
            
            st.table(pd.DataFrame(list(stats.items()), columns=["Object Class", "Coverage (%)"]))

elif mode == "Video Analysis":
    st.subheader("Video File Analytics")
    uploaded_video = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov"])
    
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        
        st_frame = st.empty()
        st_metric = st.empty()
        
        stop_btn = st.button("Stop Processing")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop_btn:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            _, overlay, conf, _ = process_image(pil_img)
            
            st_frame.image(overlay, use_container_width=True)
            st_metric.metric("Real-time Confidence", f"{conf:.2f}%")
            
        cap.release()

elif mode == "Live Camera":
    st.subheader("Real-time Camera Analytics")
    run_camera = st.checkbox("Start Camera Feed")
    
    st_frame = st.empty()
    st_metric = st.empty()
    
    if run_camera:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Camera not detected.")
        
        while run_camera:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            _, overlay, conf, _ = process_image(pil_img)
            
            st_frame.image(overlay, use_container_width=True)
            st_metric.metric("Real-time Confidence", f"{conf:.2f}%")
        
        cap.release()