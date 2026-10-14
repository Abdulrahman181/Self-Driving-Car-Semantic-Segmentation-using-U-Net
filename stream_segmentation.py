"""Streamlit inference UI for a trained eight-class U-Net model."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

NUM_CLASSES = 8
CLASS_NAMES = [
    "Void/Background",
    "Road",
    "Construction",
    "Object",
    "Nature",
    "Sky",
    "Human",
    "Vehicle",
]
COLORS = np.array(
    [
        [0, 0, 0],
        [128, 64, 128],
        [70, 70, 70],
        [250, 170, 30],
        [107, 142, 35],
        [70, 130, 180],
        [220, 20, 60],
        [0, 0, 142],
    ],
    dtype=np.uint8,
)

st.set_page_config(page_title="Cityscapes Analytics", layout="wide")
st.title("Autonomous Driving Scene Analytics System")
st.write("Semantic segmentation with confidence and class-coverage summaries.")


@st.cache_resource
def load_model(model_path: str):
    path = Path(model_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found: {path}. Set CITYSCAPES_MODEL to an existing "
            "trained .keras file; no model is bundled with this repository."
        )
    model = tf.keras.models.load_model(path)
    output_shape = model.output_shape
    if len(output_shape) != 4 or output_shape[-1] != NUM_CLASSES:
        raise ValueError(
            f"Expected model output shape [batch, height, width, {NUM_CLASSES}], "
            f"received {output_shape}."
        )
    return model


MODEL_PATH = os.environ.get("CITYSCAPES_MODEL", "cityscapes_unet_model.keras")
try:
    model = load_model(MODEL_PATH)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()
except (OSError, TypeError, ValueError, RuntimeError) as exc:
    st.error(f"Unable to load or validate the model artifact: {exc}")
    st.stop()
else:
    st.sidebar.success("System Ready")


def _target_size() -> tuple[int, int]:
    shape = model.input_shape
    if len(shape) != 4 or shape[1] is None or shape[2] is None:
        return (256, 256)
    return (int(shape[2]), int(shape[1]))


def process_image(image_pil: Image.Image):
    """Return colored mask, overlay, confidence percentage, and class mask."""
    target_size = _target_size()
    image_rgb = image_pil.convert("RGB").resize(target_size, Image.Resampling.BILINEAR)
    image_array = np.asarray(image_rgb, dtype=np.float32) / 255.0
    prediction = np.asarray(model.predict(image_array[None, ...], verbose=0))
    if prediction.ndim != 4 or prediction.shape[-1] != NUM_CLASSES:
        raise ValueError(f"Unexpected prediction shape: {prediction.shape}")

    confidence_map = np.max(prediction[0], axis=-1)
    mean_confidence = float(np.mean(confidence_map) * 100.0)
    pred_mask = np.argmax(prediction[0], axis=-1).astype(np.int32)
    colored_mask = COLORS[pred_mask]
    original_array = np.asarray(image_rgb, dtype=np.uint8)
    overlay = cv2.addWeighted(original_array, 0.6, colored_mask, 0.4, 0)
    return colored_mask, overlay, mean_confidence, pred_mask


def get_class_statistics(pred_mask: np.ndarray) -> dict[str, float]:
    total_pixels = int(pred_mask.size)
    if total_pixels == 0:
        return {}
    unique, counts = np.unique(pred_mask, return_counts=True)
    return {
        CLASS_NAMES[int(class_id)]: round(float(count / total_pixels * 100.0), 2)
        for class_id, count in zip(unique, counts)
        if 0 <= int(class_id) < NUM_CLASSES
    }


def analyze_video(uploaded_video) -> None:
    """Process a bounded video upload; Streamlit buttons cannot interrupt a tight loop."""
    suffix = Path(uploaded_video.name).suffix or ".mp4"
    temp_path: Path | None = None
    cap = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_video.getvalue())
            temp_path = Path(temp_file.name)
        cap = cv2.VideoCapture(str(temp_path))
        if not cap.isOpened():
            st.error("The uploaded video could not be opened by OpenCV.")
            return
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frames = min(frame_count if frame_count > 0 else 300, 300)
        frame_slot, metric_slot = st.empty(), st.empty()
        for _ in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, overlay, confidence, _ = process_image(Image.fromarray(frame_rgb))
            frame_slot.image(overlay, channels="RGB", use_container_width=True)
            metric_slot.metric("Mean pixel confidence", f"{confidence:.2f}%")
    finally:
        if cap is not None:
            cap.release()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


st.sidebar.title("Control Panel")
mode = st.sidebar.selectbox("Select Input Mode", ["Image Analysis", "Video Analysis", "Live Camera"])

if mode == "Image Analysis":
    st.subheader("Single Image Analytics")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Input", use_container_width=True)
        if st.button("Analyze Scene"):
            with st.spinner("Analyzing..."):
                _, overlay, confidence, pred_mask = process_image(image)
            with col2:
                st.image(overlay, caption=f"Output (Confidence: {confidence:.2f}%)", use_container_width=True)
            stats = get_class_statistics(pred_mask)
            st.markdown("### Scene Statistics")
            m1, m2, m3 = st.columns(3)
            m1.metric("Mean pixel confidence", f"{confidence:.2f}%")
            m2.metric("Road coverage", f"{stats.get('Road', 0.0):.2f}%")
            m3.metric("Vehicle coverage", f"{stats.get('Vehicle', 0.0):.2f}%")
            st.table(pd.DataFrame(list(stats.items()), columns=["Object Class", "Coverage (%)"]))

elif mode == "Video Analysis":
    st.subheader("Video File Analytics (first 300 frames)")
    uploaded_video = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov"])
    if uploaded_video and st.button("Analyze Video"):
        with st.spinner("Analyzing video..."):
            analyze_video(uploaded_video)

else:
    st.subheader("Camera Snapshot Analytics")
    st.caption("The app processes one snapshot per run to avoid an uninterruptible server-side camera loop.")
    camera_file = st.camera_input("Capture a frame")
    if camera_file:
        camera_image = Image.open(camera_file).convert("RGB")
        _, overlay, confidence, pred_mask = process_image(camera_image)
        st.image(overlay, caption=f"Output (Confidence: {confidence:.2f}%)", use_container_width=True)
        st.table(pd.DataFrame(list(get_class_statistics(pred_mask).items()), columns=["Object Class", "Coverage (%)"]))
