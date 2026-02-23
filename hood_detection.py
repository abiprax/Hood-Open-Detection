import streamlit as st
import cv2
import tempfile
import os
from collections import defaultdict
from ultralytics import YOLO

st.set_page_config(page_title="Hood Open Detection", page_icon="", layout="wide")
st.title(" Hood Open Detection")

# Load model (path hidden from UI)
MODEL_PATH = r"D:\MotoLite Phillipines\New Number plate detection\Hood Open New Model 11l.pt"

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.10, 0.95, 0.30, 0.05)
    target_fps = st.slider("Processing FPS", 1, 15, 5)
    show_labels = st.checkbox("Show Labels", value=True)
    show_conf = st.checkbox("Show Confidence", value=True)

uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_video is not None:
    # Save uploaded video to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())
    input_video_path = tfile.name

    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_interval = max(1, int(fps / target_fps))
    estimated_processed = max(1, total_frames // frame_interval)

    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))

    st_frame = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()

    frame_count = 0
    processed_frames = 0
    class_counts = defaultdict(int)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            results = model.predict(frame, conf=conf_threshold, verbose=False)

            annotated_frame = results[0].plot(labels=show_labels, conf=show_conf)
            out.write(annotated_frame)
            st_frame.image(annotated_frame, channels="BGR", use_container_width=True)

            for box in results[0].boxes:
                cls_name = model.names[int(box.cls[0])]
                class_counts[cls_name] += 1

            processed_frames += 1
            progress_bar.progress(min(processed_frames / estimated_processed, 1.0))
            status_text.text(f"Processing frame {processed_frames}/{estimated_processed}...")

        frame_count += 1

    cap.release()
    out.release()

    status_text.success(" Processing complete!")

    # Summary stats
    st.subheader("Detection Summary")
    cols = st.columns(len(class_counts) if class_counts else 1)
    for i, (cls, count) in enumerate(class_counts.items()):
        cols[i].metric(cls, count)

    # Output video + download
    st.subheader("Annotated Output")
    st.video(output_path)

    with open(output_path, "rb") as f:
        st.download_button(
            label="⬇ Download Annotated Video",
            data=f,
            file_name="hood_detection_output.mp4",
            mime="video/mp4",
        )