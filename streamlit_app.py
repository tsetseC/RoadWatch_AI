"""RoadWatch AI - CES117V Capstone Project.

Streamlit front end for YOLOv8n road-damage detection on uploaded videos.
The original Flask app.py is intentionally unchanged.
"""
import gc
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.pt"
MAX_UPLOAD_MB = 50
cv2.setNumThreads(1)

st.set_page_config(page_title="RoadWatch AI", page_icon="🛣️", layout="wide")
st.markdown("""
<style>
/* Main page and text */
.stApp {background: #0b1424; color: #f1f5f9;}
.stApp h1, .stApp h2, .stApp h3 {color: #ffffff !important;}
/* Metrics: explicitly style both text and inner nested text */
[data-testid="stMetric"] {
    background: #17243b;
    border: 1px solid #30415d;
    padding: 18px;
    border-radius: 12px;
}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
    color: #cbd5e1 !important;
    opacity: 1 !important;
}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
    color: #ffffff !important;
    opacity: 1 !important;
}
[data-testid="stMetricDelta"], [data-testid="stMetricDelta"] * {
    color: #7dd3fc !important;
    opacity: 1 !important;
}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {
    color: #cbd5e1 !important;
}
/* Form controls should be legible on a dark page */
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] * {color: #e2e8f0 !important;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading trained YOLOv8n model...")
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("best.pt is missing from the repository root")
    return YOLO(str(MODEL_PATH))


def resize_frame(frame, max_width=960):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    return cv2.resize(frame, (max_width, max(1, round(h * max_width / w))),
                      interpolation=cv2.INTER_AREA)


def class_name(model, class_id):
    names = model.names
    if isinstance(names, dict):
        return str(names.get(class_id, f"Class {class_id}"))
    return str(names[class_id]) if 0 <= class_id < len(names) else f"Class {class_id}"


def process_video(input_path, output_path, confidence, image_size,
                  frame_interval, max_frames, playback_speed, progress, status):
    """Detect on selected frames; export only those frames at an adjusted FPS.

    Bounding-box totals count appearances in frames, not distinct physical defects.
    """
    model = load_model()
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError("Could not open the video. Try MP4 with H.264 encoding.")

    writer = None
    read_count = 0
    analyzed_count = 0
    counts = Counter()
    records = []
    try:
        source_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not (0 < source_fps <= 120):
            source_fps = 25.0
        source_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        output_fps = max(1.0, source_fps / frame_interval * playback_speed)
        expected = max_frames
        if source_total > 0:
            expected = min(max_frames, (source_total + frame_interval - 1) // frame_interval)

        while analyzed_count < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index = read_count
            read_count += 1
            if frame_index % frame_interval:
                continue

            frame = resize_frame(frame)
            height, width = frame.shape[:2]
            if writer is None:
                codec = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), codec, output_fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError("Unable to initialize the MP4 output writer")

            result = model.predict(source=frame, conf=confidence, imgsz=image_size,
                                   device="cpu", max_det=50, verbose=False)[0]
            for box in result.boxes:
                category = class_name(model, int(box.cls.item()))
                score = float(box.conf.item())
                counts[category] += 1
                records.append({"Source Frame": frame_index, "Damage Type": category,
                                "Confidence": round(score, 3)})

            annotated = result.plot()
            if annotated.shape[:2] != (height, width):
                annotated = cv2.resize(annotated, (width, height))
            writer.write(annotated)
            analyzed_count += 1
            progress.progress(min(1.0, analyzed_count / max(1, expected)))
            status.text(f"Analyzing sampled frame {analyzed_count} of {expected}")

        if not analyzed_count:
            raise ValueError("No readable frames were available for detection")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        gc.collect()

    return {
        "frames_read": read_count,
        "frames_analyzed": analyzed_count,
        "source_fps": source_fps,
        "output_fps": output_fps,
        "output_duration": analyzed_count / output_fps,
        "total_detections": sum(counts.values()),
        "detections": dict(counts),
        "records": records,
    }


def convert_to_h264(source_path, dest_path):
    """Prefer browser-compatible H.264. Retain mp4v if FFmpeg is unavailable."""
    if shutil.which("ffmpeg") is None:
        return False
    try:
        run = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source_path),
             "-c:v", "libx264", "-preset", "ultrafast", "-crf", "25",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dest_path)],
            capture_output=True, text=True, timeout=180,
        )
        return run.returncode == 0 and dest_path.is_file() and dest_path.stat().st_size > 0
    except (OSError, subprocess.TimeoutExpired):
        return False


st.title("🛣️ RoadWatch AI")
st.header("Intelligent Road Damage Detection")
st.write("Upload road inspection footage to detect potholes, longitudinal cracks, "
         "transverse cracks, alligator cracks and other damage using a "
         "custom-trained YOLOv8n model.")
st.divider()

# ============================================================
# IN-APP USER GUIDE: HOW TO TEST ALL FIVE DAMAGE CLASSES
# ============================================================

with st.expander("📘 How to use and test RoadWatch AI", expanded=True):
    st.markdown("""
    **Quick testing procedure**

    1. Choose an **MP4, AVI, MOV or MKV** road-inspection video (under 50 MB).
       Prefer footage where the road surface fills most of the frame.
    2. Select **Confidence = 0.10**, **Resolution = 640**, and
       **Analyze every Nth frame = 1** for an initial diagnostic test.
    3. Start with **20 frames**. If successful, increase to **120** or more
       for a longer portion of the video.
    4. Choose **Playback Speed = 0.10×** for a slow-motion annotated output.
       This slows the video but does not generate new frames.
    5. Click **Analyze Road Video** and review the detected classes,
       confidence table and annotated output.
    6. Download the **MP4** and **CSV** for project evidence.

    **To check all five classes:** test multiple videos, each clearly showing
    a different type of road damage. Use footage from the held-out RDD 2022
    test set where possible, with known ground-truth annotations.
    """)
    st.table(pd.DataFrame([
        {"Class": "Longitudinal Crack", "What to look for": "Crack parallel to traffic direction"},
        {"Class": "Transverse Crack", "What to look for": "Crack across the direction of travel"},
        {"Class": "Alligator Crack", "What to look for": "Interconnected, mesh-like cracking"},
        {"Class": "Other Corruption", "What to look for": "Other labeled pavement damage in the training set"},
        {"Class": "Pothole", "What to look for": "Localized hole or missing road surface"},
    ]))
    st.info(
        "A zero count for a class does not prove the model cannot detect it. "
        "A positive prediction also needs visual verification: inspect the box "
        "against the actual road defect and its labeled ground truth."
    )

st.subheader("Upload Road Inspection Video")
upload = st.file_uploader("Choose an inspection video", type=["mp4", "avi", "mov", "mkv"])

with st.expander("Detection Settings", expanded=True):
    confidence = st.slider("Confidence Threshold", 0.05, 0.90, 0.10, 0.05)
    image_size = st.select_slider("Inference Resolution", options=[256, 320, 416, 512, 640], value=640)
    frame_interval = st.selectbox("Analyze every Nth frame", options=[1, 2, 5, 10], index=0)
    max_frames = st.number_input("Maximum frames to analyze", min_value=1,
                                 max_value=1000, value=20, step=10)
    playback_speed = st.selectbox("Annotated Video Playback Speed",
                                  options=[0.10, 0.25, 0.5, 0.75, 1.0], index=0,
                                  format_func=lambda speed: f"{speed}× Speed")
    st.caption("0.10× plays the annotated results very slowly. "
               "Longer source coverage requires more analyzed frames; "
               "slower playback alone does not reveal additional footage.")

if upload is not None:
    signature = (upload.name, upload.size, upload.getvalue()[:128])
    if st.session_state.get("source_signature") != signature:
        st.session_state["source_signature"] = signature
        st.session_state.pop("roadwatch_result", None)

    st.success("Video uploaded successfully.")
    st.subheader("Original Road Inspection Video")
    st.video(upload)
    st.caption(f"File size: {upload.size / (1024 * 1024):.2f} MB")

    if st.button("Analyze Road Video", type="primary"):
        if upload.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"Video exceeds the {MAX_UPLOAD_MB} MB limit")
        else:
            st.session_state.pop("roadwatch_result", None)
            progress = st.progress(0)
            status = st.empty()
            try:
                with st.spinner("Running YOLOv8n detection..."):
                    with tempfile.TemporaryDirectory() as folder:
                        work = Path(folder)
                        suffix = Path(upload.name).suffix.lower()
                        input_path = work / f"input{suffix}"
                        output_path = work / "annotated_raw.mp4"
                        converted_path = work / "annotated_h264.mp4"
                        input_path.write_bytes(upload.getvalue())

                        stats = process_video(
                            input_path, output_path, confidence, image_size,
                            frame_interval, int(max_frames), playback_speed,
                            progress, status
                        )
                        if not output_path.is_file() or not output_path.stat().st_size:
                            raise RuntimeError("Annotated MP4 was not generated")
                        status.text("Preparing downloadable video...")
                        browser_ready = convert_to_h264(output_path, converted_path)
                        selected = converted_path if browser_ready else output_path
                        video_bytes = selected.read_bytes()

                    st.session_state["roadwatch_result"] = {
                        "source": signature,
                        "stats": stats,
                        "video": video_bytes,
                        "browser_ready": browser_ready,
                    }
                    status.text("Detection completed")
                    progress.progress(1.0)
                    st.success("Road damage analysis completed successfully.")
            except Exception as exc:
                st.error("Video processing failed. See error details below.")
                st.exception(exc)

saved = st.session_state.get("roadwatch_result")
if saved is not None and upload is not None and saved["source"] == signature:
    stats = saved["stats"]
    st.divider()
    st.header("Detection Results")
    a, b, c = st.columns(3)
    a.metric("Source Frames Read", stats["frames_read"])
    b.metric("Frames Analyzed", stats["frames_analyzed"])
    c.metric("Total Detections", stats["total_detections"])

    st.divider()
    st.subheader("Video Processing Summary")
    x, y = st.columns(2)
    x.metric("Output Video Duration", f"{stats['output_duration']:.2f} sec")
    y.metric("Output Frame Rate", f"{stats['output_fps']:.2f} FPS")
    st.caption("Duration depends on number of analyzed frames, sampling interval and playback speed.")

    st.subheader("Detected Damage Categories")
    st.caption("Counts are predictions across frames, not unique physical defects.")
    # Always display all trained classes, including zero-count classes.
    trained_names = [class_name(load_model(), i) for i in range(len(load_model().names))]
    summary = pd.DataFrame([
        {"Class": name, "Detections": stats["detections"].get(name, 0)}
        for name in trained_names
    ])
    st.dataframe(summary, use_container_width=True, hide_index=True)
    if stats["detections"]:
        categories = pd.DataFrame(
            [{"Damage Type": k, "Detections": v} for k, v in stats["detections"].items()]
        ).set_index("Damage Type")
        st.bar_chart(categories)
    else:
        st.info("No damage exceeded the selected confidence threshold in analyzed frames.")

    if stats["records"]:
        st.subheader("Detection Details")
        report = pd.DataFrame(stats["records"])
        st.dataframe(report, use_container_width=True, hide_index=True)
        st.download_button("Download Detection Report (CSV)", report.to_csv(index=False).encode(),
                           file_name="RoadWatch_AI_Detections.csv", mime="text/csv")

    st.divider()
    st.subheader("Annotated Road Inspection Video")
    if saved["browser_ready"]:
        st.video(saved["video"], format="video/mp4")
    else:
        st.warning("Browser playback may be unavailable without FFmpeg/H.264. Download the MP4 to view locally.")
    st.download_button("Download Annotated Video", saved["video"],
                       file_name="RoadWatch_AI_Result.mp4", mime="video/mp4", type="primary")
    st.caption("Counts represent predictions across analyzed frames, not unique road defects. "
               "The export contains only sampled frames; slowing playback does not restore "
               "unprocessed frames. Very slow playback can look choppy.")

st.divider()
st.caption("RoadWatch AI | CES117V Capstone Project | YOLOv8n | RDD 2022")
