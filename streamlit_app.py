
# ============================================================
# ROADWATCH AI - CES117V CAPSTONE PROJECT
# STREAMLIT APPLICATION WITH YOLOv8n VIDEO DETECTION
# ============================================================

# Import the required Python libraries.
import streamlit as st
import cv2
import tempfile
import subprocess
import shutil
import os
import gc

from pathlib import Path
from collections import Counter
from ultralytics import YOLO

# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RoadWatch AI",
    page_icon="🛣️",
    layout="wide"
)

# Limit OpenCV CPU threading on shared hosting.
cv2.setNumThreads(1)

# Locate the trained YOLOv8n model.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.pt"

# Define the five supported damage categories.
CLASS_NAMES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Other Corruption",
    "Pothole"
]

# ============================================================
# 2. LOAD THE TRAINED MODEL
# ============================================================

# Cache the model to avoid loading it repeatedly
# during Streamlit interface updates.
@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Trained model best.pt was not found."
        )

    return YOLO(str(MODEL_PATH))


# ============================================================
# 3. VIDEO DETECTION FUNCTION
# ============================================================

def process_video(
    input_path,
    output_path,
    confidence,
    image_size,
    frame_interval,
    max_frames,
    progress_bar,
    status_text
):
    """
    Analyze sampled frames using YOLOv8n.

    The output contains annotated sampled frames.
    Sampling reduces computation on shared CPU hosting.
    """

    # Load the model using Streamlit's cache.
    model = load_model()

    # Open the uploaded video.
    capture = cv2.VideoCapture(str(input_path))

    if not capture.isOpened():
        raise ValueError("Unable to open the video file.")

    writer = None
    frames_read = 0
    frames_analyzed = 0

    # Store the number of detected objects per class.
    detection_counts = Counter()

    try:
        # Read the video properties.
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if source_fps <= 0 or source_fps > 120:
            source_fps = 25.0

        # Adjust output FPS to match frame sampling.
        output_fps = max(
            1.0,
            source_fps / frame_interval
        )

        # Estimate the number of frames to analyze.
        expected_frames = max_frames

        if total_frames > 0:
            expected_frames = min(
                max_frames,
                (total_frames + frame_interval - 1)
                // frame_interval
            )

        # Read frames until the limit is reached.
        while frames_analyzed < max_frames:

            success, frame = capture.read()

            if not success:
                break

            current_frame = frames_read
            frames_read += 1

            # Only analyze selected video frames.
            if current_frame % frame_interval != 0:
                continue

            # Resize the frame to reduce memory usage.
            height, width = frame.shape[:2]

            if width > 640:
                scale = 640 / width

                frame = cv2.resize(
                    frame,
                    (
                        640,
                        max(1, int(height * scale))
                    )
                )

            height, width = frame.shape[:2]

            # Create the video writer once.
            if writer is None:

                fourcc = cv2.VideoWriter_fourcc(
                    *"mp4v"
                )

                writer = cv2.VideoWriter(
                    str(output_path),
                    fourcc,
                    output_fps,
                    (width, height)
                )

                if not writer.isOpened():
                    raise RuntimeError(
                        "Could not create output video."
                    )

            # ============================================
            # PERFORM ROAD-DAMAGE OBJECT DETECTION
            # ============================================

            results = model.predict(
                source=frame,
                conf=confidence,
                imgsz=image_size,
                device="cpu",
                max_det=50,
                verbose=False
            )

            result = results[0]

            # Count detections by their class ID.
            for box in result.boxes:

                class_id = int(box.cls.item())

                class_name = (
                    CLASS_NAMES[class_id]
                    if 0 <= class_id < len(CLASS_NAMES)
                    else f"Class {class_id}"
                )

                detection_counts[class_name] += 1

            # Draw bounding boxes on the frame.
            annotated_frame = result.plot()

            # Write the result to the video.
            writer.write(annotated_frame)

            frames_analyzed += 1

            # Update the Streamlit progress indicator.
            progress_bar.progress(
                min(
                    frames_analyzed / max(
                        expected_frames, 1
                    ),
                    1.0
                )
            )

            status_text.text(
                f"Analyzing frame {frames_analyzed} "
                f"of {expected_frames}"
            )

        if frames_analyzed == 0:
            raise ValueError(
                "No readable video frames found."
            )

    finally:
        # Always close video-processing resources.
        capture.release()

        if writer is not None:
            writer.release()

        gc.collect()

    return {
        "frames_read": frames_read,
        "frames_analyzed": frames_analyzed,
        "detections": dict(detection_counts),
        "total_detections": sum(
            detection_counts.values()
        )
    }


# ============================================================
# 4. USER INTERFACE
# ============================================================

st.title("🛣️ RoadWatch AI")

st.header("Intelligent Road Damage Detection")

st.write(
    "Upload road inspection footage to detect "
    "longitudinal cracks, transverse cracks, "
    "alligator cracks, other corruption and potholes "
    "using a custom-trained YOLOv8n model."
)

st.divider()

# ============================================================
# 5. VIDEO UPLOAD
# ============================================================

st.subheader("Upload Road Inspection Video")

uploaded_video = st.file_uploader(
    "Choose an inspection video",
    type=["mp4", "avi", "mov", "mkv"]
)

# Allow users to adjust model parameters.
with st.expander("Detection Settings"):

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05
    )

    image_size = st.select_slider(
        "Inference Resolution",
        options=[256, 320, 416, 512, 640],
        value=320
    )

    frame_interval = st.selectbox(
        "Analyze every Nth frame",
        options=[1, 2, 5, 10],
        index=2
    )

    max_frames = st.number_input(
        "Maximum frames to analyze",
        min_value=1,
        max_value=300,
        value=10,
        step=1
    )

# ============================================================
# 6. RUN VIDEO ANALYSIS
# ============================================================

if uploaded_video is not None:

    # Preview the uploaded video.
    st.success("Video uploaded successfully.")

    st.video(uploaded_video)

    # Begin detection when the button is clicked.
    if st.button(
        "Analyze Road Video",
        type="primary"
    ):

        # Reject unusually large files on free hosting.
        if uploaded_video.size > 50 * 1024 * 1024:
            st.error(
                "Please upload a video smaller than 50 MB."
            )

        else:
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                with st.spinner(
                    "Running YOLOv8n road-damage detection..."
                ):

                    # Use a temporary directory for uploaded
                    # and generated video files.
                    with tempfile.TemporaryDirectory() as tmp:

                        input_path = Path(tmp) / (
                            "input" +
                            Path(uploaded_video.name).suffix.lower()
                        )

                        output_path = Path(tmp) / "result.mp4"

                        # Save the uploaded video temporarily.
                        input_path.write_bytes(
                            uploaded_video.getvalue()
                        )

                        # Run YOLOv8n detection.
                        stats = process_video(
                            input_path,
                            output_path,
                            confidence,
                            image_size,
                            frame_interval,
                            max_frames,
                            progress_bar,
                            status_text
                        )

                        # Read the generated MP4.
                        # Optionally transcode it to H.264
                        # for improved browser compatibility.
                        playable_path = output_path
                        h264_path = Path(tmp) / "result_h264.mp4"

                        if shutil.which("ffmpeg"):
                            conversion = subprocess.run(
                                [
                                    "ffmpeg", "-y",
                                    "-loglevel", "error",
                                    "-i", str(output_path),
                                    "-c:v", "libx264",
                                    "-preset", "ultrafast",
                                    "-pix_fmt", "yuv420p",
                                    "-movflags", "+faststart",
                                    str(h264_path)
                                ],
                                capture_output=True,
                                text=True
                            )

                            if (
                                conversion.returncode == 0
                                and h264_path.is_file()
                                and h264_path.stat().st_size > 0
                            ):
                                playable_path = h264_path

                        output_bytes = playable_path.read_bytes()

                    # Save results in the Streamlit session.
                    st.session_state["roadwatch_result"] = {
                        "stats": stats,
                        "video": output_bytes,
                        "browser_compatible": (
                            playable_path == h264_path
                        )
                    }

            except Exception as error:
                st.error(
                    "Unable to complete video processing."
                )
                st.exception(error)

# ============================================================
# 7. DISPLAY DETECTION RESULTS
# ============================================================

if "roadwatch_result" in st.session_state:

    saved_result = st.session_state["roadwatch_result"]

    stats = saved_result["stats"]

    st.divider()

    st.subheader("Detection Results")

    # Display summary metrics.
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Source Frames Read",
        stats["frames_read"]
    )

    col2.metric(
        "Frames Analyzed",
        stats["frames_analyzed"]
    )

    col3.metric(
        "Total Detections",
        stats["total_detections"]
    )

    # Display detected road-damage categories.
    st.subheader("Detected Damage Categories")

    if stats["detections"]:

        st.bar_chart(stats["detections"])

    else:
        st.info(
            "No road damage exceeded the selected "
            "confidence threshold in sampled frames."
        )

    # Display the annotated output.
    st.subheader("Annotated Video")

    if saved_result["browser_compatible"]:
        st.video(saved_result["video"])
    else:
        st.warning(
            "Browser playback may be unavailable. "
            "Download the MP4 to view it locally."
        )

    # Allow the user to download the output video.
    st.download_button(
        label="Download Annotated Video",
        data=saved_result["video"],
        file_name="RoadWatch_AI_Result.mp4",
        mime="video/mp4"
    )

    st.caption(
        "Detection counts represent predictions "
        "across analyzed frames, not unique road defects."
    )

# ============================================================
# 8. FOOTER
# ============================================================

st.divider()

st.caption(
    "RoadWatch AI | CES117V Capstone Project | "
    "Powered by YOLOv8n | RDD 2022"
)
