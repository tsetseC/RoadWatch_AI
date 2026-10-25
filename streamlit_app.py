
# ============================================================
# ROADWATCH AI - CES117V CAPSTONE PROJECT
# STREAMLIT APPLICATION
# YOLOv8n INTELLIGENT ROAD DAMAGE DETECTION
# ============================================================

import gc
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RoadWatch AI",
    page_icon="🛣️",
    layout="wide"
)

# Reduce CPU contention on shared hosting.
cv2.setNumThreads(1)

# Project directory and trained model.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.pt"

# Maximum permitted upload size.
MAX_UPLOAD_MB = 50

# The five classes from the training configuration.
# Actual prediction labels are read from best.pt.
EXPECTED_CLASSES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Other Corruption",
    "Pothole"
]


# ============================================================
# 2. PAGE STYLING
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1424;
        color: #f1f5f9;
    }

    h1, h2, h3 {
        color: #f1f5f9;
    }

    div[data-testid="stMetric"] {
        background-color: #17243b;
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #30415d;
    }

    div.stButton > button[kind="primary"] {
        background-color: #0284c7;
        border-radius: 9px;
        color: white;
        font-weight: 600;
    }

    div.stDownloadButton > button {
        border-radius: 9px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. LOAD THE TRAINED YOLO MODEL
# ============================================================

@st.cache_resource
def load_model():
    """
    Load the YOLOv8n model only once per app process.

    The weights contain the learned road-damage features
    and the original training class names.
    """

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Cannot find trained model: {MODEL_PATH}"
        )

    return YOLO(str(MODEL_PATH))


def get_class_name(model, class_id):
    """
    Obtain a class name from the trained model metadata.

    Avoid assuming that a manually written class list
    always matches the model's internal class IDs.
    """

    names = model.names

    if isinstance(names, dict):
        return str(
            names.get(class_id, f"Class {class_id}")
        )

    if 0 <= class_id < len(names):
        return str(names[class_id])

    return f"Class {class_id}"


# ============================================================
# 4. RESIZE VIDEO FRAMES
# ============================================================

def resize_frame(frame, max_width=960):
    """
    Resize large frames while preserving aspect ratio.

    Smaller frames remain at their original resolution.
    """

    height, width = frame.shape[:2]

    if width <= max_width:
        return frame

    scale = max_width / width

    new_width = int(width * scale)
    new_height = max(
        1,
        int(round(height * scale))
    )

    return cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )


# ============================================================
# 5. PROCESS THE VIDEO
# ============================================================

def process_video(
    input_path,
    output_path,
    confidence,
    image_size,
    frame_interval,
    max_frames,
    playback_speed,
    progress_bar,
    status_text
):
    """
    Perform YOLO road-damage detection on selected frames.

    FRAME INTERVAL:
        1 = Analyze every frame.
        5 = Analyze every fifth frame.

    MAX FRAMES:
        Maximum number of YOLO inference operations.

    PLAYBACK SPEED:
        1.0 = Normal speed.
        0.5 = Half speed.
        0.25 = Quarter speed.

    The exported video contains only analyzed frames.

    The output frame rate is adjusted to approximate
    the source-video timing after frame sampling.
    """

    model = load_model()

    capture = cv2.VideoCapture(str(input_path))

    if not capture.isOpened():
        raise ValueError(
            "Unable to open the uploaded video."
        )

    writer = None

    # Track original frames read and analyzed.
    frames_read = 0
    frames_analyzed = 0

    # Track prediction counts for each category.
    detection_counts = Counter()

    # Collect confidence information for reporting.
    detection_records = []

    try:

        # ----------------------------------------------------
        # READ VIDEO PROPERTIES
        # ----------------------------------------------------

        source_fps = float(
            capture.get(cv2.CAP_PROP_FPS)
        )

        total_frames = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if (
            not np.isfinite(source_fps)
            or source_fps <= 0
            or source_fps > 120
        ):
            source_fps = 25.0

        # ----------------------------------------------------
        # CALCULATE OUTPUT FRAME RATE
        # ----------------------------------------------------

        # Example:
        # Original FPS = 30
        # Frame interval = 1
        # Playback speed = 0.5
        #
        # Output FPS = 15
        #
        # 120 output frames / 15 FPS = 8 seconds.

        output_fps = max(
            1.0,
            (
                source_fps / frame_interval
            ) * playback_speed
        )

        # Calculate the number of frames expected.
        expected_frames = max_frames

        if total_frames > 0:
            available_samples = (
                total_frames + frame_interval - 1
            ) // frame_interval

            expected_frames = min(
                max_frames,
                available_samples
            )

        # ----------------------------------------------------
        # VIDEO PROCESSING LOOP
        # ----------------------------------------------------

        while frames_analyzed < max_frames:

            success, frame = capture.read()

            if not success:
                break

            original_frame_index = frames_read
            frames_read += 1

            # Skip frames not selected for inference.
            if (
                original_frame_index % frame_interval
                != 0
            ):
                continue

            # Reduce large frame dimensions.
            frame = resize_frame(frame)

            height, width = frame.shape[:2]

            # ------------------------------------------------
            # INITIALIZE VIDEO WRITER
            # ------------------------------------------------

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
                        "Unable to create output MP4."
                    )

            # ------------------------------------------------
            # PERFORM YOLOv8n INFERENCE
            # ------------------------------------------------

            results = model.predict(
                source=frame,
                imgsz=image_size,
                conf=confidence,
                device="cpu",
                max_det=50,
                verbose=False
            )

            result = results[0]

            # ------------------------------------------------
            # RECORD DETECTED DAMAGE
            # ------------------------------------------------

            for box in result.boxes:

                class_id = int(box.cls.item())
                score = float(box.conf.item())

                class_name = get_class_name(
                    model,
                    class_id
                )

                detection_counts[class_name] += 1

                detection_records.append({
                    "Frame": original_frame_index,
                    "Damage Type": class_name,
                    "Confidence": round(score, 3)
                })

            # ------------------------------------------------
            # DRAW BOUNDING BOXES
            # ------------------------------------------------

            annotated_frame = result.plot()

            # Ensure that output dimensions remain fixed.
            if (
                annotated_frame.shape[1] != width
                or annotated_frame.shape[0] != height
            ):

                annotated_frame = cv2.resize(
                    annotated_frame,
                    (width, height)
                )

            # Write annotated frame into output video.
            writer.write(annotated_frame)

            frames_analyzed += 1

            # ------------------------------------------------
            # UPDATE THE STREAMLIT PROGRESS BAR
            # ------------------------------------------------

            progress_bar.progress(
                min(
                    1.0,
                    frames_analyzed /
                    max(expected_frames, 1)
                )
            )

            status_text.text(
                f"Analyzing frame "
                f"{frames_analyzed} "
                f"of {expected_frames}"
            )

            # Release intermediate results.
            del results
            del result
            del annotated_frame

        if frames_analyzed == 0:
            raise ValueError(
                "No readable frames were found."
            )

    finally:

        # Always close video resources.
        capture.release()

        if writer is not None:
            writer.release()

        gc.collect()

    # --------------------------------------------------------
    # CALCULATE OUTPUT VIDEO DURATION
    # --------------------------------------------------------

    output_duration = (
        frames_analyzed / output_fps
    )

    return {
        "source_fps": source_fps,
        "source_total_frames": total_frames,
        "frames_read": frames_read,
        "frames_analyzed": frames_analyzed,
        "output_fps": output_fps,
        "output_duration": output_duration,
        "total_detections": sum(
            detection_counts.values()
        ),
        "detections": dict(detection_counts),
        "records": detection_records
    }


# ============================================================
# 6. CONVERT OUTPUT TO BROWSER-COMPATIBLE MP4
# ============================================================

def convert_to_h264(source_path, target_path):
    """
    Convert the generated video to H.264.

    This improves playback in browsers.

    Requires ffmpeg to be installed on the server.
    """

    if shutil.which("ffmpeg") is None:
        return False

    command = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-i", str(source_path),
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "25",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        str(target_path)
    ]

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=180
        )

        return (
            completed.returncode == 0
            and target_path.is_file()
            and target_path.stat().st_size > 0
        )

    except (
        subprocess.TimeoutExpired,
        OSError
    ):
        return False


# ============================================================
# 7. APPLICATION HEADER
# ============================================================

st.title("🛣️ RoadWatch AI")

st.header("Intelligent Road Damage Detection")

st.write(
    "Upload road inspection footage to detect "
    "potholes, longitudinal cracks, transverse cracks, "
    "alligator cracks and other road damage using a "
    "custom-trained YOLOv8n model."
)

st.divider()


# ============================================================
# 8. VIDEO UPLOAD INTERFACE
# ============================================================

st.subheader("Upload Road Inspection Video")

uploaded_video = st.file_uploader(
    "Choose an inspection video",
    type=["mp4", "avi", "mov", "mkv"]
)

# ------------------------------------------------------------
# DETECTION SETTINGS
# ------------------------------------------------------------

with st.expander(
    "Detection Settings",
    expanded=True
):

    # Lowest available confidence is intended for
    # diagnostic testing of weaker predictions.
    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.90,
        value=0.10,
        step=0.05
    )

    # Allow users to control inference image size.
    image_size = st.select_slider(
        "Inference Resolution",
        options=[
            256, 320, 416, 512, 640
        ],
        value=640
    )

    # Select which source frames undergo detection.
    frame_interval = st.selectbox(
        "Analyze every Nth frame",
        options=[1, 2, 5, 10],
        index=0
    )

    # Choose the number of frames to analyze.
    max_frames = st.number_input(
        "Maximum frames to analyze",
        min_value=1,
        max_value=1000,
        value=120,
        step=10
    )

    # --------------------------------------------------------
    # NEW: VIDEO PLAYBACK SPEED
    # --------------------------------------------------------

    playback_speed = st.selectbox(
        "Annotated Video Playback Speed",
        options=[0.25, 0.5, 0.75, 1.0],
        index=1,
        format_func=lambda speed: (
            f"{speed}x Speed"
        )
    )

    st.caption(
        "Slower playback increases output duration. "
        "More analyzed frames increase processing time. "
        "Sampling skips inference on intermediate frames."
    )


# ============================================================
# 9. UPLOAD PREVIEW AND ANALYSIS
# ============================================================

if uploaded_video is not None:

    # Clear old analysis if a different video is uploaded.
    upload_signature = (
        uploaded_video.name,
        uploaded_video.size
    )

    if (
        st.session_state.get("upload_signature")
        != upload_signature
    ):
        st.session_state["upload_signature"] = (
            upload_signature
        )

        st.session_state.pop(
            "roadwatch_result",
            None
        )

    st.success(
        "Video uploaded successfully."
    )

    st.subheader("Original Road Inspection Video")

    st.video(uploaded_video)

    # Display the source file size.
    st.caption(
        f"Uploaded file size: "
        f"{uploaded_video.size / (1024 * 1024):.2f} MB"
    )

    # --------------------------------------------------------
    # START INFERENCE
    # --------------------------------------------------------

    if st.button(
        "Analyze Road Video",
        type="primary"
    ):

        if (
            uploaded_video.size >
            MAX_UPLOAD_MB * 1024 * 1024
        ):
            st.error(
                f"Maximum upload size is "
                f"{MAX_UPLOAD_MB} MB."
            )

        else:

            # Remove any previous detection result.
            st.session_state.pop(
                "roadwatch_result",
                None
            )

            progress_bar = st.progress(0)
            status_text = st.empty()

            try:

                with st.spinner(
                    "Running RoadWatch AI detection..."
                ):

                    # Create temporary processing files.
                    with tempfile.TemporaryDirectory() as tmp:

                        temp_dir = Path(tmp)

                        extension = Path(
                            uploaded_video.name
                        ).suffix.lower()

                        input_path = (
                            temp_dir /
                            f"input{extension}"
                        )

                        output_path = (
                            temp_dir /
                            "roadwatch_output.mp4"
                        )

                        h264_path = (
                            temp_dir /
                            "roadwatch_h264.mp4"
                        )

                        # Save original uploaded video.
                        input_path.write_bytes(
                            uploaded_video.getvalue()
                        )

                        # ------------------------------------
                        # RUN ROAD-DAMAGE DETECTION
                        # ------------------------------------

                        stats = process_video(
                            input_path=input_path,
                            output_path=output_path,
                            confidence=confidence,
                            image_size=image_size,
                            frame_interval=frame_interval,
                            max_frames=int(max_frames),
                            playback_speed=playback_speed,
                            progress_bar=progress_bar,
                            status_text=status_text
                        )

                        # Check that MP4 was created.
                        if (
                            not output_path.is_file()
                            or output_path.stat().st_size == 0
                        ):
                            raise RuntimeError(
                                "Output video is missing."
                            )

                        # ------------------------------------
                        # CONVERT OUTPUT TO H.264
                        # ------------------------------------

                        status_text.text(
                            "Preparing annotated video..."
                        )

                        converted = convert_to_h264(
                            output_path,
                            h264_path
                        )

                        playable_path = (
                            h264_path
                            if converted
                            else output_path
                        )

                        # Keep file data after temp cleanup.
                        output_bytes = (
                            playable_path.read_bytes()
                        )

                    # ----------------------------------------
                    # STORE RESULTS IN STREAMLIT SESSION
                    # ----------------------------------------

                    st.session_state[
                        "roadwatch_result"
                    ] = {
                        "stats": stats,
                        "video": output_bytes,
                        "browser_compatible": converted,
                        "source": upload_signature,
                        "settings": {
                            "confidence": confidence,
                            "image_size": image_size,
                            "frame_interval": frame_interval,
                            "max_frames": int(max_frames),
                            "playback_speed": playback_speed
                        }
                    }

                    status_text.text(
                        "Video processing completed."
                    )

                    progress_bar.progress(1.0)

                    st.success(
                        "Road damage analysis completed!"
                    )

            except Exception as error:

                st.error(
                    "Video processing failed."
                )

                # Show traceback during capstone testing.
                st.exception(error)


# ============================================================
# 10. DETECTION RESULTS
# ============================================================

if "roadwatch_result" in st.session_state:

    saved_result = st.session_state[
        "roadwatch_result"
    ]

    # Display only for the matching uploaded video.
    if (
        uploaded_video is not None
        and saved_result["source"] ==
        (
            uploaded_video.name,
            uploaded_video.size
        )
    ):

        stats = saved_result["stats"]

        st.divider()

        st.header("Detection Results")

        # ----------------------------------------------------
        # RESULTS DASHBOARD
        # ----------------------------------------------------

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

        st.divider()

        # ----------------------------------------------------
        # VIDEO SUMMARY
        # ----------------------------------------------------

        st.subheader("Video Processing Summary")

        left, right = st.columns(2)

        left.metric(
            "Output Video Duration",
            f"{stats['output_duration']:.2f} sec"
        )

        right.metric(
            "Output Frame Rate",
            f"{stats['output_fps']:.2f} FPS"
        )

        st.caption(
            "Output duration depends on the number "
            "of analyzed frames, frame sampling, "
            "and selected playback speed."
        )

        # ----------------------------------------------------
        # DETECTED DAMAGE CATEGORIES
        # ----------------------------------------------------

        st.subheader("Detected Damage Categories")

        if stats["detections"]:

            chart_data = pd.DataFrame(
                [
                    {
                        "Damage Type": name,
                        "Detections": count
                    }
                    for name, count
                    in stats["detections"].items()
                ]
            )

            chart_data = chart_data.set_index(
                "Damage Type"
            )

            st.bar_chart(
                chart_data,
                y="Detections"
            )

        else:

            st.info(
                "No road damage exceeded the selected "
                "confidence threshold in analyzed frames."
            )

        # ----------------------------------------------------
        # DETAILED DETECTION RECORDS
        # ----------------------------------------------------

        if stats["records"]:

            st.subheader("Detection Details")

            details = pd.DataFrame(
                stats["records"]
            )

            st.dataframe(
                details,
                use_container_width=True,
                hide_index=True
            )

            # Download structured detection information.
            csv_data = details.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Detection Report (CSV)",
                data=csv_data,
                file_name="RoadWatch_AI_Detections.csv",
                mime="text/csv"
            )

        # ----------------------------------------------------
        # DISPLAY ANNOTATED VIDEO
        # ----------------------------------------------------

        st.divider()

        st.subheader("Annotated Road Inspection Video")

        if saved_result["browser_compatible"]:

            st.video(
                saved_result["video"],
                format="video/mp4"
            )

        else:

            st.warning(
                "Browser playback may not be available. "
                "Download the MP4 to view it locally."
            )

        # ----------------------------------------------------
        # DOWNLOAD ANNOTATED VIDEO
        # ----------------------------------------------------

        st.download_button(
            label="Download Annotated Video",
            data=saved_result["video"],
            file_name="RoadWatch_AI_Result.mp4",
            mime="video/mp4",
            type="primary"
        )

        st.caption(
            "Detection counts represent predictions "
            "across analyzed frames, not necessarily "
            "unique potholes or cracks."
        )

        st.caption(
            "Only selected frames are exported. "
            "The video may not cover the entire "
            "duration of the original footage."
        )


# ============================================================
# 11. FOOTER
# ============================================================

st.divider()

st.caption(
    "RoadWatch AI | CES117V Capstone Project | "
    "Powered by YOLOv8n | RDD 2022"
)
