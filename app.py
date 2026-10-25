
# ============================================================
# ROADWATCH AI - CES117V CAPSTONE PROJECT
# OPTIMIZED FLASK BACKEND FOR RENDER FREE
# ============================================================

# Flask provides the web interface and API endpoints.
from flask import Flask, render_template, request, send_file

# YOLO performs road-damage object detection.
from ultralytics import YOLO

# OpenCV reads videos and creates annotated output videos.
import cv2

# Standard Python utilities.
import os
import gc
import logging
import threading
from pathlib import Path
from uuid import uuid4

# Secure uploaded filenames.
from werkzeug.utils import secure_filename

# ============================================================
# 1. RESOURCE OPTIMIZATION
# ============================================================

# Limit native CPU threading on the small Render instance.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Allow OpenCV to use only one processing thread.
cv2.setNumThreads(1)

# Run object detection on every tenth input video frame.
FRAME_SKIP = 10

# Limit the number of actual YOLO predictions per upload.
# For a first Render test, this can be set to 1 or 2
# using the MAX_INFERENCE_FRAMES environment variable.
MAX_INFERENCE_FRAMES = max(
    1, int(os.getenv("MAX_INFERENCE_FRAMES", "5"))
)

# Reduce YOLO input resolution to lower CPU workload.
INFERENCE_SIZE = 256

# Maximum output image dimension in pixels.
MAX_OUTPUT_WIDTH = 640

# Restrict upload size for initial online demonstrations.
MAX_UPLOAD_MB = 20

# Limit each request to one processing operation at a time.
processing_lock = threading.Lock()

# ============================================================
# 2. INITIALIZE FLASK
# ============================================================

app = Flask(__name__)

# Restrict the maximum uploaded file size.
app.config["MAX_CONTENT_LENGTH"] = (
    MAX_UPLOAD_MB * 1024 * 1024
)

# Enable informative application logging.
logging.basicConfig(level=logging.INFO)

# Locate the project directory.
BASE_DIR = Path(__file__).resolve().parent

# Define upload and output directories.
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

# Create the required folders if they do not exist.
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Accepted input video formats.
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

# ============================================================
# 3. LOAD THE TRAINED ROAD DAMAGE MODEL
# ============================================================

# Use the model trained with the RDD 2022 dataset.
MODEL_PATH = BASE_DIR / "best.pt"

# Stop startup if the trained model does not exist.
if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"RoadWatch AI model not found: {MODEL_PATH}"
    )

# Load the model once instead of reloading it per request.
model = YOLO(str(MODEL_PATH))

# Run inference using CPU hardware.
MODEL_DEVICE = "cpu"

# ============================================================
# 4. VALIDATE VIDEO FILE EXTENSIONS
# ============================================================

def allowed_file(filename):
    """Check whether the upload has an allowed extension."""

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# 5. RESIZE FRAMES FOR LOWER RESOURCE USAGE
# ============================================================

def resize_frame(frame):
    """Resize large frames while preserving aspect ratio."""

    height, width = frame.shape[:2]

    # Keep smaller frames unchanged.
    if width <= MAX_OUTPUT_WIDTH:
        return frame

    # Calculate proportional output dimensions.
    scale = MAX_OUTPUT_WIDTH / width

    new_width = MAX_OUTPUT_WIDTH
    new_height = max(1, int(height * scale))

    return cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )


# ============================================================
# 6. PROCESS VIDEO USING FRAME SAMPLING
# ============================================================

def process_video(input_path, output_path):
    """
    Process a small sample of frames from the video.

    This approach reduces inference operations and creates
    a short annotated MP4 containing sampled frames.

    It does not perform detection on every source frame.
    """

    # Open the uploaded video.
    capture = cv2.VideoCapture(str(input_path))

    if not capture.isOpened():
        raise ValueError("Unable to open uploaded video.")

    writer = None

    # Count frames read from the original video.
    frames_read = 0

    # Count frames actually analyzed by YOLO.
    frames_analyzed = 0

    # Count bounding-box detections across sampled frames.
    total_detections = 0

    try:
        # Read original video properties.
        fps = capture.get(cv2.CAP_PROP_FPS)

        if not (0 < fps <= 120):
            fps = 25.0

        # Set output FPS to preserve approximate video timing.
        # For example, sampling 30 FPS every 10 frames gives
        # an output video frame rate of 3 FPS.
        output_fps = max(1.0, fps / FRAME_SKIP)

        app.logger.info(
            "Starting video processing: FPS=%.2f, "
            "sampling interval=%d, max predictions=%d",
            fps,
            FRAME_SKIP,
            MAX_INFERENCE_FRAMES
        )

        # Read frames sequentially.
        while frames_analyzed < MAX_INFERENCE_FRAMES:

            success, frame = capture.read()

            # Stop when the input video ends.
            if not success:
                break

            # Track original frame index.
            current_index = frames_read
            frames_read += 1

            # Skip frames not selected for inference.
            if current_index % FRAME_SKIP != 0:
                continue

            # Resize the sampled frame to reduce memory use.
            frame = resize_frame(frame)

            # Create the output writer using actual frame size.
            if writer is None:
                height, width = frame.shape[:2]

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                writer = cv2.VideoWriter(
                    str(output_path),
                    fourcc,
                    output_fps,
                    (width, height)
                )

                if not writer.isOpened():
                    raise RuntimeError(
                        "Unable to create annotated MP4."
                    )

            # =================================================
            # RUN YOLOv8n ROAD-DAMAGE DETECTION
            # =================================================

            # Use a small inference image and CPU execution.
            # max_det limits output boxes, not processing frames.
            results = model.predict(
                source=frame,
                imgsz=INFERENCE_SIZE,
                conf=0.25,
                device=MODEL_DEVICE,
                max_det=30,
                verbose=False
            )

            # Count detections from the current sampled frame.
            detection_count = len(results[0].boxes)

            total_detections += detection_count

            # Draw boxes and detected road-damage labels.
            annotated_frame = results[0].plot()

            # Ensure consistent frame dimensions for MP4.
            if annotated_frame.shape[:2] != (height, width):
                annotated_frame = cv2.resize(
                    annotated_frame,
                    (width, height)
                )

            # Write the annotated sampled frame.
            writer.write(annotated_frame)

            # Update the number of predictions performed.
            frames_analyzed += 1

            # Log progress for diagnosis in Render.
            app.logger.info(
                "Sample %d/%d | Source frame %d | "
                "Detections: %d",
                frames_analyzed,
                MAX_INFERENCE_FRAMES,
                current_index,
                detection_count
            )

            # Release intermediate inference results.
            del results
            del annotated_frame

        # Reject videos without readable frames.
        if frames_analyzed == 0:
            raise ValueError(
                "No readable video frames were found."
            )

    finally:
        # Close resources even when processing fails.
        capture.release()

        if writer is not None:
            writer.release()

        # Run garbage collection after video processing.
        gc.collect()

    # Report the processing statistics.
    app.logger.info(
        "Processing completed. Frames read=%d, "
        "analyzed=%d, detections=%d",
        frames_read,
        frames_analyzed,
        total_detections
    )

    return {
        "frames_read": frames_read,
        "frames_analyzed": frames_analyzed,
        "total_detections": total_detections
    }


# ============================================================
# 7. HOME PAGE
# ============================================================

@app.route("/")
def index():
    """Display the RoadWatch AI upload interface."""

    return render_template("index.html")


# ============================================================
# 8. VIDEO UPLOAD AND DETECTION API
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():
    """Receive and analyze an uploaded road inspection video."""

    # Confirm that the upload field exists.
    if "video" not in request.files:
        return {"error": "No video uploaded."}, 400

    uploaded_file = request.files["video"]

    # Confirm a filename was supplied.
    if not uploaded_file.filename:
        return {"error": "Please select a video."}, 400

    # Validate the video extension.
    if not allowed_file(uploaded_file.filename):
        return {
            "error": "Unsupported video format."
        }, 400

    # Prevent simultaneous model inference requests.
    if not processing_lock.acquire(blocking=False):
        return {
            "error": (
                "RoadWatch AI is currently processing "
                "another video. Please try again later."
            )
        }, 503

    # Generate safe, unique file paths.
    original_name = secure_filename(
        uploaded_file.filename
    )

    file_id = uuid4().hex

    extension = Path(original_name).suffix.lower()

    input_path = UPLOAD_DIR / f"{file_id}{extension}"
    output_path = OUTPUT_DIR / f"{file_id}.mp4"

    try:
        # Save the uploaded inspection video.
        uploaded_file.save(input_path)

        # Run frame-sampled road-damage inference.
        statistics = process_video(
            input_path,
            output_path
        )

        # Confirm the annotated video was generated.
        if (
            not output_path.is_file()
            or output_path.stat().st_size == 0
        ):
            raise RuntimeError(
                "Processed video was not generated."
            )

        # Return JSON compatible with the existing frontend.
        return {
            "message": "Video processed successfully.",
            "frames_processed": statistics["frames_analyzed"],
            "frames_read": statistics["frames_read"],
            "frames_analyzed": statistics["frames_analyzed"],
            "total_detections": statistics["total_detections"],
            "download_url": f"/download/{file_id}"
        }, 200

    except Exception:
        # Record detailed errors privately in Render logs.
        app.logger.exception(
            "RoadWatch AI video processing failed."
        )

        # Remove any partial output.
        output_path.unlink(missing_ok=True)

        return {
            "error": "Unable to process this video."
        }, 500

    finally:
        # Delete the uploaded source video.
        input_path.unlink(missing_ok=True)

        # Allow the next video request to be processed.
        processing_lock.release()


# ============================================================
# 9. DOWNLOAD ANNOTATED VIDEO
# ============================================================

@app.route("/download/<file_id>")
def download(file_id):
    """Download a processed, annotated MP4 video."""

    # Validate the unique identifier.
    if len(file_id) != 32 or any(
        char not in "0123456789abcdef"
        for char in file_id
    ):
        return {
            "error": "Invalid file identifier."
        }, 400

    # Find the generated video.
    output_path = OUTPUT_DIR / f"{file_id}.mp4"

    if not output_path.is_file():
        return {
            "error": "Video not found."
        }, 404

    # Send the annotated MP4 to the browser.
    return send_file(
        output_path,
        as_attachment=True,
        download_name="RoadWatch_AI_Result.mp4",
        mimetype="video/mp4"
    )


# ============================================================
# 10. HANDLE OVERSIZED UPLOADS
# ============================================================

@app.errorhandler(413)
def file_too_large(error):
    """Return JSON instead of an HTML upload-size error."""

    return {
        "error": (
            f"Video exceeds the {MAX_UPLOAD_MB} MB "
            "upload limit."
        )
    }, 413


# ============================================================
# 11. START THE APPLICATION
# ============================================================

if __name__ == "__main__":

    # Run Flask locally for development and testing.
    # Render starts the production app using Gunicorn.
    app.run(
        host="0.0.0.0",
        port=7860,
        debug=False
    )
