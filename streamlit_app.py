
# ============================================================
# ROADWATCH AI - CES117V CAPSTONE PROJECT
# FLASK BACKEND: app.py
# ============================================================

# Import Flask tools for creating the web application.
from flask import Flask, render_template, request, send_file

# Import YOLO for road-damage detection.
from ultralytics import YOLO

# Import OpenCV for processing video frames.
import cv2

# Import utilities for file and directory handling.
from pathlib import Path
from werkzeug.utils import secure_filename
from uuid import uuid4

# ============================================================
# 1. INITIALIZE THE FLASK APPLICATION
# ============================================================

app = Flask(__name__)

# Limit uploaded files to 100 MB for resource management.
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

# Identify the project directory.
BASE_DIR = Path(__file__).resolve().parent

# Create folders for uploaded and processed videos.
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Define permitted video file extensions.
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

# ============================================================
# 2. LOAD THE TRAINED YOLOv8n MODEL
# ============================================================

# Load the model trained using the RDD 2022 dataset.
MODEL_PATH = BASE_DIR / "best.pt"

model = YOLO(str(MODEL_PATH))

# ============================================================
# 3. VALIDATE UPLOADED FILE TYPES
# ============================================================

def allowed_file(filename):
    # Check that the filename has a supported extension.
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )

# ============================================================
# 4. PROCESS THE ROAD INSPECTION VIDEO
# ============================================================

def process_video(input_path, output_path):

    # Open the uploaded video for reading.
    capture = cv2.VideoCapture(str(input_path))

    # Verify that OpenCV opened the video.
    if not capture.isOpened():
        raise ValueError("Unable to open uploaded video.")

    # Read video properties.
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Use a fallback frame rate if the source is invalid.
    if fps <= 0 or fps > 120:
        fps = 25.0

    # Validate the dimensions before creating the output.
    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError("Invalid video dimensions.")

    # Configure the MP4 video encoder.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Create the annotated output video.
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Unable to create output video.")

    # Track the number of processed frames.
    frame_count = 0

    try:
        # Process the uploaded video one frame at a time.
        while True:

            # Read the next frame.
            success, frame = capture.read()

            # Stop when the video ends.
            if not success:
                break

            # Run road-damage detection on this frame.
            # Run RoadWatch AI object detection.
            # A lower confidence threshold helps investigate
            # weak detections that the model may otherwise reject.
            results = model.predict(
                source=frame,
                conf=0.10,
                imgsz=512,
                verbose=False
            )

            # Count the detections in the current frame.
            detection_count = len(results[0].boxes)

            # Print detections to the terminal for debugging.
            if detection_count > 0:
                print(
                    f"Frame {frame_count}: "
                    f"{detection_count} road-damage detections"
                )

            # Draw bounding boxes and class labels.
            annotated_frame = results[0].plot()

            # Write the annotated frame to the output video.
            writer.write(annotated_frame)

            # Update the processed frame counter.
            frame_count += 1

    finally:
        # Release video resources even if an error occurs.
        capture.release()
        writer.release()

    if frame_count == 0:
        raise ValueError("No video frames could be processed.")

    return frame_count

# ============================================================
# 5. HOME PAGE
# ============================================================

@app.route("/")
def index():
    # Display the HTML video-upload interface.
    return render_template("index.html")

# ============================================================
# 6. HANDLE VIDEO UPLOADS
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    # Check whether the request contains a video file.
    if "video" not in request.files:
        return {"error": "No video uploaded."}, 400

    uploaded_file = request.files["video"]

    # Confirm a filename was supplied.
    if not uploaded_file.filename:
        return {"error": "Please select a video."}, 400

    # Reject unsupported file extensions.
    if not allowed_file(uploaded_file.filename):
        return {"error": "Unsupported video format."}, 400

    # Sanitize the original filename.
    original_name = secure_filename(
        uploaded_file.filename
    )

    # Create unique filenames to avoid collisions.
    file_id = uuid4().hex

    extension = Path(original_name).suffix.lower()

    input_path = UPLOAD_DIR / f"{file_id}{extension}"
    output_path = OUTPUT_DIR / f"{file_id}.mp4"

    try:
        # Save the uploaded video.
        uploaded_file.save(input_path)

        # Perform YOLOv8n road-damage detection.
        frames = process_video(
            input_path,
            output_path
        )

        # Return the download link to the frontend.
        return {
            "message": "Video processed successfully.",
            "frames_processed": frames,
            "download_url": f"/download/{file_id}"
        }

    except Exception:
        # Avoid exposing internal server details to users.
        app.logger.exception("Video processing failed.")

        # Remove incomplete output files.
        output_path.unlink(missing_ok=True)

        return {
            "error": "Unable to process this video."
        }, 500

    finally:
        # Remove the original video after processing.
        input_path.unlink(missing_ok=True)

# ============================================================
# 7. DOWNLOAD ANNOTATED VIDEO
# ============================================================

@app.route("/download/<file_id>")
def download(file_id):

    # Reject invalid identifiers.
    if len(file_id) != 32 or any(
        char not in "0123456789abcdef" for char in file_id
    ):
        return {"error": "Invalid file identifier."}, 400

    # Locate the generated annotated video.
    output_path = OUTPUT_DIR / f"{file_id}.mp4"

    # Check that the processed video exists.
    if not output_path.is_file():
        return {"error": "Video not found."}, 404

    # Download the annotated MP4 file.
    return send_file(
        output_path,
        as_attachment=True,
        download_name="RoadWatch_AI_Result.mp4",
        mimetype="video/mp4"
    )

# ============================================================
# 8. START THE APPLICATION
# ============================================================

if __name__ == "__main__":

    # Run the development server for local testing.
    app.run(
        host="0.0.0.0",
        port=7860,
        debug=False
    )
