
# ============================================================
# ROADWATCH AI - CES117V CAPSTONE PROJECT
# STREAMLIT DEPLOYMENT TEST
# ============================================================

# Import Streamlit to build the application interface.
import streamlit as st

# Configure the application page.
st.set_page_config(
    page_title="RoadWatch AI",
    page_icon="🛣️",
    layout="wide"
)

# Display the application heading.
st.title("RoadWatch AI")

st.subheader("Intelligent Road Damage Detection")

st.write(
    "Upload road inspection footage to identify potholes, "
    "longitudinal cracks, transverse cracks, alligator "
    "cracks and other road damage using YOLOv8n."
)

# Create the video upload component.
uploaded_video = st.file_uploader(
    "Upload Road Inspection Video",
    type=["mp4", "avi", "mov", "mkv"]
)

# Display the uploaded video.
if uploaded_video is not None:
    st.success("Video uploaded successfully.")

    st.video(uploaded_video)

# Display the project information.
st.caption(
    "RoadWatch AI | CES117V Capstone Project | "
    "Powered by YOLOv8n"
)
