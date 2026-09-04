import streamlit as st
import os
import cv2
import pandas as pd
import subprocess
import sys
from pathlib import Path


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Urban Intelligence Platform",
    page_icon="🛣️",
    layout="wide"
)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).parent

UPLOAD_DIR = BASE_DIR / "uploads" / "road_videos"
DATA_DIR = BASE_DIR / "data"
EVIDENCE_DIR = BASE_DIR / "evidence"

VIDEO_PATH = UPLOAD_DIR / "Road Video.mp4"

CSV_PATH = DATA_DIR / "smart_detection_results.csv"

SMART_DETECTOR_PATH = (
    BASE_DIR / "ai_engine" / "smart_pothole_detector.py"
)

INCIDENT_GENERATOR_PATH = (
    BASE_DIR / "ai_engine" / "incident_generator.py"
)


# =========================================================
# CREATE REQUIRED FOLDERS
# =========================================================

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        text-align: center;
        color: #777777;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 25px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    .card {
        padding: 20px;
        border-radius: 12px;
        background-color: #f5f5f5;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="main-title">
        🛣️ AI-Powered Urban Road Intelligence Platform
    </div>

    <div class="subtitle">
        Smart Pothole Detection • Evidence Generation • Incident Analysis
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🛠️ Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "🏠 Dashboard",
        "🎥 Road Video",
        "🤖 AI Detection",
        "🚨 Incident Analysis",
        "📸 Evidence",
        "📊 Reports"
    ]
)


# =========================================================
# FUNCTION: LOAD CSV
# =========================================================

def load_detection_data():

    if CSV_PATH.exists():

        try:

            df = pd.read_csv(CSV_PATH)

            return df

        except Exception as e:

            st.error(
                f"CSV file read nahi ho pa rahi: {e}"
            )

            return pd.DataFrame()

    return pd.DataFrame()


# =========================================================
# FUNCTION: GET VIDEO INFO
# =========================================================

def get_video_info(video_path):

    if not video_path.exists():

        return None

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():

        return None

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    duration = 0

    if fps and fps > 0:

        duration = frame_count / fps

    cap.release()

    return {
        "frames": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration": duration
    }


# =========================================================
# DASHBOARD PAGE
# =========================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="section-title">Project Dashboard</div>',
        unsafe_allow_html=True
    )

    df = load_detection_data()

    video_info = get_video_info(
        VIDEO_PATH
    )

    total_detections = 0

    if not df.empty:

        total_detections = len(df)

    evidence_count = 0

    smart_evidence_dir = (
        EVIDENCE_DIR / "smart_detections"
    )

    if smart_evidence_dir.exists():

        evidence_count = len(
            list(
                smart_evidence_dir.glob(
                    "*.jpg"
                )
            )
        )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "🎥 Video Available",
            "Yes"
            if VIDEO_PATH.exists()
            else "No"
        )

    with col2:

        st.metric(
            "🕳️ Pothole Detections",
            total_detections
        )

    with col3:

        st.metric(
            "📸 Evidence Images",
            evidence_count
        )

    with col4:

        if video_info:

            duration_text = (
                f"{video_info['duration']:.1f} sec"
            )

        else:

            duration_text = "N/A"

        st.metric(
            "⏱️ Video Duration",
            duration_text
        )


    st.divider()


    st.subheader(
        "🚀 System Overview"
    )


    st.write(
        """
        This prototype analyzes urban road videos using
        Artificial Intelligence.

        The system identifies possible potholes,
        filters detections, generates evidence images
        and groups detections into road incidents.
        """
    )


    if video_info:

        st.subheader(
            "🎬 Video Information"
        )

        info_col1, info_col2, info_col3 = (
            st.columns(3)
        )

        info_col1.metric(
            "Total Frames",
            video_info["frames"]
        )

        info_col2.metric(
            "FPS",
            f"{video_info['fps']:.2f}"
        )

        info_col3.metric(
            "Resolution",
            (
                f"{video_info['width']}"
                f" × "
                f"{video_info['height']}"
            )
        )


# =========================================================
# ROAD VIDEO PAGE
# =========================================================

elif page == "🎥 Road Video":

    st.markdown(
        '<div class="section-title">Road Video Input</div>',
        unsafe_allow_html=True
    )


    uploaded_file = st.file_uploader(
        "Upload Road Video",
        type=[
            "mp4",
            "avi",
            "mov"
        ]
    )


    if uploaded_file is not None:

        save_path = (
            UPLOAD_DIR
            / uploaded_file.name
        )

        with open(
            save_path,
            "wb"
        ) as file:

            file.write(
                uploaded_file.getbuffer()
            )

        st.success(
            "Video uploaded successfully!"
        )

        st.video(
            str(save_path)
        )


    st.divider()


    st.subheader(
        "Available Road Videos"
    )


    video_files = list(
        UPLOAD_DIR.glob(
            "*"
        )
    )


    valid_videos = []

    for file in video_files:

        if file.suffix.lower() in [
            ".mp4",
            ".avi",
            ".mov"
        ]:

            valid_videos.append(
                file
            )


    if len(valid_videos) == 0:

        st.warning(
            "No road videos found."
        )

    else:

        selected_video = st.selectbox(
            "Select Video",
            valid_videos,
            format_func=lambda x: x.name
        )

        st.video(
            str(selected_video)
        )


# =========================================================
# AI DETECTION PAGE
# =========================================================

elif page == "🤖 AI Detection":

    st.markdown(
        '<div class="section-title">AI Pothole Detection</div>',
        unsafe_allow_html=True
    )


    st.write(
        """
        This module runs the Smart Pothole Detection
        engine on extracted road frames.
        """
    )


    if st.button(
        "▶️ Run Smart Pothole Detection"
    ):

        if not SMART_DETECTOR_PATH.exists():

            st.error(
                "smart_pothole_detector.py file not found!"
            )

        else:

            with st.spinner(
                "AI analysis running..."
            ):

                try:

                    result = subprocess.run(
                        [
                            sys.executable,
                            str(
                                SMART_DETECTOR_PATH
                            )
                        ],
                        capture_output=True,
                        text=True,
                        cwd=str(
                            BASE_DIR
                        )
                    )


                    if result.returncode == 0:

                        st.success(
                            "Smart AI Analysis Completed Successfully!"
                        )

                        st.code(
                            result.stdout
                        )

                    else:

                        st.error(
                            "AI Analysis Failed"
                        )

                        st.code(
                            result.stderr
                        )

                except Exception as e:

                    st.error(
                        f"Error: {e}"
                    )


    st.divider()


    df = load_detection_data()


    if not df.empty:

        st.subheader(
            "Detection Results"
        )

        st.dataframe(
            df,
            use_container_width=True
        )

    else:

        st.info(
            "Run AI Detection to generate results."
        )


# =========================================================
# INCIDENT ANALYSIS PAGE
# =========================================================

elif page == "🚨 Incident Analysis":

    st.markdown(
        '<div class="section-title">Incident Generation</div>',
        unsafe_allow_html=True
    )


    st.write(
        """
        Multiple pothole detections occurring
        across nearby frames are grouped
        as a single road incident.
        """
    )


    if st.button(
        "🚨 Generate Incidents"
    ):

        if not INCIDENT_GENERATOR_PATH.exists():

            st.error(
                "incident_generator.py file not found!"
            )

        else:

            with st.spinner(
                "Generating incidents..."
            ):

                try:

                    result = subprocess.run(
                        [
                            sys.executable,
                            str(
                                INCIDENT_GENERATOR_PATH
                            )
                        ],
                        capture_output=True,
                        text=True,
                        cwd=str(
                            BASE_DIR
                        )
                    )


                    if result.returncode == 0:

                        st.success(
                            "Incident Analysis Completed!"
                        )

                        st.code(
                            result.stdout
                        )

                    else:

                        st.error(
                            "Incident Generation Failed"
                        )

                        st.code(
                            result.stderr
                        )

                except Exception as e:

                    st.error(
                        f"Error: {e}"
                    )


    st.divider()


    st.subheader(
        "Incident Analysis Explanation"
    )


    st.write(
        """
        Example:

        Frame 1 → Pothole Detected

        Frame 2 → Same Road Area

        Frame 3 → Same Pothole

        Instead of counting these as
        three separate potholes,
        the system can group them
        into one incident.
        """
    )


# =========================================================
# EVIDENCE PAGE
# =========================================================

elif page == "📸 Evidence":

    st.markdown(
        '<div class="section-title">AI Detection Evidence</div>',
        unsafe_allow_html=True
    )


    evidence_folders = []

    if EVIDENCE_DIR.exists():

        evidence_folders = [
            folder
            for folder in EVIDENCE_DIR.iterdir()
            if folder.is_dir()
        ]


    images = list(
        EVIDENCE_DIR.glob(
            "*.jpg"
        )
    )


    for folder in evidence_folders:

        images.extend(
            folder.glob(
                "*.jpg"
            )
        )


    if len(images) == 0:

        st.info(
            "No evidence images available yet."
        )

    else:

        st.success(
            f"{len(images)} evidence images found."
        )


        columns = st.columns(
            3
        )


        for index, image_path in enumerate(
            images
        ):

            with columns[
                index % 3
            ]:

                st.image(
                    str(image_path),
                    caption=image_path.name,
                    use_container_width=True
                )


# =========================================================
# REPORTS PAGE
# =========================================================

elif page == "📊 Reports":

    st.markdown(
        '<div class="section-title">Detection Reports</div>',
        unsafe_allow_html=True
    )


    df = load_detection_data()


    if df.empty:

        st.warning(
            "No detection report available."
        )

    else:

        st.subheader(
            "Smart Detection CSV Report"
        )


        st.dataframe(
            df,
            use_container_width=True
        )


        csv_data = df.to_csv(
            index=False
        ).encode(
            "utf-8"
        )


        st.download_button(
            label="⬇️ Download Detection Report",
            data=csv_data,
            file_name="urban_intelligence_report.csv",
            mime="text/csv"
        )


        st.divider()


        st.subheader(
            "Detection Statistics"
        )


        st.metric(
            "Total Records",
            len(df)
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()


st.markdown(
    """
    <center>

    <b>SIH26124 – AI-Powered Urban Intelligence Platform</b>

    <br>

    Prototype developed for Smart India Hackathon

    </center>
    """,
    unsafe_allow_html=True
)