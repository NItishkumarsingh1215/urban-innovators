# ============================================================
# WATERLOGGING DETECTOR
# SIH26124 - AI-Powered Mobile Urban Intelligence Platform
# ============================================================

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


# ============================================================
# FRAME ANALYSIS
# ============================================================

def analyze_frame(frame):
    """
    Analyze a single video frame for possible waterlogging.

    Returns:
        detected: True / False
        score: 0 - 100
    """

    if frame is None:
        return False, 0

    # Resize frame for faster processing
    frame = cv2.resize(
        frame,
        (640, 360)
    )

    # Convert BGR -> HSV
    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    # ========================================================
    # WATER COLOR DETECTION
    # ========================================================

    # Blue / cyan water
    lower_blue = np.array(
        [80, 25, 40]
    )

    upper_blue = np.array(
        [140, 255, 255]
    )

    blue_mask = cv2.inRange(
        hsv,
        lower_blue,
        upper_blue
    )

    # ========================================================
    # REFLECTIVE / LIGHT WATER DETECTION
    # ========================================================

    lower_reflection = np.array(
        [0, 0, 80]
    )

    upper_reflection = np.array(
        [180, 70, 255]
    )

    reflection_mask = cv2.inRange(
        hsv,
        lower_reflection,
        upper_reflection
    )

    # ========================================================
    # REMOVE NOISE
    # ========================================================

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    blue_mask = cv2.morphologyEx(
        blue_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    blue_mask = cv2.morphologyEx(
        blue_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    reflection_mask = cv2.morphologyEx(
        reflection_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # ========================================================
    # TOTAL AREA
    # ========================================================

    height, width = frame.shape[:2]

    total_pixels = height * width

    blue_area = (
        cv2.countNonZero(blue_mask)
        / total_pixels
    )

    reflection_area = (
        cv2.countNonZero(reflection_mask)
        / total_pixels
    )

    # ========================================================
    # ROAD AREA
    # Mainly analyze lower part of image
    # ========================================================

    road_start = int(
        height * 0.45
    )

    road_blue = blue_mask[
        road_start:,
        :
    ]

    road_reflection = reflection_mask[
        road_start:,
        :
    ]

    road_pixels = (
        road_blue.shape[0]
        * road_blue.shape[1]
    )

    road_blue_area = (
        cv2.countNonZero(road_blue)
        / road_pixels
    )

    road_reflection_area = (
        cv2.countNonZero(road_reflection)
        / road_pixels
    )

    # ========================================================
    # WATERLOGGING SCORE
    # ========================================================

    score = 0

    # Blue water area
    if blue_area >= 0.08:

        score += 35

    elif blue_area >= 0.04:

        score += 25

    elif blue_area >= 0.02:

        score += 12

    # Blue area on road
    if road_blue_area >= 0.10:

        score += 30

    elif road_blue_area >= 0.05:

        score += 20

    elif road_blue_area >= 0.02:

        score += 10

    # Reflection
    if reflection_area >= 0.25:

        score += 20

    elif reflection_area >= 0.15:

        score += 10

    # Reflection on road
    if road_reflection_area >= 0.30:

        score += 15

    elif road_reflection_area >= 0.15:

        score += 8

    # Maximum score = 100
    score = min(
        score,
        100
    )

    # Detection threshold
    detected = score >= 35

    return detected, score


# ============================================================
# VIDEO ANALYSIS
# ============================================================

def analyze_video(
    video_path,
    output_csv="data/waterlogging_results.csv",
    evidence_dir="evidence/waterlogging_detections",
    sample_every=15
):
    """
    Analyze uploaded road video.

    Parameters:
        video_path:
            Path of uploaded video.

        output_csv:
            CSV file where results are stored.

        evidence_dir:
            Folder where detected frames are saved.

        sample_every:
            Analyze every Nth frame.
    """

    video_path = Path(
        video_path
    )

    output_csv = Path(
        output_csv
    )

    evidence_dir = Path(
        evidence_dir
    )

    # ========================================================
    # CREATE DIRECTORIES
    # ========================================================

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    evidence_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # CHECK VIDEO
    # ========================================================

    if not video_path.exists():

        return {
            "success": False,
            "message": "Video file not found.",
            "total_frames": 0,
            "analyzed_frames": 0,
            "waterlogging_frames": 0,
            "max_score": 0,
            "risk": "UNKNOWN"
        }

    # ========================================================
    # OPEN VIDEO
    # ========================================================

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        return {
            "success": False,
            "message": "Unable to open video.",
            "total_frames": 0,
            "analyzed_frames": 0,
            "waterlogging_frames": 0,
            "max_score": 0,
            "risk": "UNKNOWN"
        }

    # ========================================================
    # VIDEO INFORMATION
    # ========================================================

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 25

    # ========================================================
    # VARIABLES
    # ========================================================

    frame_number = 0

    analyzed_frames = 0

    waterlogging_frames = 0

    max_score = 0

    evidence_count = 0

    results = []

    # ========================================================
    # READ VIDEO
    # ========================================================

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        # ----------------------------------------------------
        # Analyze every Nth frame
        # ----------------------------------------------------

        if frame_number % sample_every != 0:
            continue

        analyzed_frames += 1

        # ----------------------------------------------------
        # Analyze frame
        # ----------------------------------------------------

        detected, score = analyze_frame(
            frame
        )

        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        timestamp = (
            frame_number / fps
        )

        # ----------------------------------------------------
        # Count detection
        # ----------------------------------------------------

        if detected:

            waterlogging_frames += 1

        # ----------------------------------------------------
        # Maximum score
        # ----------------------------------------------------

        if score > max_score:

            max_score = score

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        results.append({

            "Timestamp": round(
                timestamp,
                2
            ),

            "Frame": frame_number,

            "Waterlogging_Detected":
                "YES" if detected else "NO",

            "Score": score
        })

        # ====================================================
        # SAVE EVIDENCE FRAME
        # ====================================================

        if detected and evidence_count < 30:

            evidence_file = (
                evidence_dir
                / f"waterlogging_frame_{frame_number}.jpg"
            )

            # Add visual label
            evidence_frame = frame.copy()

            cv2.putText(
                evidence_frame,
                f"Waterlogging Score: {score}%",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

            cv2.putText(
                evidence_frame,
                "WATERLOGGING DETECTED",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

            cv2.imwrite(
                str(evidence_file),
                evidence_frame
            )

            evidence_count += 1

    # ========================================================
    # CLOSE VIDEO
    # ========================================================

    cap.release()

    # ========================================================
    # CALCULATE RISK
    # ========================================================

    if analyzed_frames == 0:

        risk = "UNKNOWN"

        detection_percentage = 0

    else:

        detection_percentage = (
            waterlogging_frames
            / analyzed_frames
        ) * 100

        # HIGH
        if (
            max_score >= 70
            or detection_percentage >= 40
        ):

            risk = "HIGH"

        # MEDIUM
        elif (
            max_score >= 45
            or detection_percentage >= 20
        ):

            risk = "MEDIUM"

        # LOW
        else:

            risk = "LOW"

    # ========================================================
    # SAVE CSV
    # ========================================================

    df = pd.DataFrame(
        results
    )

    if not df.empty:

        df["Risk_Level"] = risk

        df["Detection_Percentage"] = round(
            detection_percentage,
            2
        )

        df["Analysis_Time"] = (
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        df.to_csv(
            output_csv,
            index=False
        )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    return {

        "success": True,

        "message":
            "Video analysis completed successfully.",

        "total_frames":
            total_frames,

        "analyzed_frames":
            analyzed_frames,

        "waterlogging_frames":
            waterlogging_frames,

        "max_score":
            max_score,

        "detection_percentage":
            round(
                detection_percentage,
                2
            ),

        "risk":
            risk,

        "csv_path":
            str(output_csv),

        "evidence_dir":
            str(evidence_dir)
    }


# ============================================================
# TEST MODE
# ============================================================

if __name__ == "__main__":

    print(
        "\n======================================"
    )

    print(
        "🌧️ WATERLOGGING VIDEO DETECTOR"
    )

    print(
        "======================================\n"
    )

    video = input(
        "Enter road video path: "
    ).strip()

    result = analyze_video(
        video
    )

    print(
        "\n========== RESULT =========="
    )

    print(
        "Status:",
        result["success"]
    )

    print(
        "Risk:",
        result["risk"]
    )

    print(
        "Maximum Score:",
        result["max_score"]
    )

    print(
        "Analyzed Frames:",
        result["analyzed_frames"]
    )

    print(
        "Waterlogging Frames:",
        result["waterlogging_frames"]
    )

    print(
        "Detection Percentage:",
        result.get(
            "detection_percentage",
            0
        ),
        "%"
    )

    print(
        "CSV:",
        result["csv_path"]
    )

    print(
        "Evidence:",
        result["evidence_dir"]
    )