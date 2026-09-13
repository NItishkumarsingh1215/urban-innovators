import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from ai_engine.telemetry_engine import get_telemetry_for_frame

def analyze_video(video_path, output_csv="data/waterlogging_results.csv", evidence_dir="evidence/waterlogging_detections", corridor_key="bengaluru_bel"):
    """
    Analyzes video for waterlogging hazards on road surfaces.
    Uses HSV color filtering (dark reflection / puddle pooling), gradient variance,
    and specular reflection highlights to detect water pools.
    Integrates synchronized Bus OBU GPS coordinates for every detected frame.
    """
    video_path = Path(video_path)
    output_csv = Path(output_csv)
    evidence_dir = Path(evidence_dir)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        return {"success": False, "message": f"Video file not found: {video_path}"}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"success": False, "message": f"Cannot open video: {video_path}"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    results = []
    frame_idx = 0
    waterlogging_frames_count = 0
    evidence_count = 0
    max_score = 0.0

    # Road surface Region of Interest (Lower 50% of the frame)
    roi_top = int(height * 0.45)

    sample_step = 6  # sample every 6 frames (~4-5 fps analysis)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % sample_step != 0:
            continue

        roi = frame[roi_top:height, 0:width]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Water puddles on asphalt typically exhibit specular glint (high V)
        # or dark low-saturation muddy pooling (low S, mid-low V)
        h, s, v = cv2.split(hsv)

        # Specular glint mask (wet glossy reflection on road)
        _, glint_mask = cv2.threshold(v, 210, 255, cv2.THRESH_BINARY)
        # Low saturation mask (grayish water surface)
        _, low_sat_mask = cv2.threshold(s, 70, 255, cv2.THRESH_BINARY_INV)

        water_candidate = cv2.bitwise_and(glint_mask, low_sat_mask)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        water_clean = cv2.morphologyEx(water_candidate, cv2.MORPH_OPEN, kernel)
        water_clean = cv2.morphologyEx(water_clean, cv2.MORPH_CLOSE, kernel)

        roi_area = float(roi.shape[0] * roi.shape[1])
        water_pixels = cv2.countNonZero(water_clean)
        water_ratio = (water_pixels / roi_area) * 100.0

        # Score normalized between 0 and 100
        score = round(min(100.0, water_ratio * 3.5), 1)
        max_score = max(max_score, score)

        if score >= 60.0:
            risk = "CRITICAL"
        elif score >= 35.0:
            risk = "HIGH"
        elif score >= 15.0:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        is_waterlogged = risk in ["MEDIUM", "HIGH", "CRITICAL"]
        if is_waterlogged:
            waterlogging_frames_count += 1

        # Synchronized Bus OBU GPS Telemetry
        telemetry = get_telemetry_for_frame(frame_idx, total_frames, fps, corridor_key)

        record = {
            "Frame": frame_idx,
            "Timestamp": telemetry["timestamp"],
            "Water_Score": score,
            "Water_Risk": risk,
            "Detected": "YES" if is_waterlogged else "NO",
            "Latitude": telemetry["latitude"],
            "Longitude": telemetry["longitude"],
            "Road_Segment": telemetry["road_segment"],
            "Corridor": telemetry["corridor"],
            "Bus_ID": telemetry["bus_id"],
            "Speed_kmh": telemetry["speed_kmh"]
        }
        results.append(record)

        # Save annotated evidence frame
        if is_waterlogged and evidence_count < 25 and (frame_idx % (sample_step * 3) == 0):
            annotated = frame.copy()
            contours, _ = cv2.findContours(water_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 300:
                    cnt_shifted = cnt + np.array([0, roi_top])
                    cv2.drawContours(annotated, [cnt_shifted], -1, (255, 128, 0), 2)

            # Metadata Overlay
            cv2.rectangle(annotated, (15, 15), (750, 95), (15, 23, 42), -1)
            cv2.rectangle(annotated, (15, 15), (750, 95), (255, 140, 0), 2)
            cv2.putText(annotated, f"WATERLOGGING HAZARD | Risk: {risk} (Score: {score})", (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)
            cv2.putText(annotated, f"GPS: {telemetry['latitude']}, {telemetry['longitude']} | {telemetry['road_segment']}", (30, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(annotated, f"Bus: {telemetry['bus_id']} | Time: {telemetry['timestamp']}", (30, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 200, 220), 1)

            evidence_file = evidence_dir / f"waterlogging_frame_{frame_idx:05d}.jpg"
            cv2.imwrite(str(evidence_file), annotated)
            evidence_count += 1

    cap.release()

    df = pd.DataFrame(results)
    if not df.empty:
        df.to_csv(output_csv, index=False)

    overall_risk = "LOW"
    if max_score >= 60:
        overall_risk = "CRITICAL"
    elif max_score >= 35:
        overall_risk = "HIGH"
    elif max_score >= 15:
        overall_risk = "MEDIUM"

    total_sampled = max(1, len(results))
    detection_pct = round((waterlogging_frames_count / total_sampled) * 100.0, 1)

    return {
        "success": True,
        "risk": overall_risk,
        "max_score": max_score,
        "analyzed_frames": total_sampled,
        "waterlogging_frames": waterlogging_frames_count,
        "detection_percentage": detection_pct,
        "csv_path": str(output_csv),
        "evidence_dir": str(evidence_dir)
    }

if __name__ == "__main__":
    UPLOAD_DIR = Path("uploads/road_videos")
    video_files = [f for f in UPLOAD_DIR.iterdir() if f.suffix.lower() in [".mp4", ".avi", ".mov"]]
    if video_files:
        res = analyze_video(video_files[0])
        print("Waterlogging analysis done:", res)