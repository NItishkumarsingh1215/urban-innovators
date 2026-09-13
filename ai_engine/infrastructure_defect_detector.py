import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from ai_engine.telemetry_engine import get_telemetry_for_frame

def analyze_infrastructure(video_path, output_csv="data/infrastructure_defects.csv", evidence_dir="evidence/infrastructure_detections", corridor_key="bengaluru_bel"):
    """
    Analyzes video for Urban Infrastructure Deficiencies as specified in BEL PS 26124:
    1. Missing / Broken Road Dividers & Median Barriers
    2. Missing / Faded Zebra Crossings
    3. Missing / Damaged Traffic Signboards
    Associates exact frame-by-frame GPS coordinates from Bus OBU Telemetry.
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
    evidence_count = 0
    sample_step = 10

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % sample_step != 0:
            continue

        telemetry = get_telemetry_for_frame(frame_idx, total_frames, fps, corridor_key)
        annotated = frame.copy()
        defects_found = []

        # -------------------------------------------------------------
        # 1. Road Surface Zebra Crossing & Lane / Divider Analysis
        # -------------------------------------------------------------
        roi_road = frame[int(height * 0.55):height, int(width * 0.15):int(width * 0.85)]
        gray_road = cv2.cvtColor(roi_road, cv2.COLOR_BGR2GRAY)
        
        # High contrast white thresholding for zebra markings and divider paint
        _, thresh_white = cv2.threshold(gray_road, 195, 255, cv2.THRESH_BINARY)
        
        # Horizontal gradient / stripe projection to detect zebra crossing patterns
        horizontal_proj = np.sum(thresh_white, axis=1) / (thresh_white.shape[1] * 255.0)
        stripe_peaks = np.where(horizontal_proj > 0.15)[0]

        # Median / Divider Region (Center-left lane divider inspection)
        divider_strip = roi_road[:, int(roi_road.shape[1] * 0.45):int(roi_road.shape[1] * 0.55)]
        divider_gray = cv2.cvtColor(divider_strip, cv2.COLOR_BGR2GRAY)
        divider_edges = cv2.Canny(divider_gray, 50, 150)
        divider_edge_density = np.count_nonzero(divider_edges) / float(divider_edges.size)

        # -------------------------------------------------------------
        # 2. Roadside Signboard Health (Upper peripheral regions)
        # -------------------------------------------------------------
        sign_roi = frame[int(height * 0.1):int(height * 0.5), int(width * 0.75):width]
        hsv_sign = cv2.cvtColor(sign_roi, cv2.COLOR_BGR2HSV)
        
        # Detect circular or triangular yellow/red caution sign hues
        lower_caution = np.array([15, 100, 100])
        upper_caution = np.array([35, 255, 255])
        sign_mask = cv2.inRange(hsv_sign, lower_caution, upper_caution)
        sign_pixels = cv2.countNonZero(sign_mask)

        # -------------------------------------------------------------
        # Heuristic Defect Classification
        # -------------------------------------------------------------
        # Check for Faded Zebra Crossing in pedestrian school/hospital zones
        is_ped_zone = "School" in telemetry["road_segment"] or "Crossing" in telemetry["road_segment"]
        if is_ped_zone:
            # If in pedestrian zone but zebra stripe density is low (< 0.08)
            if len(stripe_peaks) < 5:
                defects_found.append({
                    "type": "FADED / MISSING ZEBRA CROSSING",
                    "severity": "HIGH",
                    "action": "Urgent Thermoplastic Paint Repainting Required (PWD)",
                    "color": (0, 165, 255)
                })

        # Check for Missing Road Divider / Damaged Median
        if divider_edge_density < 0.015 and frame_idx % 40 == 0:
            defects_found.append({
                "type": "MISSING / DAMAGED ROAD DIVIDER",
                "severity": "MEDIUM",
                "action": "Median Barrier / Cat-Eye Reflector Installation (NHAI)",
                "color": (0, 0, 255)
            })

        # Check for Damaged / Obstructed Signboard
        if sign_pixels > 80 and sign_pixels < 350:
            defects_found.append({
                "type": "DAMAGED / WEATHERED TRAFFIC SIGNBOARD",
                "severity": "LOW",
                "action": "Signboard Surface Replacement & Pruning (Traffic Police)",
                "color": (255, 191, 0)
            })

        for defect in defects_found:
            results.append({
                "Frame": frame_idx,
                "Timestamp": telemetry["timestamp"],
                "Defect_Type": defect["type"],
                "Severity": defect["severity"],
                "Recommended_Action": defect["action"],
                "Latitude": telemetry["latitude"],
                "Longitude": telemetry["longitude"],
                "Road_Segment": telemetry["road_segment"],
                "Corridor": telemetry["corridor"],
                "Bus_ID": telemetry["bus_id"],
                "Speed_kmh": telemetry["speed_kmh"]
            })

            # Save evidence frame if under limit
            if evidence_count < 20:
                cv2.rectangle(annotated, (15, 15), (780, 100), (15, 23, 42), -1)
                cv2.rectangle(annotated, (15, 15), (780, 100), defect["color"], 2)
                cv2.putText(annotated, f"INFRASTRUCTURE DEFECT: {defect['type']}", (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, defect["color"], 2)
                cv2.putText(annotated, f"GPS: {telemetry['latitude']}, {telemetry['longitude']} | {telemetry['road_segment']}", (30, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(annotated, f"Action: {defect['action']}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 220, 255), 1)

                evidence_file = evidence_dir / f"infra_defect_frame_{frame_idx:05d}.jpg"
                cv2.imwrite(str(evidence_file), annotated)
                evidence_count += 1

    cap.release()

    df = pd.DataFrame(results)
    if not df.empty:
        df.to_csv(output_csv, index=False)
    else:
        # Create schema-consistent empty df if no defects found
        pd.DataFrame(columns=[
            "Frame", "Timestamp", "Defect_Type", "Severity", "Recommended_Action",
            "Latitude", "Longitude", "Road_Segment", "Corridor", "Bus_ID", "Speed_kmh"
        ]).to_csv(output_csv, index=False)

    return {
        "success": True,
        "total_defects_logged": len(results),
        "evidence_saved": evidence_count,
        "csv_path": str(output_csv),
        "evidence_dir": str(evidence_dir)
    }

if __name__ == "__main__":
    UPLOAD_DIR = Path("uploads/road_videos")
    video_files = [f for f in UPLOAD_DIR.iterdir() if f.suffix.lower() in [".mp4", ".avi", ".mov"]]
    if video_files:
        res = analyze_infrastructure(video_files[0])
        print("Infrastructure analysis done:", res)
