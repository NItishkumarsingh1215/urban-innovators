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
        
        # Dual-channel color space: LAB for dark stagnant water pools + HSV for specular sheen
        lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l_ch = lab_roi[:, :, 0]
        mean_l = float(np.mean(l_ch))
        _, dark_puddle = cv2.threshold(l_ch, max(15, int(mean_l * 0.62)), 255, cv2.THRESH_BINARY_INV)

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_roi)
        _, glint_mask = cv2.threshold(v, 215, 255, cv2.THRESH_BINARY)
        _, low_sat_mask = cv2.threshold(s, 60, 255, cv2.THRESH_BINARY_INV)
        glint_water = cv2.bitwise_and(glint_mask, low_sat_mask)

        # Gradient filter to suppress white painted lane stripes
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray_roi, cv2.CV_16S, 1, 0)
        grad_x_abs = cv2.convertScaleAbs(grad_x)
        _, sharp_lane_borders = cv2.threshold(grad_x_abs, 45, 255, cv2.THRESH_BINARY)

        water_candidate = cv2.bitwise_or(dark_puddle, glint_water)
        water_candidate = cv2.bitwise_and(water_candidate, cv2.bitwise_not(sharp_lane_borders))
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        water_clean = cv2.morphologyEx(water_candidate, cv2.MORPH_OPEN, kernel)
        water_clean = cv2.morphologyEx(water_clean, cv2.MORPH_CLOSE, kernel)

        w_contours, _ = cv2.findContours(water_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_puddle_cnts = []
        filtered_water_mask = np.zeros_like(water_clean)

        for cnt in w_contours:
            c_area = cv2.contourArea(cnt)
            if c_area < 55:
                continue
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            aspect = cw / float(max(1, ch))
            if aspect > 4.5 or aspect < 0.22:
                continue
            puddle_patch = gray_roi[cy:cy+ch, cx:cx+cw]
            if puddle_patch.size > 20 and float(np.std(puddle_patch)) > 42.0:
                continue
            valid_puddle_cnts.append((cnt, c_area, cx, cy, cw, ch))
            cv2.drawContours(filtered_water_mask, [cnt], -1, 255, -1)

        roi_area = float(roi.shape[0] * roi.shape[1])
        water_pixels = cv2.countNonZero(filtered_water_mask)
        water_ratio = (water_pixels / roi_area) * 100.0

        score = round(water_ratio, 1)
        max_score = max(max_score, score)

        is_waterlogged = score >= 5.0 and len(valid_puddle_cnts) >= 1
        if score >= 25.0:
            risk = "CRITICAL"
        elif score >= 14.0:
            risk = "HIGH"
        elif is_waterlogged:
            risk = "MEDIUM"
        else:
            risk = "LOW"

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
            "Puddles_Count": len(valid_puddle_cnts),
            "Latitude": telemetry["latitude"],
            "Longitude": telemetry["longitude"],
            "Road_Segment": telemetry["road_segment"],
            "Corridor": telemetry["corridor"],
            "Bus_ID": telemetry["bus_id"],
            "Speed_kmh": telemetry["speed_kmh"],
            "Action_Required": "Municipal Suction Pump Dispatch & Stormwater Clearance" if is_waterlogged else "Routine Monitoring"
        }
        results.append(record)

        # Save annotated evidence frame with semi-transparent overlay
        if is_waterlogged and evidence_count < 30 and (frame_idx % (sample_step * 2) == 0):
            annotated = frame.copy()
            roi_overlay = annotated[roi_top:height, 0:width].copy()
            roi_overlay[filtered_water_mask > 0] = [235, 160, 40]
            cv2.addWeighted(roi_overlay, 0.45, annotated[roi_top:height, 0:width], 0.55, 0, annotated[roi_top:height, 0:width])

            for cnt, c_area, cx, cy, cw, ch in valid_puddle_cnts:
                cnt_shifted = cnt + np.array([0, roi_top])
                cv2.drawContours(annotated, [cnt_shifted], -1, (255, 140, 0), 2)
                cv2.putText(annotated, f"PUDDLE {int(c_area)}px", (cx, max(16, cy + roi_top - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 140, 0), 1)

            # Metadata Overlay
            w_clr = (0, 0, 255) if risk in ["CRITICAL", "HIGH"] else (59, 130, 246)
            cv2.rectangle(annotated, (15, 15), (780, 95), (15, 23, 42), -1)
            cv2.rectangle(annotated, (15, 15), (780, 95), w_clr, 2)
            cv2.putText(annotated, f"WATERLOGGING HAZARD | {risk} (Coverage: {score}%)", (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)
            cv2.putText(annotated, f"GPS: {telemetry['latitude']}, {telemetry['longitude']} | {telemetry['road_segment']} | {len(valid_puddle_cnts)} Puddles", (30, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(annotated, f"Bus: {telemetry['bus_id']} | Speed: {telemetry['speed_kmh']} km/h | Municipal Drain Action", (30, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 200, 220), 1)

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