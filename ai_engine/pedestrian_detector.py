import cv2
import pandas as pd
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from ai_engine.telemetry_engine import get_telemetry_for_frame

def analyze_pedestrians(video_path, output_csv="data/pedestrian_results.csv", evidence_dir="evidence/pedestrian_detections", corridor_key="bengaluru_bel"):
    """
    YOLOv8 pedestrian safety analyzer.
    Detects pedestrians, clusters, school children crossing roads, and vulnerable road users.
    Synchronizes Bus OBU GPS coordinates for every detected pedestrian event.
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
        return {"success": False, "message": f"Unable to open video: {video_path}"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    model = YOLO("yolov8n.pt")
    results = []
    frame_number = 0
    evidence_count = 0
    sample_step = 15

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1

        if frame_number % sample_step != 0:
            continue

        telemetry = get_telemetry_for_frame(frame_number, total_frames, fps, corridor_key)
        annotated_frame = frame.copy()

        # Track person class (0 in COCO)
        yolo_results = model.track(frame, persist=True, verbose=False, conf=0.35, classes=[0])
        
        pedestrian_count = 0
        school_children_count = 0
        is_school_zone = "School" in telemetry["road_segment"] or "Junction" in telemetry["road_segment"]

        if yolo_results[0].boxes is not None:
            boxes = yolo_results[0].boxes.xyxy.cpu().numpy().astype(int)
            for box in boxes:
                pedestrian_count += 1
                x1, y1, x2, y2 = box
                h = y2 - y1
                w = x2 - x1

                # School children heuristic: smaller bbox height / aspect ratio in or near school zones
                is_child = (h < 120 and is_school_zone) or (h < 90)
                if is_child:
                    school_children_count += 1
                    color = (0, 140, 255)  # Orange for children
                    label = "School Child"
                else:
                    color = (255, 0, 0)    # Blue for regular pedestrian
                    label = "Pedestrian"

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label, (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        is_vulnerable = pedestrian_count >= 2 or school_children_count > 0 or (pedestrian_count > 0 and is_school_zone)

        results.append({
            "Frame": frame_number,
            "Timestamp": telemetry["timestamp"],
            "Pedestrian_Count": pedestrian_count,
            "School_Children_Count": school_children_count,
            "Vulnerable_Situation": "CRITICAL - SCHOOL ZONE" if (school_children_count > 0 and is_school_zone) else ("YES" if is_vulnerable else "NO"),
            "Latitude": telemetry["latitude"],
            "Longitude": telemetry["longitude"],
            "Road_Segment": telemetry["road_segment"],
            "Corridor": telemetry["corridor"],
            "Bus_ID": telemetry["bus_id"],
            "Speed_kmh": telemetry["speed_kmh"]
        })

        # Save evidence if vulnerable pedestrian situation detected
        if is_vulnerable and evidence_count < 20 and (frame_number % (sample_step * 2) == 0):
            status_text = "SCHOOL CHILDREN CROSSING" if school_children_count > 0 else f"VULNERABLE PEDESTRIANS ({pedestrian_count})"
            box_color = (0, 0, 255) if school_children_count > 0 else (0, 165, 255)

            cv2.rectangle(annotated_frame, (15, 15), (780, 100), (15, 23, 42), -1)
            cv2.rectangle(annotated_frame, (15, 15), (780, 100), box_color, 2)
            cv2.putText(annotated_frame, f"SAFETY ALERT: {status_text}", (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
            cv2.putText(annotated_frame, f"GPS: {telemetry['latitude']}, {telemetry['longitude']} | {telemetry['road_segment']}", (30, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(annotated_frame, f"Bus: {telemetry['bus_id']} | Bus Speed: {telemetry['speed_kmh']} km/h", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 220, 255), 1)

            evidence_file = evidence_dir / f"pedestrian_hazard_frame_{frame_number:05d}.jpg"
            cv2.imwrite(str(evidence_file), annotated_frame)
            evidence_count += 1

    cap.release()

    df = pd.DataFrame(results)
    if not df.empty:
        df.to_csv(output_csv, index=False)

    return {
        "success": True,
        "records_logged": len(results),
        "evidence_saved": evidence_count,
        "csv_path": str(output_csv),
        "evidence_dir": str(evidence_dir)
    }

if __name__ == "__main__":
    UPLOAD_DIR = Path("uploads/road_videos")
    video_files = [f for f in UPLOAD_DIR.iterdir() if f.suffix.lower() in [".mp4", ".avi", ".mov"]]
    if video_files:
        res = analyze_pedestrians(video_files[0])
        print("Pedestrian analysis done:", res)