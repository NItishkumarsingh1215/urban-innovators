import cv2
import pandas as pd
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

def analyze_pedestrians(video_path, output_csv="data/pedestrian_results.csv", evidence_dir="evidence/pedestrian_detections"):
    video_path = Path(video_path)
    output_csv = Path(output_csv)
    evidence_dir = Path(evidence_dir)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        return {"success": False, "message": "Video file not found."}

    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return {"success": False, "message": "Unable to open video."}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    results = []
    frame_number = 0
    evidence_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1

        # Process every 15th frame for efficiency
        if frame_number % 15 != 0:
            continue

        # YOLOv8 tracking for class 0 (person)
        yolo_results = model.track(frame, persist=True, verbose=False, conf=0.4, classes=[0])
        
        pedestrian_count = 0
        annotated_frame = frame.copy()

        if yolo_results[0].boxes is not None and yolo_results[0].boxes.id is not None:
            boxes = yolo_results[0].boxes.xyxy.cpu().numpy().astype(int)
            for box in boxes:
                pedestrian_count += 1
                x1, y1, x2, y2 = box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(annotated_frame, "Pedestrian", (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        timestamp = round(frame_number / fps, 2)
        is_hazard = pedestrian_count >= 3  # High footfall / potential crossing hazard if 3+ persons detected

        results.append({
            "Frame": frame_number,
            "Timestamp": timestamp,
            "Pedestrian_Count": pedestrian_count,
            "Vulnerable_Situation": "YES" if is_hazard else "NO"
        })

        # Save evidence if hazard or high pedestrian activity
        if is_hazard and evidence_count < 20:
            evidence_file = evidence_dir / f"pedestrian_hazard_{frame_number}.jpg"
            cv2.putText(annotated_frame, f"⚠️ VULNERABLE PEDESTRIAN ZONE ({pedestrian_count})", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imwrite(str(evidence_file), annotated_frame)
            evidence_count += 1

    cap.release()

    df = pd.DataFrame(results)
    if not df.empty:
        df["Analysis_Time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df.to_csv(output_csv, index=False)

    return {"success": True, "message": "Pedestrian analysis completed.", "csv_path": str(output_csv)}

if __name__ == "__main__":
    UPLOAD_DIR = Path("uploads/road_videos")
    video_files = [f for f in UPLOAD_DIR.iterdir() if f.suffix.lower() in [".mp4", ".avi", ".mov"]]
    if video_files:
        analyze_pedestrians(video_files[0])
        print("Pedestrian detection test completed.")