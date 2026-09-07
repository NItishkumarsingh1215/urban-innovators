import os
import cv2
import pandas as pd
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# =========================================================
# PROJECT PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BASE_DIR / "uploads" / "road_videos"
DATA_DIR = BASE_DIR / "data"
EVIDENCE_DIR = BASE_DIR / "evidence" / "traffic_detections"
OUTPUT_DIR = BASE_DIR / "output"

for folder in [DATA_DIR, EVIDENCE_DIR, OUTPUT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov"]
video_files = [f for f in UPLOAD_DIR.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS]

if len(video_files) == 0:
    print("ERROR: No road video found in uploads/road_videos!")
    raise SystemExit

VIDEO_PATH = video_files[0]
print("Loading YOLOv8 vehicle detection & tracking model...")
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(str(VIDEO_PATH))

width, height, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
out = cv2.VideoWriter(str(OUTPUT_DIR / f"processed_{VIDEO_PATH.name}"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

VEHICLE_CLASSES = {2: "Car", 3: "Bike", 5: "Bus", 7: "Truck"}
results_data = []
unique_vehicle_ids = set()
total_cars = total_bikes = total_buses = total_trucks = 0
last_saved_gray = None
emergency_mode = False

def is_duplicate_frame(frame, previous_frame):
    if previous_frame is None: return False
    gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64))
    return cv2.absdiff(gray, previous_frame).mean() < 8

print("Processing video... (Press 'e' for Emergency Override, 'q' to stop).")

while True:
    success, frame = cap.read()
    if not success: break

    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    annotated_frame = frame.copy()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    results = model.track(frame, persist=True, verbose=False, conf=0.35, classes=[2, 3, 5, 7])
    current_frame_vehicles = 0 

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        
        for box, class_id, track_id in zip(boxes, class_ids, track_ids):
            current_frame_vehicles += 1
            vehicle_name = VEHICLE_CLASSES[class_id]

            if track_id not in unique_vehicle_ids:
                unique_vehicle_ids.add(track_id)
                if vehicle_name == "Car": total_cars += 1
                elif vehicle_name == "Bike": total_bikes += 1
                elif vehicle_name == "Bus": total_buses += 1
                elif vehicle_name == "Truck": total_trucks += 1

            x1, y1, x2, y2 = box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"{vehicle_name} ID:{track_id}", (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if current_frame_vehicles >= 8: traffic_level, level_color = "High", (0, 0, 255)
    elif current_frame_vehicles >= 4: traffic_level, level_color = "Medium", (0, 165, 255)
    else: traffic_level, level_color = "Low", (0, 255, 0)

    cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 100 if emergency_mode else 80), (0, 0, 0), -1)
    cv2.putText(annotated_frame, f"Time: {current_time}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(annotated_frame, f"Unique -> Cars: {total_cars} | Bikes: {total_bikes} | Buses: {total_buses} | Trucks: {total_trucks}", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(annotated_frame, f"Traffic: {traffic_level} ({current_frame_vehicles} in frame)", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, level_color, 2)
    if emergency_mode: cv2.putText(annotated_frame, "🚨 EMERGENCY OVERRIDE ACTIVE 🚨", (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 👉 YEH LINE WEB DASHBOARD KO LIVE RAKHEGI
    cv2.imwrite(str(DATA_DIR / "live_frame.jpg"), annotated_frame)

    if frame_number % 30 == 0 and current_frame_vehicles > 0:
        gray_small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64))
        if not is_duplicate_frame(frame, last_saved_gray):
            cv2.imwrite(str(EVIDENCE_DIR / f"traffic_evidence_{frame_number:04d}.jpg"), annotated_frame)
            last_saved_gray = gray_small

        results_data.append({
            "Frame": frame_number, "Timestamp": current_time, 
            "Live_Vehicles_in_Frame": current_frame_vehicles,
            "Traffic_Level": traffic_level, "Emergency": emergency_mode
        })
        pd.DataFrame(results_data).to_csv(DATA_DIR / "traffic_results.csv", index=False)

    out.write(annotated_frame)
    try: cv2.imshow("Live AI Camera", annotated_frame)
    except: pass

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('e'): emergency_mode = not emergency_mode

cap.release()
out.release()
cv2.destroyAllWindows()