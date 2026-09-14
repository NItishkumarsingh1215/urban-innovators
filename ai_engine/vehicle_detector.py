import cv2
import pandas as pd
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from ai_engine.telemetry_engine import get_telemetry_for_frame

VEHICLE_CLASSES = {2: "Car", 3: "Bike", 5: "Bus", 7: "Truck"}

def analyze_traffic(video_path, output_csv="data/traffic_results.csv", evidence_dir="evidence/traffic_detections", corridor_key="bengaluru_bel"):
    """
    YOLOv8 headless vehicle tracking & traffic density estimation engine.
    Detects cars, bikes, buses, trucks, estimates vehicle density, bottleneck index,
    and synchronizes exact Bus OBU GPS coordinates for every detection frame.
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

    model = YOLO("yolov8n.pt")

    results_data = []
    unique_vehicle_ids = set()
    total_cars = 0
    total_bikes = 0
    total_buses = 0
    total_trucks = 0
    evidence_count = 0
    frame_number = 0
    sample_step = 15  # Process every 15th frame (~2 fps) for edge speed

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1

        if frame_number % sample_step != 0:
            continue

        telemetry = get_telemetry_for_frame(frame_number, total_frames, fps, corridor_key)
        annotated_frame = frame.copy()

        # Run YOLO tracking on vehicle classes: car(2), motorcycle(3), bus(5), truck(7)
        yolo_res = model.track(frame, persist=True, verbose=False, conf=0.35, classes=[2, 3, 5, 7])
        
        current_frame_vehicles = 0
        current_cars = 0
        current_bikes = 0
        current_buses = 0
        current_trucks = 0

        if yolo_res[0].boxes is not None and yolo_res[0].boxes.id is not None:
            boxes = yolo_res[0].boxes.xyxy.cpu().numpy().astype(int)
            class_ids = yolo_res[0].boxes.cls.cpu().numpy().astype(int)
            track_ids = yolo_res[0].boxes.id.cpu().numpy().astype(int)

            for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                current_frame_vehicles += 1
                vehicle_name = VEHICLE_CLASSES.get(class_id, "Vehicle")

                if track_id not in unique_vehicle_ids:
                    unique_vehicle_ids.add(track_id)
                    if vehicle_name == "Car": total_cars += 1
                    elif vehicle_name == "Bike": total_bikes += 1
                    elif vehicle_name == "Bus": total_buses += 1
                    elif vehicle_name == "Truck": total_trucks += 1

                if vehicle_name == "Car": current_cars += 1
                elif vehicle_name == "Bike": current_bikes += 1
                elif vehicle_name == "Bus": current_buses += 1
                elif vehicle_name == "Truck": current_trucks += 1

                x1, y1, x2, y2 = box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"{vehicle_name} #{track_id}", (x1, max(y1 - 8, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        # Traffic congestion & bottleneck categorization
        traffic_level = "High" if current_frame_vehicles >= 4 else ("Medium" if current_frame_vehicles >= 2 else "Low")
        is_bottleneck = current_frame_vehicles >= 4 or (current_frame_vehicles >= 3 and telemetry["speed_kmh"] < 36.0) or (current_frame_vehicles >= 2 and telemetry["speed_kmh"] < 32.0)
        level_color = (0, 0, 255) if is_bottleneck else ((0, 165, 255) if traffic_level == "High" else (0, 255, 0))

        results_data.append({
            "Frame": frame_number,
            "Timestamp": telemetry["timestamp"],
            "Vehicles_In_Frame": current_frame_vehicles,
            "Cars": current_cars,
            "Bikes": current_bikes,
            "Buses": current_buses,
            "Trucks": current_trucks,
            "Traffic_Level": traffic_level,
            "Bottleneck": "YES" if is_bottleneck else "NO",
            "Latitude": telemetry["latitude"],
            "Longitude": telemetry["longitude"],
            "Road_Segment": telemetry["road_segment"],
            "Corridor": telemetry["corridor"],
            "Bus_ID": telemetry["bus_id"],
            "Speed_kmh": telemetry["speed_kmh"]
        })

        # Save annotated evidence frame when traffic or bottleneck is detected
        if (current_frame_vehicles >= 2 or is_bottleneck) and evidence_count < 35 and (frame_number % (sample_step * 2) == 0):
            # Overlay info header
            cv2.rectangle(annotated_frame, (15, 15), (780, 100), (15, 23, 42), -1)
            cv2.rectangle(annotated_frame, (15, 15), (780, 100), level_color, 2)
            banner_txt = f"🚨 BOTTLENECK CHOKE POINT ({current_frame_vehicles} Vehicles)" if is_bottleneck else f"TRAFFIC DENSITY: {traffic_level.upper()} ({current_frame_vehicles} Vehicles)"
            cv2.putText(annotated_frame, banner_txt,
                        (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, level_color, 2)
            cv2.putText(annotated_frame, f"GPS: {telemetry['latitude']}, {telemetry['longitude']} | {telemetry['road_segment']}",
                        (30, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(annotated_frame, f"Speed: {telemetry['speed_kmh']} km/h | Total Seen -> Cars: {total_cars} | Bikes: {total_bikes} | Buses: {total_buses} | Trucks: {total_trucks}",
                        (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 220, 255), 1)

            evidence_file = evidence_dir / f"traffic_frame_{frame_number:05d}.jpg"
            cv2.imwrite(str(evidence_file), annotated_frame)
            evidence_count += 1

    cap.release()

    df = pd.DataFrame(results_data)
    if not df.empty:
        df.to_csv(output_csv, index=False)

    return {
        "success": True,
        "total_unique_vehicles": len(unique_vehicle_ids),
        "total_cars": total_cars,
        "total_bikes": total_bikes,
        "total_buses": total_buses,
        "total_trucks": total_trucks,
        "records_logged": len(results_data),
        "evidence_saved": evidence_count,
        "csv_path": str(output_csv),
        "evidence_dir": str(evidence_dir)
    }

if __name__ == "__main__":
    UPLOAD_DIR = Path("uploads/road_videos")
    video_files = [f for f in UPLOAD_DIR.iterdir() if f.suffix.lower() in [".mp4", ".avi", ".mov"]]
    if video_files:
        res = analyze_traffic(video_files[0])
        print("Traffic analysis done:", res)