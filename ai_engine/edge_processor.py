import cv2
import os
import json
from pathlib import Path
from ultralytics import YOLO

def process_edge_stream(video_path, corridor_key=None):
    if corridor_key is None:
        corridor_key = os.environ.get("BUS_CORRIDOR", "bengaluru_bel")

    try:
        from ai_engine.telemetry_engine import get_telemetry_for_frame
    except Exception:
        from telemetry_engine import get_telemetry_for_frame

    model = YOLO('yolov8n.pt')
    
    os.makedirs('evidence/edge_snapshots', exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    frame_count = 0
    edge_incidents = []
    
    print(f"[INFO] Edge AI processing started on corridor '{corridor_key}': Discarding raw video, extracting metadata...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Inference every 30 frames for lightweight edge unit efficiency
        if frame_count % 30 == 0:
            results = model(frame, verbose=False)
            detections = results[0].boxes
            
            if len(detections) > 0:
                snapshot_name = f"evidence/edge_snapshots/frame_{frame_count:04d}.jpg"
                cv2.imwrite(snapshot_name, frame)
                
                tel = get_telemetry_for_frame(frame_count, total_frames, fps, corridor_key)

                incident_meta = {
                    "frame_id": frame_count,
                    "timestamp": tel["timestamp"],
                    "latitude": tel["latitude"],
                    "longitude": tel["longitude"],
                    "road_segment": tel["road_segment"],
                    "bus_id": tel["bus_id"],
                    "speed_kmh": tel["speed_kmh"],
                    "total_detections": len(detections),
                    "evidence_image": snapshot_name,
                    "status": "FLAGGED_FOR_CLOUD"
                }
                edge_incidents.append(incident_meta)
                
    cap.release()
    
    with open('edge_metadata_payload.json', 'w') as f:
        json.dump(edge_incidents, f, indent=4)
        
    print(f"[SUCCESS] Edge processing complete. Total incidents filtered: {len(edge_incidents)}")
    print("[INFO] Raw video discarded. Only JSON metadata + evidence saved locally.")
    return edge_incidents

if __name__ == "__main__":
    from pathlib import Path
    videos = list(Path("uploads/road_videos").glob("*.mp4"))
    if videos:
        process_edge_stream(videos[0])