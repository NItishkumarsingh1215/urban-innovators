import cv2
import os
import json
import pandas as pd
from ultralytics import YOLO

def process_edge_stream(video_path):
    # YOLOv8 model load karna edge inference ke liye
    model = YOLO('yolov8n.pt')
    
    # Output folders setup (Sirf lightweight evidence save karne ke liye)
    os.makedirs('evidence/edge_snapshots', exist_ok='true')
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    edge_incidents = []
    
    print("[INFO] Edge AI processing started: Discarding raw video, extracting metadata...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Har 30th frame par inference chalana (Edge optimization ke liye)
        if frame_count % 30 == 0:
            results = model(frame, verbose=False)
            detections = results[0].boxes
            
            # Agar koi hazard ya object detect hota hai
            if len(detections) > 0:
                # Evidence snapshot crop/save karo
                snapshot_name = f"evidence/edge_snapshots/frame_{frame_count:04d}.jpg"
                cv2.imwrite(snapshot_name, frame)
                
                # Lightweight Metadata payload banana (Bandwidth bachane ke liye)
                incident_meta = {
                    "frame_id": frame_count,
                    "timestamp": f"2026-09-08 12:00:{frame_count%60:02d}",
                    "latitude": 26.738600,  # Simulated/Extracted GPS
                    "longitude": 83.363600,
                    "total_detections": len(detections),
                    "evidence_image": snapshot_name,
                    "status": "FLAGGED_FOR_CLOUD"
                }
                edge_incidents.append(incident_meta)
                
    cap.release()
    
    # Sirf JSON metadata central server ko bhejne ke liye save karna
    with open('edge_metadata_payload.json', 'w') as f:
        json.dump(edge_incidents, f, indent=4)
        
    print(f"[SUCCESS] Edge processing complete. Total incidents filtered: {len(edge_incidents)}")
    print("[INFO] Raw video discarded. Only JSON metadata + evidence saved locally.")

if __name__ == "__main__":
    # Test ke liye apni video path de sakte hain
    process_edge_stream("uploads/sample_road.mp4")