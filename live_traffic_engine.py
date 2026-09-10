import os
import time
import cv2
import pandas as pd
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LIVE_FRAME_PATH = DATA_DIR / "live_frame.jpg"
LIVE_STOP_FILE = DATA_DIR / "stop_live_traffic.flag"
CSV_PATH = DATA_DIR / "traffic_results.csv"
MODEL_PATH = BASE_DIR / "yolov8n.pt"

DATA_DIR.mkdir(parents=True, exist_ok=True)

def atomic_write_image(path, image):
    tmp = path.with_name(path.stem + "_tmp" + path.suffix)
    if cv2.imwrite(str(tmp), image):
        try:
            tmp.replace(path)
        except Exception:
            try:
                os.replace(str(tmp), str(path))
            except Exception:
                pass

def main():
    if not MODEL_PATH.exists():
        print("ERROR: yolov8n.pt not found.")
        return

    # Remove stale stop flag.
    try:
        LIVE_STOP_FILE.unlink(missing_ok=True)
    except Exception:
        pass

    model = YOLO(str(MODEL_PATH))
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == "nt" else 0)

    if not cap.isOpened():
        print("ERROR: Local webcam could not be opened.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    class_names = {2: "Car", 3: "Bike", 5: "Bus", 7: "Truck"}
    unique_ids = set()
    rows = []

    frame_no = 0

    try:
        while not LIVE_STOP_FILE.exists():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            frame_no += 1

            try:
                result = model.track(
                    source=frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=0.35,
                    iou=0.5,
                    classes=[2, 3, 5, 7],
                    verbose=False
                )[0]
            except Exception:
                result = model.predict(
                    source=frame,
                    conf=0.35,
                    classes=[2, 3, 5, 7],
                    verbose=False
                )[0]

            annotated = result.plot()

            counts = {"Car": 0, "Bike": 0, "Bus": 0, "Truck": 0}
            live_ids = set()

            if result.boxes is not None and len(result.boxes):
                for i in range(len(result.boxes)):
                    try:
                        cls_id = int(result.boxes.cls[i].item())
                        label = class_names.get(cls_id)
                        if label:
                            counts[label] += 1

                        if result.boxes.id is not None:
                            track_id = int(result.boxes.id[i].item())
                            live_ids.add(track_id)
                            unique_ids.add(track_id)
                    except Exception:
                        continue

            total_live = sum(counts.values())
            if total_live >= 12:
                traffic_level = "High"
            elif total_live >= 6:
                traffic_level = "Medium"
            else:
                traffic_level = "Low"

            # Emergency override is kept conservative; normal cars/bikes are
            # never labelled emergency just because they are present.
            emergency = False

            cv2.putText(
                annotated,
                f"LIVE | Vehicles: {total_live} | Traffic: {traffic_level}",
                (15, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2
            )
            cv2.putText(
                annotated,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                (15, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            atomic_write_image(LIVE_FRAME_PATH, annotated)

            rows.append({
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Frame": frame_no,
                "Cars": counts["Car"],
                "Bikes": counts["Bike"],
                "Buses": counts["Bus"],
                "Trucks": counts["Truck"],
                "Live_Vehicles_in_Frame": total_live,
                "Traffic_Level": traffic_level,
                "Emergency": emergency,
                "Total_Unique_Cars": counts["Car"],
                "Total_Unique_Bikes": counts["Bike"],
                "Total_Unique_Buses": counts["Bus"],
                "Total_Unique_Trucks": counts["Truck"],
                "Unique_Tracked_Vehicles": len(unique_ids)
            })

            # Keep only recent rows so the CSV remains lightweight.
            if len(rows) > 200:
                rows = rows[-200:]

            try:
                pd.DataFrame(rows).to_csv(CSV_PATH, index=False)
            except Exception:
                pass

            time.sleep(0.03)

    finally:
        cap.release()
        try:
            LIVE_FRAME_PATH.unlink(missing_ok=True)
        except Exception:
            pass

if __name__ == "__main__":
    main()
