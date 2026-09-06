import os
import cv2
import pandas as pd
from pathlib import Path
from ultralytics import YOLO


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BASE_DIR / "uploads" / "road_videos"
DATA_DIR = BASE_DIR / "data"

EVIDENCE_DIR = (
    BASE_DIR
    / "evidence"
    / "traffic_detections"
)

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

EVIDENCE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# FIND VIDEO
# =========================================================

VIDEO_EXTENSIONS = [
    ".mp4",
    ".avi",
    ".mov"
]


video_files = [
    file
    for file in UPLOAD_DIR.iterdir()
    if file.suffix.lower()
    in VIDEO_EXTENSIONS
]


if len(video_files) == 0:

    print(
        "ERROR: No road video found!"
    )

    raise SystemExit


VIDEO_PATH = video_files[0]


# =========================================================
# LOAD YOLO MODEL
# =========================================================

print(
    "Loading YOLOv8 vehicle detection model..."
)


model = YOLO(
    "yolov8n.pt"
)


# =========================================================
# OPEN VIDEO
# =========================================================

cap = cv2.VideoCapture(
    str(VIDEO_PATH)
)


if not cap.isOpened():

    print(
        "ERROR: Unable to open video!"
    )

    raise SystemExit


total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)


fps = cap.get(
    cv2.CAP_PROP_FPS
)


print(
    f"Video: {VIDEO_PATH.name}"
)


print(
    f"Total Frames: {total_frames}"
)


print(
    f"FPS: {fps}"
)


# =========================================================
# VEHICLE CLASSES
# =========================================================

VEHICLE_CLASSES = {
    2: "Car",
    3: "Bike",
    5: "Bus",
    7: "Truck"
}


# =========================================================
# RESULTS
# =========================================================

results_data = []


# =========================================================
# DUPLICATE PREVENTION
# =========================================================

last_saved_gray = None


def is_duplicate_frame(
    frame,
    previous_frame
):

    if previous_frame is None:

        return False


    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    gray = cv2.resize(
        gray,
        (64, 64)
    )


    difference = cv2.absdiff(
        gray,
        previous_frame
    )


    score = difference.mean()


    # Lower difference means
    # scene is almost same
    if score < 8:

        return True


    return False


# =========================================================
# PROCESS VIDEO
# =========================================================

frame_number = 0


while True:

    success, frame = cap.read()


    if not success:

        break


    frame_number += 1


    # Process every 30th frame
    if frame_number % 30 != 0:

        continue


    # =====================================================
    # YOLO DETECTION
    # =====================================================

    detections = model(
        frame,
        verbose=False,
        conf=0.35
    )


    car_count = 0

    bike_count = 0

    bus_count = 0

    truck_count = 0

    total_vehicles = 0


    annotated_frame = frame.copy()


    for result in detections:

        boxes = result.boxes


        if boxes is None:

            continue


        for box in boxes:

            class_id = int(
                box.cls[0]
            )


            confidence = float(
                box.conf[0]
            )


            if class_id not in VEHICLE_CLASSES:

                continue


            vehicle_name = (
                VEHICLE_CLASSES[class_id]
            )


            total_vehicles += 1


            if vehicle_name == "Car":

                car_count += 1


            elif vehicle_name == "Bike":

                bike_count += 1


            elif vehicle_name == "Bus":

                bus_count += 1


            elif vehicle_name == "Truck":

                truck_count += 1


            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            label = (
                f"{vehicle_name} "
                f"{confidence:.2f}"
            )


            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            cv2.putText(
                annotated_frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


    # =====================================================
    # TRAFFIC LEVEL
    # =====================================================

    if total_vehicles >= 8:

        traffic_level = "High"


    elif total_vehicles >= 4:

        traffic_level = "Medium"


    else:

        traffic_level = "Low"


    # =====================================================
    # ADD INFORMATION ON IMAGE
    # =====================================================

    info_text = (
        f"Vehicles: {total_vehicles} | "
        f"Cars: {car_count} | "
        f"Bikes: {bike_count} | "
        f"Buses: {bus_count} | "
        f"Trucks: {truck_count}"
    )


    traffic_text = (
        f"Traffic Level: {traffic_level}"
    )


    cv2.rectangle(
        annotated_frame,
        (0, 0),
        (
            annotated_frame.shape[1],
            75
        ),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        annotated_frame,
        info_text,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        annotated_frame,
        traffic_text,
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )


    # =====================================================
    # SAVE UNIQUE TRAFFIC EVIDENCE
    # =====================================================

    if total_vehicles > 0:

        gray_small = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        gray_small = cv2.resize(
            gray_small,
            (64, 64)
        )


        duplicate = is_duplicate_frame(
            frame,
            last_saved_gray
        )


        if not duplicate:

            evidence_path = (
                EVIDENCE_DIR
                / (
                    f"traffic_frame_"
                    f"{frame_number:04d}.jpg"
                )
            )


            cv2.imwrite(
                str(evidence_path),
                annotated_frame
            )


            last_saved_gray = (
                gray_small
            )


    # =====================================================
    # SAVE CSV DATA
    # =====================================================

    results_data.append(
        {
            "Frame": frame_number,
            "Cars": car_count,
            "Bikes": bike_count,
            "Buses": bus_count,
            "Trucks": truck_count,
            "Total_Vehicles": total_vehicles,
            "Traffic_Level": traffic_level
        }
    )


    print(
        f"Frame {frame_number} | "
        f"Vehicles: {total_vehicles} | "
        f"Cars: {car_count} | "
        f"Bikes: {bike_count} | "
        f"Buses: {bus_count} | "
        f"Trucks: {truck_count} | "
        f"Traffic: {traffic_level}"
    )


# =========================================================
# RELEASE VIDEO
# =========================================================

cap.release()


# =========================================================
# SAVE CSV
# =========================================================

df = pd.DataFrame(
    results_data
)


csv_path = (
    DATA_DIR
    / "traffic_results.csv"
)


df.to_csv(
    csv_path,
    index=False
)


print()


print(
    "Traffic analysis completed!"
)


print(
    f"CSV saved: {csv_path}"
)


print(
    f"Traffic evidence folder: {EVIDENCE_DIR}"
)


print(
    f"Total unique traffic evidence: "
    f"{len(list(EVIDENCE_DIR.glob('*.jpg')))}"
)