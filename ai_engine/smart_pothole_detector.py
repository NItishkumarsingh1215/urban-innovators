from ultralytics import YOLO
import os
import cv2
import pandas as pd


# ==========================================
# PROJECT PATHS
# ==========================================

MODEL_PATH = "models/pothole_model.pt"
FRAMES_DIR = "data/extracted_frames"

EVIDENCE_DIR = "evidence/smart_detections"

RESULTS_FILE = "data/smart_detection_results.csv"


# ==========================================
# DETECTION SETTINGS
# ==========================================

CONFIDENCE_THRESHOLD = 0.50

# Detection image ke bilkul neeche hone par
# false detection hone ki possibility zyada hai.
BOTTOM_REJECTION_ZONE = 0.85


# ==========================================
# CHECK VALID ROAD DETECTION
# ==========================================

def is_valid_pothole(box, image_height):

    x1, y1, x2, y2 = box.xyxy[0].tolist()

    # Bounding box ka center
    center_y = (y1 + y2) / 2

    # Normalize position
    normalized_y = center_y / image_height

    # Image ke bahut neeche wale part me
    # detection ko reject karenge.
    if normalized_y > BOTTOM_REJECTION_ZONE:
        return False

    return True


# ==========================================
# MAIN FUNCTION
# ==========================================

def main():

    # Check model
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found -> {MODEL_PATH}")
        return

    # Check frames folder
    if not os.path.exists(FRAMES_DIR):
        print(f"ERROR: Frames folder not found -> {FRAMES_DIR}")
        return

    # Create folders
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Get all frames
    frame_files = sorted(
        [
            file
            for file in os.listdir(FRAMES_DIR)
            if file.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ]
    )

    if len(frame_files) == 0:
        print("No frames found.")
        return

    print("=" * 55)
    print("URBAN INTELLIGENCE - SMART POTHOLE DETECTION")
    print("=" * 55)

    print(f"Total frames: {len(frame_files)}")

    # Load model
    print("Loading AI model...")

    model = YOLO(MODEL_PATH)

    print("AI model loaded successfully.")
    print("Starting smart analysis...\n")

    detection_data = []

    valid_detection_count = 0
    rejected_detection_count = 0


    # ==========================================
    # ANALYZE ALL FRAMES
    # ==========================================

    for index, frame_file in enumerate(
        frame_files,
        start=1
    ):

        frame_path = os.path.join(
            FRAMES_DIR,
            frame_file
        )

        print(
            f"[{index}/{len(frame_files)}] "
            f"Analyzing {frame_file}"
        )

        # Read image
        image = cv2.imread(
            frame_path
        )

        if image is None:

            print("   → Could not read image")

            continue

        image_height, image_width = image.shape[:2]

        # Run YOLO
        results = model(
            frame_path,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )

        result = results[0]

        valid_boxes = []

        # Check detections
        if (
            result.boxes is not None
            and len(result.boxes) > 0
        ):

            for box in result.boxes:

                confidence = float(
                    box.conf[0].item()
                )

                class_id = int(
                    box.cls[0].item()
                )

                class_name = result.names[
                    class_id
                ]

                # Location filtering
                if not is_valid_pothole(
                    box,
                    image_height
                ):

                    rejected_detection_count += 1

                    print(
                        f"   → REJECTED: "
                        f"{class_name} "
                        f"{confidence * 100:.2f}% "
                        f"(bottom false-detection zone)"
                    )

                    continue

                # Valid detection
                valid_boxes.append(
                    {
                        "box": box,
                        "confidence": confidence,
                        "class_name": class_name
                    }
                )


        # ==========================================
        # IF VALID DETECTION FOUND
        # ==========================================

        if len(valid_boxes) > 0:

            # Find best detection
            best_detection = max(
                valid_boxes,
                key=lambda item: item[
                    "confidence"
                ]
            )

            best_confidence = best_detection[
                "confidence"
            ]

            best_class_name = best_detection[
                "class_name"
            ]

            # Create image copy
            annotated_image = image.copy()

            # Draw ONLY valid boxes
            for detection in valid_boxes:

                box = detection["box"]

                x1, y1, x2, y2 = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                confidence = detection[
                    "confidence"
                ]

                class_name = detection[
                    "class_name"
                ]

                label = (
                    f"{class_name} "
                    f"{confidence * 100:.1f}%"
                )

                cv2.rectangle(
                    annotated_image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    annotated_image,
                    label,
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            # Save evidence
            evidence_filename = (
                f"smart_{frame_file}"
            )

            evidence_path = os.path.join(
                EVIDENCE_DIR,
                evidence_filename
            )

            cv2.imwrite(
                evidence_path,
                annotated_image
            )

            valid_detection_count += 1

            detection_data.append(
                {
                    "frame": frame_file,
                    "status": "VALID_POTHOLE",
                    "class_name": best_class_name,
                    "confidence": round(
                        best_confidence * 100,
                        2
                    ),
                    "evidence_image": evidence_path
                }
            )

            print(
                f"   → VALID: "
                f"{best_class_name} | "
                f"{best_confidence * 100:.2f}%"
            )


        # ==========================================
        # NO VALID DETECTION
        # ==========================================

        else:

            detection_data.append(
                {
                    "frame": frame_file,
                    "status": "NO_VALID_POTHOLE",
                    "class_name": None,
                    "confidence": None,
                    "evidence_image": None
                }
            )

            print(
                "   → No valid pothole detected"
            )


    # ==========================================
    # SAVE RESULTS
    # ==========================================

    df = pd.DataFrame(
        detection_data
    )

    df.to_csv(
        RESULTS_FILE,
        index=False
    )


    # ==========================================
    # FINAL SUMMARY
    # ==========================================

    print("\n" + "=" * 55)

    print("SMART ANALYSIS COMPLETED")

    print("=" * 55)

    print(
        f"Total frames analyzed: "
        f"{len(frame_files)}"
    )

    print(
        f"Valid pothole detections: "
        f"{valid_detection_count}"
    )

    print(
        f"Rejected detections: "
        f"{rejected_detection_count}"
    )

    print(
        f"Evidence folder: "
        f"{EVIDENCE_DIR}"
    )

    print(
        f"CSV report: "
        f"{RESULTS_FILE}"
    )

    print("=" * 55)


if __name__ == "__main__":
    main()