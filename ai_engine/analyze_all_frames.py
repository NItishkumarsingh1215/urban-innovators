from ultralytics import YOLO
import os
import cv2
import pandas as pd


# ==============================
# PROJECT PATHS
# ==============================

MODEL_PATH = "models/pothole_model.pt"

FRAMES_DIR = "data/extracted_frames"

EVIDENCE_DIR = "evidence/all_detections"

RESULTS_FILE = "data/pothole_detection_results.csv"


# ==============================
# MAIN FUNCTION
# ==============================

def main():

    # Check model
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found -> {MODEL_PATH}")
        return

    # Check frames folder
    if not os.path.exists(FRAMES_DIR):
        print(f"ERROR: Frames folder not found -> {FRAMES_DIR}")
        return

    # Create output folders
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Get all image frames
    frame_files = sorted([
        file
        for file in os.listdir(FRAMES_DIR)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if len(frame_files) == 0:
        print("No frames found.")
        return

    print("=" * 50)
    print("URBAN INTELLIGENCE - FULL FRAME ANALYSIS")
    print("=" * 50)

    print(f"Total frames found: {len(frame_files)}")
    print("Loading pothole AI model...")

    # Load AI model
    model = YOLO(MODEL_PATH)

    print("Model loaded successfully.")
    print("Starting analysis...\n")

    detection_data = []

    total_detections = 0

    # ==============================
    # ANALYZE EACH FRAME
    # ==============================

    for index, frame_file in enumerate(frame_files, start=1):

        frame_path = os.path.join(
            FRAMES_DIR,
            frame_file
        )

        print(
            f"[{index}/{len(frame_files)}] "
            f"Analyzing: {frame_file}"
        )

        results = model(
            frame_path,
            conf=0.40,
            verbose=False
        )

        result = results[0]

        # No detection
        if result.boxes is None or len(result.boxes) == 0:

            detection_data.append({
                "frame": frame_file,
                "detected": False,
                "class_name": None,
                "confidence": None,
                "evidence_image": None
            })

            print("   → No pothole detected")

            continue

        # Detection found
        best_confidence = 0
        best_class_name = None

        for box in result.boxes:

            class_id = int(
                box.cls[0].item()
            )

            confidence = float(
                box.conf[0].item()
            )

            class_name = result.names[
                class_id
            ]

            if confidence > best_confidence:

                best_confidence = confidence
                best_class_name = class_name

        # Save annotated evidence image
        annotated_image = result.plot()

        evidence_filename = (
            f"detected_{frame_file}"
        )

        evidence_path = os.path.join(
            EVIDENCE_DIR,
            evidence_filename
        )

        cv2.imwrite(
            evidence_path,
            annotated_image
        )

        # Save detection data
        detection_data.append({
            "frame": frame_file,
            "detected": True,
            "class_name": best_class_name,
            "confidence": round(
                best_confidence * 100,
                2
            ),
            "evidence_image": evidence_path
        })

        total_detections += 1

        print(
            f"   → DETECTED: "
            f"{best_class_name} | "
            f"{best_confidence * 100:.2f}%"
        )

    # ==============================
    # SAVE CSV RESULTS
    # ==============================

    df = pd.DataFrame(
        detection_data
    )

    df.to_csv(
        RESULTS_FILE,
        index=False
    )

    # ==============================
    # FINAL SUMMARY
    # ==============================

    print("\n" + "=" * 50)

    print("ANALYSIS COMPLETED")

    print("=" * 50)

    print(f"Total frames analyzed: {len(frame_files)}")

    print(
        f"Frames with detection: "
        f"{total_detections}"
    )

    print(
        f"Evidence images folder: "
        f"{EVIDENCE_DIR}"
    )

    print(
        f"CSV results saved: "
        f"{RESULTS_FILE}"
    )

    print("=" * 50)


if __name__ == "__main__":
    main()