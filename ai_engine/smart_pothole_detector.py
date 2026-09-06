from ultralytics import YOLO
import os
import cv2
import pandas as pd
import hashlib


# =========================================================
# PROJECT PATHS
# =========================================================

MODEL_PATH = "models/pothole_model.pt"

FRAMES_DIR = "data/extracted_frames"

EVIDENCE_DIR = "evidence/smart_detections"

RESULTS_FILE = (
    "data/smart_detection_results.csv"
)


# =========================================================
# DETECTION SETTINGS
# =========================================================

# 0.50 bahut strict tha.
# Lower confidence se zyada possible potholes detect honge.
CONFIDENCE_THRESHOLD = 0.30


# =========================================================
# SIZE FILTER
# =========================================================

# Bahut chhote random detections ko ignore karenge.
MIN_BOX_WIDTH = 20
MIN_BOX_HEIGHT = 12

MIN_BOX_AREA = 300


# =========================================================
# DUPLICATE DETECTION SETTINGS
# =========================================================

FRAME_HASH_SIZE = (64, 64)

DUPLICATE_DISTANCE_THRESHOLD = 5


# =========================================================
# CREATE IMAGE HASH
# =========================================================

def get_image_hash(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    small_image = cv2.resize(
        gray,
        FRAME_HASH_SIZE
    )

    return hashlib.md5(
        small_image.tobytes()
    ).hexdigest()


# =========================================================
# CHECK DUPLICATE IMAGE
# =========================================================

def is_duplicate_image(
    image,
    previous_hash
):

    current_hash = get_image_hash(
        image
    )

    if previous_hash is None:

        return False, current_hash

    if current_hash == previous_hash:

        return True, current_hash

    return False, current_hash


# =========================================================
# CHECK VALID POTHOLE
# =========================================================

def is_valid_pothole(
    box,
    image_width,
    image_height
):

    x1, y1, x2, y2 = (
        box.xyxy[0]
        .cpu()
        .numpy()
    )

    box_width = x2 - x1

    box_height = y2 - y1

    box_area = (
        box_width
        *
        box_height
    )


    # =====================================================
    # REMOVE VERY SMALL NOISE
    # =====================================================

    if box_width < MIN_BOX_WIDTH:

        return False


    if box_height < MIN_BOX_HEIGHT:

        return False


    if box_area < MIN_BOX_AREA:

        return False


    # =====================================================
    # IMPORTANT:
    #
    # Pehle bottom rejection zone tha:
    #
    # normalized_y > 0.85
    #
    # Usse lower road ke potholes reject ho rahe the.
    #
    # Ab position ke basis par pothole reject nahi hoga.
    # =====================================================

    return True


# =========================================================
# MAIN FUNCTION
# =========================================================

def main():

    # =====================================================
    # CHECK MODEL
    # =====================================================

    if not os.path.exists(
        MODEL_PATH
    ):

        print(
            f"ERROR: Model not found -> "
            f"{MODEL_PATH}"
        )

        return


    # =====================================================
    # CHECK FRAMES
    # =====================================================

    if not os.path.exists(
        FRAMES_DIR
    ):

        print(
            f"ERROR: Frames folder not found -> "
            f"{FRAMES_DIR}"
        )

        return


    # =====================================================
    # CREATE FOLDERS
    # =====================================================

    os.makedirs(
        EVIDENCE_DIR,
        exist_ok=True
    )

    os.makedirs(
        "data",
        exist_ok=True
    )


    # =====================================================
    # GET FRAME FILES
    # =====================================================

    frame_files = sorted(
        [

            file

            for file in os.listdir(
                FRAMES_DIR
            )

            if file.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png"
                )
            )

        ]
    )


    if len(frame_files) == 0:

        print(
            "No frames found."
        )

        return


    # =====================================================
    # HEADER
    # =====================================================

    print(
        "=" * 60
    )

    print(
        "URBAN INTELLIGENCE PLATFORM"
    )

    print(
        "IMPROVED SMART POTHOLE DETECTION"
    )

    print(
        "=" * 60
    )

    print(
        f"Total frames: "
        f"{len(frame_files)}"
    )

    print(
        f"Confidence threshold: "
        f"{CONFIDENCE_THRESHOLD}"
    )


    # =====================================================
    # LOAD MODEL
    # =====================================================

    print(
        "\nLoading AI pothole model..."
    )

    model = YOLO(
        MODEL_PATH
    )

    print(
        "AI model loaded successfully."
    )

    print(
        "\nStarting analysis...\n"
    )


    # =====================================================
    # RESULT VARIABLES
    # =====================================================

    detection_data = []

    frames_with_potholes = 0

    total_potholes_detected = 0

    rejected_detection_count = 0

    saved_evidence_count = 0

    duplicate_frame_count = 0

    previous_evidence_hash = None


    # =====================================================
    # ANALYZE ALL FRAMES
    # =====================================================

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


        # =================================================
        # READ IMAGE
        # =================================================

        image = cv2.imread(
            frame_path
        )


        if image is None:

            print(
                "   → Could not read image"
            )

            continue


        image_height, image_width = (
            image.shape[:2]
        )


        # =================================================
        # RUN YOLO
        # =================================================

        results = model(

            image,

            conf=CONFIDENCE_THRESHOLD,

            verbose=False

        )


        result = results[0]


        valid_detections = []


        # =================================================
        # CHECK DETECTIONS
        # =================================================

        if (

            result.boxes is not None

            and

            len(result.boxes) > 0

        ):


            for box in result.boxes:


                confidence = float(
                    box.conf[0].item()
                )


                class_id = int(
                    box.cls[0].item()
                )


                class_name = (
                    result.names[
                        class_id
                    ]
                )


                # =========================================
                # VALIDITY CHECK
                # =========================================

                if not is_valid_pothole(

                    box,

                    image_width,

                    image_height

                ):

                    rejected_detection_count += 1

                    print(
                        f"   → FILTERED: "
                        f"{class_name} | "
                        f"{confidence * 100:.1f}%"
                    )

                    continue


                # =========================================
                # VALID DETECTION
                # =========================================

                valid_detections.append(

                    {

                        "box": box,

                        "confidence": confidence,

                        "class_name": class_name

                    }

                )


        # =================================================
        # VALID DETECTION FOUND
        # =================================================

        if len(valid_detections) > 0:


            frames_with_potholes += 1


            total_potholes_detected += (

                len(
                    valid_detections
                )

            )


            # =============================================
            # CREATE ANNOTATED IMAGE
            # =============================================

            annotated_image = (
                image.copy()
            )


            best_confidence = 0

            best_class_name = "Pothole"


            # =============================================
            # DRAW ALL VALID BOXES
            # =============================================

            for detection in (
                valid_detections
            ):


                box = (
                    detection["box"]
                )


                x1, y1, x2, y2 = (

                    box.xyxy[0]

                    .cpu()

                    .numpy()

                    .astype(int)

                )


                confidence = (
                    detection[
                        "confidence"
                    ]
                )


                class_name = (
                    detection[
                        "class_name"
                    ]
                )


                if confidence > best_confidence:

                    best_confidence = (
                        confidence
                    )

                    best_class_name = (
                        class_name
                    )


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

                    (
                        x1,
                        max(
                            y1 - 10,
                            25
                        )
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.65,

                    (0, 255, 0),

                    2

                )


            # =============================================
            # ADD HEADER INFORMATION
            # =============================================

            cv2.rectangle(

                annotated_image,

                (0, 0),

                (
                    annotated_image.shape[1],
                    45
                ),

                (0, 0, 0),

                -1

            )


            info_text = (

                f"Potholes: "

                f"{len(valid_detections)}"

                f" | "

                f"Best Confidence: "

                f"{best_confidence * 100:.1f}%"

            )


            cv2.putText(

                annotated_image,

                info_text,

                (15, 30),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.65,

                (255, 255, 255),

                2

            )


            # =============================================
            # CHECK DUPLICATE EVIDENCE
            # =============================================

            is_duplicate, current_hash = (

                is_duplicate_image(

                    image,

                    previous_evidence_hash

                )

            )


            evidence_path = None


            if is_duplicate:


                duplicate_frame_count += 1


                print(

                    f"   → VALID but duplicate "

                    f"evidence skipped"

                )


            else:


                evidence_filename = (

                    f"pothole_"

                    f"{index:04d}_"

                    f"{frame_file}"

                )


                evidence_path = (

                    os.path.join(

                        EVIDENCE_DIR,

                        evidence_filename

                    )

                )


                cv2.imwrite(

                    evidence_path,

                    annotated_image

                )


                previous_evidence_hash = (

                    current_hash

                )


                saved_evidence_count += 1


                print(

                    f"   → POTHOLES FOUND: "

                    f"{len(valid_detections)} "

                    f"| Best: "

                    f"{best_confidence * 100:.2f}%"

                )


            # =============================================
            # SAVE CSV RECORD
            # =============================================

            detection_data.append(

                {

                    "frame": frame_file,

                    "status": "POTHOLE_DETECTED",

                    "potholes_in_frame": (
                        len(
                            valid_detections
                        )
                    ),

                    "class_name": (
                        best_class_name
                    ),

                    "best_confidence": round(

                        best_confidence * 100,

                        2

                    ),

                    "evidence_image": (
                        evidence_path
                    )

                }

            )


        # =================================================
        # NO DETECTION
        # =================================================

        else:


            detection_data.append(

                {

                    "frame": frame_file,

                    "status": (
                        "NO_POTHOLE"
                    ),

                    "potholes_in_frame": 0,

                    "class_name": None,

                    "best_confidence": None,

                    "evidence_image": None

                }

            )


            print(

                "   → No pothole detected"

            )


    # =====================================================
    # SAVE RESULTS
    # =====================================================

    df = pd.DataFrame(
        detection_data
    )


    df.to_csv(

        RESULTS_FILE,

        index=False

    )


    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    print()

    print(
        "=" * 60
    )

    print(
        "SMART POTHOLE ANALYSIS COMPLETED"
    )

    print(
        "=" * 60
    )


    print(

        f"Total frames analyzed: "

        f"{len(frame_files)}"

    )


    print(

        f"Frames containing potholes: "

        f"{frames_with_potholes}"

    )


    print(

        f"Total potholes detected: "

        f"{total_potholes_detected}"

    )


    print(

        f"Filtered detections: "

        f"{rejected_detection_count}"

    )


    print(

        f"Duplicate evidence skipped: "

        f"{duplicate_frame_count}"

    )


    print(

        f"Unique evidence images: "

        f"{saved_evidence_count}"

    )


    print(

        f"Evidence folder: "

        f"{EVIDENCE_DIR}"

    )


    print(

        f"CSV report: "

        f"{RESULTS_FILE}"

    )


    print(
        "=" * 60
    )


if __name__ == "__main__":

    main()