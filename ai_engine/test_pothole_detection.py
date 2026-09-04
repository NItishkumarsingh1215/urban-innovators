from ultralytics import YOLO
import os


# Project paths
MODEL_PATH = "models/pothole_model.pt"
IMAGE_PATH = "data/extracted_frames/frame_0000.jpg"
OUTPUT_DIR = "evidence"


def main():

    # Check model file
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found -> {MODEL_PATH}")
        return

    # Check test image
    if not os.path.exists(IMAGE_PATH):
        print(f"ERROR: Image not found -> {IMAGE_PATH}")
        return

    # Create evidence folder if needed
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading pothole AI model...")

    # Load trained model
    model = YOLO(MODEL_PATH)

    print("Model loaded successfully.")
    print("Running pothole detection...")

    # Run detection
    results = model(
        IMAGE_PATH,
        conf=0.40,
        verbose=False
    )

    # Get first result
    result = results[0]

    # Print detection information
    print("\nDetection Results:")

    if result.boxes is None or len(result.boxes) == 0:
        print("No pothole detected in this frame.")

    else:
        for box in result.boxes:

            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            class_name = result.names[class_id]

            print(
                f"Detected: {class_name} | "
                f"Confidence: {confidence * 100:.2f}%"
            )

    # Create annotated image with bounding boxes
    annotated_image = result.plot()

    # Save evidence image
    output_path = os.path.join(
        OUTPUT_DIR,
        "pothole_detection_result.jpg"
    )

    import cv2
    cv2.imwrite(
        output_path,
        annotated_image
    )

    print(f"\nEvidence image saved successfully:")
    print(output_path)


if __name__ == "__main__":
    main()