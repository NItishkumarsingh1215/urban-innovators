from ultralytics import YOLO
import os


class PotholeDetector:

    def __init__(self, model_path):
        """
        Initialize the trained YOLO pothole detection model.
        """

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        self.model = YOLO(model_path)


    def detect(self, image_path, confidence=0.40):
        """
        Detect potholes in a single image.

        Returns:
            results -> YOLO detection results
        """

        results = self.model(
            image_path,
            conf=confidence,
            verbose=False
        )

        return results


    def get_detections(self, image_path, confidence=0.40):
        """
        Convert YOLO results into simple structured detections.
        """

        results = self.detect(
            image_path,
            confidence
        )

        detections = []

        for result in results:

            boxes = result.boxes

            if boxes is None:
                continue

            for box in boxes:

                class_id = int(
                    box.cls[0].item()
                )

                class_name = result.names[
                    class_id
                ]

                confidence_score = float(
                    box.conf[0].item()
                )

                coordinates = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .tolist()
                )

                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidence_score,
                        "bbox": coordinates
                    }
                )

        return detections