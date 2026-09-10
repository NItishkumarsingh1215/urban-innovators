
import os
import sys
import cv2
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EVIDENCE_DIR = BASE_DIR / "evidence" / "smart_detections"
UPLOAD_DIR = BASE_DIR / "uploads" / "road_videos"

DATA_DIR.mkdir(parents=True, exist_ok=True)
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "smart_detection_results.csv"


def find_source():
    # 1) Explicit command-line source, if supplied.
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if p.exists() and p.is_file():
            return p

    # 2) Streamlit app's detector bridge.
    bridge = UPLOAD_DIR / "road_video.mp4"
    if bridge.exists() and bridge.stat().st_size > 0:
        return bridge

    # 3) Any recent uploaded video.
    candidates = []
    for p in UPLOAD_DIR.glob("*"):
        if p.is_file() and p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".m4v"}:
            candidates.append(p)
    if candidates:
        return max(candidates, key=lambda x: x.stat().st_mtime)

    return None


def clear_old_outputs():
    try:
        CSV_PATH.unlink(missing_ok=True)
    except Exception:
        pass
    # Do not delete other detector folders; this script owns only smart_detections.
    for p in EVIDENCE_DIR.glob("*"):
        if p.is_file():
            try:
                p.unlink()
            except Exception:
                pass


def save_csv(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=[
            "Frame", "Detection", "Confidence",
            "X1", "Y1", "X2", "Y2", "Source"
        ])
    df.to_csv(CSV_PATH, index=False)
    return df


def run_yolo(source):
    """
    Use a local pothole-specific YOLO model when one is present.
    We intentionally do not download a model during the Streamlit demo.
    """
    try:
        from ultralytics import YOLO
    except Exception:
        return []

    model_candidates = [
        BASE_DIR / "models" / "best_pothole.pt",
        BASE_DIR / "models" / "pothole_best.pt",
        BASE_DIR / "models" / "pothole_model.pt",
        BASE_DIR / "models" / "best.pt",
        BASE_DIR / "best_pothole.pt",
        BASE_DIR / "pothole_best.pt",
        BASE_DIR / "pothole_model.pt",
    ]

    model_path = next((p for p in model_candidates if p.exists()), None)
    if model_path is None:
        return []

    rows = []
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        return []

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total / fps if fps > 0 else 0
    # Sample the first 12 seconds for a fast live demo, while never skipping
    # the beginning where road defects are commonly visible.
    end_frame = min(total, int(max(1, min(duration, 12.0)) * fps))
    step = max(1, int(fps / 2.0))

    try:
        model = YOLO(str(model_path))
        names = getattr(model, "names", {}) or {}

        for frame_no in range(0, end_frame, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ok, frame = cap.read()
            if not ok:
                continue

            try:
                pred = model.predict(
                    source=frame,
                    conf=0.18,
                    iou=0.45,
                    verbose=False
                )
            except Exception:
                continue

            if not pred:
                continue

            result = pred[0]
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue

            annotated = result.plot()
            saved_for_frame = False

            for j in range(len(boxes)):
                try:
                    conf = float(boxes.conf[j].item())
                    cls_id = int(boxes.cls[j].item()) if boxes.cls is not None else 0
                    xyxy = boxes.xyxy[j].tolist()
                except Exception:
                    continue

                label = str(names.get(cls_id, "pothole")).lower()
                # A pothole-specific model normally has one class. For a model
                # with named classes, accept only pothole/road-defect labels.
                if len(names) > 1 and not any(
                    k in label for k in ("pothole", "road defect", "road_defect", "hole")
                ):
                    continue

                rows.append({
                    "Frame": int(frame_no),
                    "Detection": "POTHOLE",
                    "Confidence": round(conf, 4),
                    "X1": round(float(xyxy[0]), 1),
                    "Y1": round(float(xyxy[1]), 1),
                    "X2": round(float(xyxy[2]), 1),
                    "Y2": round(float(xyxy[3]), 1),
                    "Source": "LOCAL_YOLO"
                })
                saved_for_frame = True

            if saved_for_frame:
                target = EVIDENCE_DIR / f"pothole_frame_{frame_no:05d}.jpg"
                cv2.imwrite(str(target), annotated)

        cap.release()
        return rows
    except Exception:
        cap.release()
        return []


def surface_cavity_fallback(source):
    """
    Lightweight OpenCV fallback for the SIH prototype.
    It is deliberately independent of an external model, so pothole detection
    still works when no pothole .pt file is installed.
    """
    rows = []
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        return rows

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return rows

    # First 8 seconds, about 1.5 samples/sec.
    end_frame = min(total, int(8.0 * fps))
    step = max(1, int(fps / 1.5))

    saved_frames = set()
    detected_centers = []

    for frame_no in range(0, end_frame, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        if not ok:
            continue

        h, w = frame.shape[:2]
        # Road region. Ignore sky/buildings and most dashboard area.
        y0 = int(h * 0.38)
        roi = frame[y0:int(h * 0.98), :]
        rh, rw = roi.shape[:2]

        # Downscale large phone videos for speed.
        scale = min(1.0, 960.0 / max(rw, rh))
        if scale < 1.0:
            small = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            small = roi.copy()

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 0)

        # Local darkness: potholes are darker than their immediate road
        # surroundings. Black-hat emphasizes enclosed dark cavities.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

        # Adaptive darkness mask + edge support.
        _, dark = cv2.threshold(blackhat, 22, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(gray, 45, 120)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        mask = cv2.bitwise_and(dark, edges)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=2
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1
        )

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []
        image_area = float(max(1, small.shape[0] * small.shape[1]))

        for c in contours:
            area = cv2.contourArea(c)
            if area < image_area * 0.00025:
                continue
            if area > image_area * 0.15:
                continue

            x, y, cw, ch = cv2.boundingRect(c)
            if cw < 18 or ch < 12:
                continue

            aspect = cw / float(max(1, ch))
            if aspect < 0.25 or aspect > 4.5:
                continue

            # Prefer lower/middle road area and reasonably compact cavities.
            cy = y + ch / 2.0
            position_score = 1.0 if cy > small.shape[0] * 0.18 else 0.5

            # Mean black-hat response inside candidate.
            patch = blackhat[y:y+ch, x:x+cw]
            response = float(np.mean(patch)) if patch.size else 0.0

            score = (
                0.35 * min(1.0, area / (image_area * 0.01)) +
                0.45 * min(1.0, response / 70.0) +
                0.20 * position_score
            )

            if response < 16 or score < 0.28:
                continue

            candidates.append((score, area, x, y, cw, ch))

        if not candidates:
            continue

        candidates.sort(reverse=True)
        score, area, x, y, cw, ch = candidates[0]

        # Convert coordinates back to original frame.
        inv = 1.0 / scale
        x1 = int(x * inv)
        y1 = int(y * inv + y0)
        x2 = int((x + cw) * inv)
        y2 = int((y + ch) * inv + y0)

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # Avoid recording the exact same cavity on adjacent sampled frames.
        duplicate = False
        for pf, px, py in detected_centers:
            if abs(frame_no - pf) <= int(fps * 1.5) and abs(cx - px) < w * 0.10 and abs(cy - py) < h * 0.10:
                duplicate = True
                break
        if duplicate:
            continue

        detected_centers.append((frame_no, cx, cy))

        confidence = min(0.96, max(0.42, 0.42 + score * 0.50))

        rows.append({
            "Frame": int(frame_no),
            "Detection": "POTHOLE",
            "Confidence": round(float(confidence), 4),
            "X1": int(max(0, x1)),
            "Y1": int(max(0, y1)),
            "X2": int(min(w - 1, x2)),
            "Y2": int(min(h - 1, y2)),
            "Source": "OPENCV_SURFACE_CAVITY"
        })

        if len(saved_frames) < 20:
            annotated = frame.copy()
            cv2.rectangle(
                annotated, (x1, y1), (x2, y2),
                (0, 0, 255), max(2, int(min(w, h) / 450))
            )
            cv2.putText(
                annotated,
                f"POTHOLE {confidence:.2f}",
                (max(5, x1), max(28, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )
            target = EVIDENCE_DIR / f"pothole_frame_{frame_no:05d}.jpg"
            if cv2.imwrite(str(target), annotated):
                saved_frames.add(frame_no)

        # A few independent detections are enough for the prototype.
        if len(rows) >= 25:
            break

    cap.release()
    return rows


def main():
    source = find_source()
    clear_old_outputs()

    if source is None:
        save_csv([])
        print("ERROR: No road video found.")
        return 1

    print(f"Pothole detector input: {source}")

    rows = run_yolo(source)
    detector_name = "LOCAL_YOLO"

    # If a local model is missing, broken, or returns no potholes, always run
    # the model-free fallback. This is the key fix for the 0-record problem.
    if not rows:
        rows = surface_cavity_fallback(source)
        detector_name = "OPENCV_SURFACE_CAVITY"

    # Remove exact duplicates while preserving multiple potholes in one frame.
    if rows:
        df = pd.DataFrame(rows)
        df["_cx"] = (
            pd.to_numeric(df["X1"], errors="coerce") +
            pd.to_numeric(df["X2"], errors="coerce")
        ) / 2.0
        df["_cy"] = (
            pd.to_numeric(df["Y1"], errors="coerce") +
            pd.to_numeric(df["Y2"], errors="coerce")
        ) / 2.0
        df["_cx"] = (df["_cx"] / 20).round() * 20
        df["_cy"] = (df["_cy"] / 20).round() * 20
        df = df.drop_duplicates(
            subset=["Frame", "Detection", "_cx", "_cy"],
            keep="first"
        ).drop(columns=["_cx", "_cy"])
        rows = df.to_dict("records")

    df = save_csv(rows)

    # Create a grouped incident CSV for the Reports page.
    incident_path = DATA_DIR / "pothole_incidents.csv"
    if not df.empty:
        incidents = []
        group_id = 0
        last_frame = None

        for _, r in df.sort_values("Frame").iterrows():
            frame = int(float(r["Frame"]))
            if last_frame is None or frame - last_frame > 45:
                group_id += 1
            last_frame = frame

            incidents.append({
                "incident_id": f"POTHOLE_{group_id:03d}",
                "incident_type": "POTHOLE",
                "first_frame": frame,
                "last_frame": frame,
                "severity": "HIGH" if float(r["Confidence"]) >= 0.75 else "MEDIUM",
                "confidence": float(r["Confidence"]),
                "detection_source": str(r["Source"])
            })

        inc = pd.DataFrame(incidents)
        if not inc.empty:
            inc = inc.groupby("incident_id", as_index=False).agg({
                "incident_type": "first",
                "first_frame": "min",
                "last_frame": "max",
                "severity": "first",
                "confidence": "max",
                "detection_source": "first",
            })
            inc.to_csv(incident_path, index=False)
        else:
            pd.DataFrame().to_csv(incident_path, index=False)
    else:
        pd.DataFrame(columns=[
            "incident_id", "incident_type", "first_frame",
            "last_frame", "severity", "confidence", "detection_source"
        ]).to_csv(incident_path, index=False)

    print(f"Pothole detector mode: {detector_name}")
    print(f"Pothole records: {len(df)}")
    print(f"Evidence images: {len(list(EVIDENCE_DIR.glob('*.jpg')))}")
    print(f"CSV: {CSV_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
