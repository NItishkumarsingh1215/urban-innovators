import streamlit as st
import cv2
import pandas as pd
import subprocess
import sys
import tempfile
import shutil
import json
import io
import zipfile
import hashlib
import re
import time
import urllib.parse
import urllib.request
import pydeck as pdk
from pathlib import Path

st.set_page_config(page_title="Urban Intelligence Platform", page_icon="🛣️", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads" / "road_videos"
DATA_DIR = BASE_DIR / "data"
EVIDENCE_DIR = BASE_DIR / "evidence"
AI_DIR = BASE_DIR / "ai_engine"

SMART = AI_DIR / "smart_pothole_detector.py"
TRAFFIC = AI_DIR / "vehicle_detector.py"
WATER = AI_DIR / "waterlogging_detector.py"
PEDESTRIAN = AI_DIR / "pedestrian_detector.py"
ANPR = AI_DIR / "anpr_detector.py"
FLEET = AI_DIR / "fleet_aggregator.py"
OD_ANALYSIS = AI_DIR / "od_analytics.py"
INCIDENT = AI_DIR / "incident_generator.py"
EDGE_PROCESSOR = AI_DIR / "edge_processor.py"

LIVE_FRAME_PATH = DATA_DIR / "live_frame.jpg"
LIVE_STOP_FILE = DATA_DIR / "stop_live_traffic.flag"
LIVE_ENGINE = BASE_DIR / "live_traffic_engine.py"

CSV_FILES = {
    "smart_detection_results.csv": DATA_DIR / "smart_detection_results.csv",
    "traffic_results.csv": DATA_DIR / "traffic_results.csv",
    "waterlogging_results.csv": DATA_DIR / "waterlogging_results.csv",
    "pedestrian_results.csv": DATA_DIR / "pedestrian_results.csv",
    "anpr_results.csv": DATA_DIR / "anpr_results.csv",
    "fleet_summary.csv": DATA_DIR / "fleet_summary.csv",
    "od_delay_results.csv": DATA_DIR / "od_delay_results.csv",
    "pothole_incidents.csv": DATA_DIR / "pothole_incidents.csv",
    "waterlogging_incidents.csv": DATA_DIR / "waterlogging_incidents.csv",
    "incidents.csv": DATA_DIR / "incidents.csv",
}

for d in [UPLOAD_DIR, DATA_DIR, EVIDENCE_DIR, AI_DIR]:
    d.mkdir(parents=True, exist_ok=True)

defaults = {
    "active_video_bytes": None,
    "browser_video_bytes": None,
    "active_video_name": None,
    "active_video_mime": "video/mp4",
    "active_video_signature": None,
    "video_location": None,
    "analysis_results": {},
    "evidence_memory": {},
    "analysis_complete": False,
    "pipeline_running": False,
    "pipeline_results": {},
    "live_mode": False,
    "live_process_pid": None
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def images():
    """Return all evidence images recursively from the project evidence folder."""
    if not EVIDENCE_DIR.exists():
        return []
    return sorted(
        [
            p for p in EVIDENCE_DIR.rglob("*")
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ],
        key=lambda p: str(p).lower()
    )


EVIDENCE_CATEGORY_LABELS = {
    "smart_detections": "🕳️ Pothole / Road Defect Evidence",
    "traffic_detections": "🚗 Traffic Evidence",
    "waterlogging_detections": "🌊 Waterlogging Evidence",
    "pedestrian_detections": "🚶‍♂️ Pedestrian Safety Evidence",
    "anpr_detections": "🔍 ANPR Evidence",
    "all_detections": "🚨 Incident Evidence",
    "edge_detections": "⚡ Edge AI Evidence",
}


def evidence_category(path):
    """Map an evidence file to a stable UI category."""
    try:
        rel = Path(path).resolve().relative_to(EVIDENCE_DIR.resolve())
        parts = rel.parts
        if len(parts) >= 2:
            folder = parts[0].lower()
        else:
            folder = "all_detections"
    except Exception:
        folder = "all_detections"

    if folder in EVIDENCE_CATEGORY_LABELS:
        return folder

    # Handle common alternate folder names.
    aliases = {
        "pothole_detections": "smart_detections",
        "pothole": "smart_detections",
        "traffic": "traffic_detections",
        "waterlogging": "waterlogging_detections",
        "pedestrian": "pedestrian_detections",
        "anpr": "anpr_detections",
        "edge": "edge_detections",
        "edge_ai": "edge_detections",
    }
    return aliases.get(folder, folder or "all_detections")


def evidence_snapshot():
    out = {}
    for p in images():
        try:
            s = p.stat()
            out[str(p.resolve())] = (s.st_mtime_ns, s.st_size)
        except Exception:
            pass
    return out


def _append_evidence_item(category, path, data):
    bucket = st.session_state["evidence_memory"].setdefault(category, [])
    digest = hashlib.sha256(data).hexdigest()
    existing = {
        x.get("sha256") for x in bucket
        if isinstance(x, dict) and x.get("sha256")
    }
    if digest in existing:
        return
    bucket.append({
        "name": path.name,
        "data": data,
        "sha256": digest,
        "path": str(path)
    })


def capture_new_evidence(category, before=None):
    """Capture evidence generated/updated during a detector run."""
    for p in images():
        key = str(p.resolve())
        if before is not None:
            try:
                s = p.stat()
                if before.get(key) == (s.st_mtime_ns, s.st_size):
                    continue
            except Exception:
                continue
        try:
            _append_evidence_item(category, p, p.read_bytes())
        except Exception:
            pass


def refresh_evidence_from_disk():
    """Load all existing evidence into session memory, including files created earlier."""
    memory = {}
    for p in images():
        category = evidence_category(p)
        try:
            data = p.read_bytes()
        except Exception:
            continue

        digest = hashlib.sha256(data).hexdigest()
        bucket = memory.setdefault(category, [])
        if digest not in {x["sha256"] for x in bucket}:
            bucket.append({
                "name": p.name,
                "data": data,
                "sha256": digest,
                "path": str(p)
            })

    st.session_state["evidence_memory"] = memory


def keep_existing_evidence():
    refresh_evidence_from_disk()


def clear_runtime_outputs():
    """
    Remove stale incident outputs before a new uploaded video is analyzed.
    Detection CSVs are cleared for a new uploaded video so old results never
    mix with the current video's results. The detectors recreate these files.
    """
    for key in [
        "smart_detection_results.csv",
        "traffic_results.csv",
        "waterlogging_results.csv",
        "pedestrian_results.csv",
        "anpr_results.csv",
        "fleet_summary.csv",
        "od_delay_results.csv",
        "pothole_incidents.csv",
        "waterlogging_incidents.csv",
        "incidents.csv",
    ]:
        p = CSV_FILES.get(key)
        if p and p.exists():
            try:
                p.unlink()
            except Exception:
                pass


def clear_evidence_folders_for_new_video():
    """
    Remove old detector evidence so the Evidence page never mixes evidence
    from an earlier video with the current upload.
    """
    for folder in [
        "smart_detections",
        "traffic_detections",
        "waterlogging_detections",
        "pedestrian_detections",
        "anpr_detections",
        "all_detections",
        "edge_detections",
    ]:
        p = EVIDENCE_DIR / folder
        if p.exists():
            try:
                shutil.rmtree(p)
            except Exception:
                pass
        p.mkdir(parents=True, exist_ok=True)

def find_ffprobe():
    found = shutil.which("ffprobe")
    if found:
        return found
    candidates = [
        Path("C:/ffmpeg/bin/ffprobe.exe"),
        Path("C:/Program Files/ffmpeg/bin/ffprobe.exe"),
        Path("C:/Program Files (x86)/ffmpeg/bin/ffprobe.exe"),
        Path.home() / "ffmpeg" / "bin" / "ffprobe.exe",
    ]
    for root in [Path.home() / "AppData/Local/Microsoft/WinGet/Packages", Path.home() / "scoop/apps", Path("C:/ProgramData/chocolatey/bin")]:
        if root.exists():
            try:
                candidates.extend(root.rglob("ffprobe.exe"))
            except Exception: pass
    for item in candidates:
        try:
            if item.exists() and item.is_file(): return str(item)
        except Exception: pass
    return None


def find_ffmpeg():
    found = shutil.which("ffmpeg")
    if found: return found
    candidates = [
        Path("C:/ffmpeg/bin/ffmpeg.exe"), Path("C:/Program Files/ffmpeg/bin/ffmpeg.exe"),
        Path("C:/Program Files (x86)/ffmpeg/bin/ffmpeg.exe"), Path.home() / "ffmpeg" / "bin" / "ffmpeg.exe",
    ]
    for root in [Path.home() / "AppData/Local/Microsoft/WinGet/Packages", Path.home() / "scoop/apps", Path("C:/ProgramData/chocolatey/bin")]:
        if root.exists():
            try: candidates.extend(root.rglob("ffmpeg.exe"))
            except Exception: pass
    ffprobe = find_ffprobe()
    if ffprobe:
        candidates.insert(0, Path(ffprobe).parent / "ffmpeg.exe")
    for item in candidates:
        try:
            if item.exists() and item.is_file(): return str(item)
        except Exception: pass
    return None


def prepare_browser_video(data, name):
    ffmpeg = find_ffmpeg()
    if not ffmpeg or not data: return data, False
    src, out = None, None
    try:
        src = temp_source(data, name)
        out = Path(tempfile.mktemp(suffix=".mp4"))
        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(src), "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-c:a", "aac", "-b:a", "128k", str(out)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0 and out.exists() and out.stat().st_size > 0:
            return out.read_bytes(), True
        return data, False
    except Exception: return data, False
    finally:
        if src:
            try: src.unlink(missing_ok=True)
            except Exception: pass
        if out:
            try: out.unlink(missing_ok=True)
            except Exception: pass


def parse_location(value):
    value = str(value).strip().strip('"').strip("'").rstrip("/").strip()
    match = re.search(r"([+-])(\d{1,3}(?:\.\d+)?)([+-])(\d{1,3}(?:\.\d+)?)", value)
    if not match: return None
    try:
        latitude = float(match.group(1) + match.group(2))
        longitude = float(match.group(3) + match.group(4))
    except ValueError: return None
    if -90 <= latitude <= 90 and -180 <= longitude <= 180:
        return {"latitude": latitude, "longitude": longitude, "source": "VIDEO_GPS_METADATA"}
    return None


def extract_gps(video):
    ffprobe = find_ffprobe()
    if not ffprobe: return None
    video = Path(video)
    cmd = [ffprobe, "-v", "error", "-show_entries", "format_tags:stream_tags", "-of", "json", str(video)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            metadata = json.loads(result.stdout)
            for key, value in metadata.get("format", {}).get("tags", {}).items():
                if str(key).lower() in {"location", "location-eng"}:
                    loc = parse_location(value)
                    if loc: return loc
            for stream in metadata.get("streams", []):
                for key, value in stream.get("tags", {}).items():
                    if str(key).lower() in {"location", "location-eng"}:
                        loc = parse_location(value)
                        if loc: return loc
    except Exception: pass
    cmd = [ffprobe, "-v", "error", "-show_entries", "format_tags:stream_tags", "-of", "default=noprint_wrappers=1", str(video)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "location=" in line.lower():
                    loc = parse_location(line.split("=", 1)[1].strip())
                    if loc: return loc
    except Exception: pass
    return None


def reverse_geocode(lat, lon):
    try:
        q = urllib.parse.urlencode({"lat":lat,"lon":lon,"format":"jsonv2","zoom":18})
        req = urllib.request.Request("https://nominatim.openstreetmap.org/reverse?"+q, headers={"User-Agent":"SIH26124-Urban-Intelligence/1.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read().decode())
        a = data.get("address", {})
        vals = []
        for k in ["road","neighbourhood","suburb","city","town","village","state"]:
            if a.get(k) and a[k] not in vals:
                vals.append(a[k])
        return ", ".join(vals[:4]) or data.get("display_name")
    except Exception: return None


def signature(name, data):
    return f"{name}|{len(data)}|{hashlib.sha256(data).hexdigest()[:16]}"


def temp_source(data, name):
    suffix = Path(name).suffix or ".mp4"
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    f.write(data)
    f.close()
    return Path(f.name)


def detector_bridge(source):
    bridge = UPLOAD_DIR / "road_video.mp4"
    shutil.copy2(source, bridge)
    return bridge


def cleanup_bridge():
    for p in [UPLOAD_DIR/"road_video.mp4", UPLOAD_DIR/"_streamlit_temp_video.mp4"]:
        try: p.unlink(missing_ok=True)
        except Exception: pass


def run_detector(script, source):
    if not script.exists(): return False, f"{script.name} not found"
    before = evidence_snapshot()
    try:
        detector_bridge(source)
        r = subprocess.run([sys.executable, str(script)], cwd=str(BASE_DIR), capture_output=True, text=True, timeout=900)
        msg = r.stdout.strip() or r.stderr.strip()
        return r.returncode == 0, msg
    except Exception as e: return False, str(e)
    finally: cleanup_bridge()


def collect_csvs():
    for name, path in CSV_FILES.items():
        if path.exists():
            try: st.session_state["analysis_results"][name] = pd.read_csv(path)
            except Exception: pass


def load_csv(path):
    name = Path(path).name
    if name in st.session_state["analysis_results"]:
        return st.session_state["analysis_results"][name].copy()
    if Path(path).exists():
        try: return pd.read_csv(path)
        except Exception: pass
    return pd.DataFrame()


def _first_existing_column(df, candidates):
    lower = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    for c in df.columns:
        cl = str(c).lower()
        if any(candidate.lower() in cl for candidate in candidates):
            return c
    return None


def _numeric_value(row, column, default=None):
    if column is None:
        return default
    try:
        value = pd.to_numeric(row.get(column), errors="coerce")
        if pd.notna(value):
            return float(value)
    except Exception:
        pass
    return default


def build_real_incidents():
    """
    Build the central incidents.csv only from actual detector output.
    No demo coordinates and no DEMO_SIMULATED records are created.
    """
    records = []
    loc = st.session_state.get("video_location")

    smart = load_csv(CSV_FILES["smart_detection_results.csv"])
    if not smart.empty:
        frame_col = _first_existing_column(
            smart, ["Frame", "frame", "frame_number", "frame_id", "Frame_Number"]
        )
        conf_col = _first_existing_column(
            smart, ["Confidence", "confidence", "Score", "score", "confidence_score"]
        )

        for i, (_, row) in enumerate(smart.iterrows(), start=1):
            frame = _numeric_value(row, frame_col, i)
            conf = _numeric_value(row, conf_col, None)
            if conf is not None and conf <= 1:
                severity = "HIGH" if conf >= 0.75 else ("MEDIUM" if conf >= 0.45 else "LOW")
            else:
                severity = "MEDIUM"

            rec = {
                "incident_id": f"POTHOLE_{i:03d}",
                "incident_type": "POTHOLE",
                "status": "DETECTED",
                "severity": severity,
                "first_frame": int(frame) if frame is not None else i,
                "last_frame": int(frame) if frame is not None else i,
                "detection_source": "SMART_POTHOLE_DETECTOR",
            }
            if conf is not None:
                rec["confidence"] = round(conf, 4)

            if loc:
                rec["Latitude"] = loc["latitude"]
                rec["Longitude"] = loc["longitude"]
                rec["Location_Source"] = loc["source"]
                rec["Location"] = loc.get(
                    "label",
                    f'{loc["latitude"]:.6f}, {loc["longitude"]:.6f}'
                )
            else:
                rec["Latitude"] = None
                rec["Longitude"] = None
                rec["Location_Source"] = "GPS_UNAVAILABLE"
                rec["Location"] = "GPS unavailable in video metadata"

            records.append(rec)

    water = load_csv(CSV_FILES["waterlogging_results.csv"])
    if not water.empty:
        frame_col = _first_existing_column(
            water, ["Frame", "frame", "frame_number", "frame_id", "Frame_Number"]
        )
        risk_col = _first_existing_column(
            water, ["Risk_Level", "risk_level", "Risk", "risk"]
        )
        conf_col = _first_existing_column(
            water, ["Confidence", "confidence", "Score", "score"]
        )

        for i, (_, row) in enumerate(water.iterrows(), start=1):
            frame = _numeric_value(row, frame_col, i)
            risk = str(row.get(risk_col, "MEDIUM")).upper() if risk_col else "MEDIUM"
            if risk not in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}:
                risk = "MEDIUM"

            rec = {
                "incident_id": f"WATER_{i:03d}",
                "incident_type": "WATERLOGGING",
                "status": "DETECTED",
                "severity": risk,
                "first_frame": int(frame) if frame is not None else i,
                "last_frame": int(frame) if frame is not None else i,
                "detection_source": "WATERLOGGING_DETECTOR",
            }
            conf = _numeric_value(row, conf_col, None)
            if conf is not None:
                rec["confidence"] = round(conf, 4)

            if loc:
                rec["Latitude"] = loc["latitude"]
                rec["Longitude"] = loc["longitude"]
                rec["Location_Source"] = loc["source"]
                rec["Location"] = loc.get(
                    "label",
                    f'{loc["latitude"]:.6f}, {loc["longitude"]:.6f}'
                )
            else:
                rec["Latitude"] = None
                rec["Longitude"] = None
                rec["Location_Source"] = "GPS_UNAVAILABLE"
                rec["Location"] = "GPS unavailable in video metadata"

            records.append(rec)

    result = pd.DataFrame(records)

    # Never allow old/demo rows to survive in the central report.
    if not result.empty:
        result = result.drop_duplicates(
            subset=["incident_type", "first_frame", "last_frame", "detection_source"],
            keep="first"
        )

    result.to_csv(CSV_FILES["incidents.csv"], index=False)
    return result


def _run_python_detector(module_name, function_name, source, **kwargs):
    """Safely execute a detector module/function and return (success, message)."""
    try:
        if str(AI_DIR) not in sys.path:
            sys.path.append(str(AI_DIR))
        module = __import__(module_name)
        function = getattr(module, function_name)
        bridge_path = detector_bridge(source)
        try:
            result = function(video_path=bridge_path, **kwargs)
        finally:
            cleanup_bridge()

        if isinstance(result, dict):
            return bool(result.get("success", False)), str(
                result.get("message", "Completed")
            )
        return True, "Completed"
    except Exception as e:
        cleanup_bridge()
        return False, str(e)


def _model_path_candidates():
    return [
        BASE_DIR / "models" / "pothole_model.pt",
        BASE_DIR / "models" / "best.pt",
        BASE_DIR / "pothole_model.pt",
        BASE_DIR / "yolov8n.pt",
    ]


def _video_frame_indices(video_path, max_frames=12):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if total <= 0:
        return []
    count = min(max_frames, total)
    return sorted(set(
        int(x) for x in pd.Series(
            [i * (total - 1) / max(1, count - 1) for i in range(count)]
        ).round().tolist()
    ))


def _generate_vehicle_evidence_fallback(source, max_images=12):
    """
    If vehicle_detector.py produced CSV but no images, create a small set of
    annotated evidence frames from the current uploaded video.
    """
    out_dir = EVIDENCE_DIR / "traffic_detections"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
    except Exception:
        return 0

    model_path = BASE_DIR / "yolov8n.pt"
    if not model_path.exists():
        return 0

    try:
        model = YOLO(str(model_path))
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            return 0

        indices = _video_frame_indices(source, max_images)
        saved = 0

        for frame_no in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ok, frame = cap.read()
            if not ok:
                continue

            results = model.predict(
                source=frame,
                conf=0.35,
                verbose=False,
                classes=[2, 3, 5, 7],
            )
            if not results:
                continue

            result = results[0]
            if result.boxes is None or len(result.boxes) == 0:
                continue

            annotated = result.plot()
            target = out_dir / f"traffic_frame_{frame_no:05d}.jpg"
            cv2.imwrite(str(target), annotated)
            saved += 1

            if saved >= max_images:
                break

        cap.release()
        return saved
    except Exception:
        return 0


def _generate_pothole_evidence_fallback(source, max_images=20):
    """
    If the smart pothole detector did not save evidence images, use the
    project pothole model (when present) to create annotated evidence frames.
    """
    out_dir = EVIDENCE_DIR / "smart_detections"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
    except Exception:
        return 0

    model_path = None
    for candidate in [
        BASE_DIR / "models" / "pothole_model.pt",
        BASE_DIR / "models" / "best.pt",
    ]:
        if candidate.exists():
            model_path = candidate
            break

    if model_path is None:
        return 0

    try:
        model = YOLO(str(model_path))
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            return 0

        indices = _video_frame_indices(source, max_images)
        saved = 0

        for frame_no in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ok, frame = cap.read()
            if not ok:
                continue

            results = model.predict(
                source=frame,
                conf=0.20,
                verbose=False,
            )
            if not results:
                continue

            result = results[0]
            if result.boxes is None or len(result.boxes) == 0:
                continue

            annotated = result.plot()
            target = out_dir / f"pothole_frame_{frame_no:05d}.jpg"
            cv2.imwrite(str(target), annotated)
            saved += 1

            if saved >= max_images:
                break

        cap.release()
        return saved
    except Exception:
        return 0



def _save_atomic_image(path, image):
    """Write an image atomically so the UI never reads a half-written JPEG."""
    path = Path(path)
    tmp = path.with_name(path.stem + "_tmp" + path.suffix)
    ok = cv2.imwrite(str(tmp), image)
    if ok:
        try:
            tmp.replace(path)
        except Exception:
            try:
                shutil.move(str(tmp), str(path))
            except Exception:
                pass
    return ok


def _rescan_potholes_if_needed(source, max_frames=140):
    """
    Second-pass pothole scan. It is only used when the primary detector returns
    very few records. More frames + a lower confidence threshold make the
    prototype less likely to report just one frame.
    """
    csv_path = CSV_FILES["smart_detection_results.csv"]
    current = load_csv(csv_path)

    # Don't duplicate work when the detector already found a healthy amount.
    if len(current) >= 5:
        return 0

    model_path = None
    for candidate in [
        BASE_DIR / "models" / "pothole_model.pt",
        BASE_DIR / "models" / "best.pt",
    ]:
        if candidate.exists():
            model_path = candidate
            break
    if model_path is None:
        return 0

    try:
        from ultralytics import YOLO
        model = YOLO(str(model_path))
    except Exception:
        return 0

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return 0

    step = max(1, total // max_frames)
    evidence_dir = EVIDENCE_DIR / "smart_detections"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    existing_keys = set()
    if not current.empty:
        fcol = _first_existing_column(current, ["Frame", "frame", "frame_number", "frame_id"])
        dcol = _first_existing_column(current, ["Detection", "Class", "class", "Label", "label"])
        for _, r in current.iterrows():
            frame_value = int(_numeric_value(r, fcol, -1) or -1)
            label_value = str(r.get(dcol, "POTHOLE")).upper()
            x1c = _numeric_value(r, _first_existing_column(current, ["X1", "x1"]), None)
            y1c = _numeric_value(r, _first_existing_column(current, ["Y1", "y1"]), None)
            x2c = _numeric_value(r, _first_existing_column(current, ["X2", "x2"]), None)
            y2c = _numeric_value(r, _first_existing_column(current, ["Y2", "y2"]), None)
            if None not in (x1c, y1c, x2c, y2c):
                cx = round(((x1c + x2c) / 2.0) / 20.0) * 20
                cy = round(((y1c + y2c) / 2.0) / 20.0) * 20
                existing_keys.add((frame_value, label_value, int(cx), int(cy)))
            else:
                # Legacy rows without box coordinates are treated as frame-level.
                existing_keys.add((frame_value, label_value, -1, -1))

    rows = []
    saved = 0

    for frame_no in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        if not ok:
            continue

        try:
            preds = model.predict(
                source=frame,
                conf=0.18,
                iou=0.45,
                verbose=False
            )
        except Exception:
            continue

        if not preds:
            continue

        result = preds[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue

        annotated = result.plot()
        added_this_frame = False

        for j in range(len(boxes)):
            try:
                conf = float(boxes.conf[j].item())
                xyxy = boxes.xyxy[j].tolist()
            except Exception:
                continue

            # Use frame + approximate box centre as the identity, so
            # multiple potholes in the same frame are all retained.
            cx = round((float(xyxy[0]) + float(xyxy[2])) / 2.0 / 20.0) * 20
            cy = round((float(xyxy[1]) + float(xyxy[3])) / 2.0 / 20.0) * 20
            key = (int(frame_no), "POTHOLE", int(cx), int(cy))
            if key in existing_keys:
                continue

            rows.append({
                "Frame": int(frame_no),
                "Detection": "POTHOLE",
                "Confidence": round(conf, 4),
                "X1": round(float(xyxy[0]), 1),
                "Y1": round(float(xyxy[1]), 1),
                "X2": round(float(xyxy[2]), 1),
                "Y2": round(float(xyxy[3]), 1),
                "Source": "SECOND_PASS_YOLO"
            })
            existing_keys.add(key)
            added_this_frame = True

        if added_this_frame and saved < 25:
            target = evidence_dir / f"pothole_frame_{frame_no:05d}.jpg"
            if _save_atomic_image(target, annotated):
                saved += 1

    cap.release()

    if rows:
        merged = pd.concat([current, pd.DataFrame(rows)], ignore_index=True)
        # Keep separate potholes even when they occur in the same frame.
        fcol = _first_existing_column(merged, ["Frame", "frame", "frame_number", "frame_id"])
        dcol = _first_existing_column(merged, ["Detection", "Class", "class", "Label", "label"])
        x1col = _first_existing_column(merged, ["X1", "x1"])
        y1col = _first_existing_column(merged, ["Y1", "y1"])
        x2col = _first_existing_column(merged, ["X2", "x2"])
        y2col = _first_existing_column(merged, ["Y2", "y2"])
        if fcol and dcol and all([x1col, y1col, x2col, y2col]):
            merged["_p_cx"] = (
                pd.to_numeric(merged[x1col], errors="coerce")
                + pd.to_numeric(merged[x2col], errors="coerce")
            ) / 2
            merged["_p_cy"] = (
                pd.to_numeric(merged[y1col], errors="coerce")
                + pd.to_numeric(merged[y2col], errors="coerce")
            ) / 2
            merged["_p_cx"] = (merged["_p_cx"] / 20).round() * 20
            merged["_p_cy"] = (merged["_p_cy"] / 20).round() * 20
            merged = merged.drop_duplicates(
                subset=[fcol, dcol, "_p_cx", "_p_cy"], keep="first"
            ).drop(columns=["_p_cx", "_p_cy"], errors="ignore")
        elif fcol and dcol:
            merged = merged.drop_duplicates(subset=[fcol, dcol], keep="first")
        merged.to_csv(csv_path, index=False)
        st.session_state["analysis_results"][csv_path.name] = merged.copy()

    return len(rows)


def _waterlogging_fallback_scan(source, max_frames=120):
    """
    More permissive HSV/contour second pass for the waterlogging prototype.
    It does not claim deep-learning accuracy; it simply ensures the current
    video is sampled broadly and multiple qualifying water regions can be
    recorded when present.
    """
    csv_path = CSV_FILES["waterlogging_results.csv"]
    current = load_csv(csv_path)

    if len(current) >= 5:
        return 0

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return 0

    step = max(1, total // max_frames)
    evidence_dir = EVIDENCE_DIR / "waterlogging_detections"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    existing_frames = set()
    if not current.empty:
        fcol = _first_existing_column(current, ["Frame", "frame", "frame_number", "frame_id"])
        for _, r in current.iterrows():
            val = _numeric_value(r, fcol, None)
            if val is not None:
                existing_frames.add(int(val))

    rows = []
    saved = 0

    for frame_no in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        if not ok:
            continue

        h, w = frame.shape[:2]
        # Road-focused lower 70% ROI; ignore sky/buildings.
        roi_y = int(h * 0.30)
        roi = frame[roi_y:h, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Broad dark/wet + low-saturation reflection masks.
        dark = cv2.inRange(hsv, (0, 0, 20), (180, 255, 135))
        blue = cv2.inRange(hsv, (85, 15, 35), (135, 255, 230))
        mask = cv2.bitwise_or(dark, blue)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < max(500, (w * h) * 0.002):
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            if cw < 30 or ch < 15:
                continue
            candidates.append((area, x, y + roi_y, cw, ch))

        if not candidates:
            continue

        candidates.sort(reverse=True)
        area, x, y, cw, ch = candidates[0]
        score = min(0.99, max(0.20, area / float(max(1, w * h)) * 3.0))

        if frame_no in existing_frames:
            continue

        annotated = frame.copy()
        cv2.rectangle(annotated, (x, y), (x + cw, y + ch), (255, 0, 0), 2)
        cv2.putText(
            annotated,
            f"WATERLOGGING {score:.2f}",
            (max(5, x), max(25, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 0, 0),
            2
        )

        risk = "HIGH" if score >= 0.60 else ("MEDIUM" if score >= 0.35 else "LOW")
        rows.append({
            "Frame": int(frame_no),
            "Detection": "WATERLOGGING",
            "Score": round(float(score), 4),
            "Confidence": round(float(score), 4),
            "Risk_Level": risk,
            "Source": "HSV_SECOND_PASS"
        })
        existing_frames.add(frame_no)

        if saved < 25:
            target = evidence_dir / f"waterlogging_frame_{frame_no:05d}.jpg"
            if _save_atomic_image(target, annotated):
                saved += 1

    cap.release()

    if rows:
        merged = pd.concat([current, pd.DataFrame(rows)], ignore_index=True)
        fcol = _first_existing_column(merged, ["Frame", "frame", "frame_number", "frame_id"])
        if fcol:
            merged = merged.drop_duplicates(subset=[fcol], keep="first")
        merged.to_csv(csv_path, index=False)
        st.session_state["analysis_results"][csv_path.name] = merged.copy()

    return len(rows)


def run_pipeline(source, progress):
    results = {}

    # Always start with a clean evidence view for the new video.
    clear_evidence_folders_for_new_video()
    clear_runtime_outputs()
    st.session_state["evidence_memory"] = {}

    # 0. Edge AI
    progress.info("⏳ ⚡ Running Edge AI Processor...")
    try:
        if str(AI_DIR) not in sys.path:
            sys.path.append(str(AI_DIR))
        import edge_processor
        edge_processor.process_edge_stream(str(source))
        edge_ok = True
        edge_msg = "Edge processing completed successfully."
    except Exception as e:
        edge_ok = False
        edge_msg = str(e)
    results["⚡ Edge AI Processing"] = {"success": edge_ok, "message": edge_msg}

    # 1. Pothole
    progress.info("⏳ 🕳️ Pothole Detection...")
    before = evidence_snapshot()
    ok, msg = run_detector(SMART, source)
    capture_new_evidence("smart_detections", before)
    pothole_evidence_count = sum(
        1 for p in images() if evidence_category(p) == "smart_detections"
    )
    if pothole_evidence_count < 3:
        fallback_count = _generate_pothole_evidence_fallback(
            source, max_images=20
        )
        if fallback_count:
            msg += f" | Generated {fallback_count} fallback evidence images."
    extra_potholes = _rescan_potholes_if_needed(source)
    if extra_potholes:
        msg += f" | Second-pass scan added {extra_potholes} additional pothole detections."
        capture_new_evidence("smart_detections", before)
    results["🕳️ Pothole Detection"] = {"success": ok, "message": msg}

    # 2. Traffic
    progress.info("⏳ 🚗 Traffic Detection...")
    before = evidence_snapshot()
    ok, msg = run_detector(TRAFFIC, source)
    capture_new_evidence("traffic_detections", before)
    traffic_evidence_count = sum(
        1 for p in images() if evidence_category(p) == "traffic_detections"
    )
    if traffic_evidence_count < 3:
        fallback_count = _generate_vehicle_evidence_fallback(
            source, max_images=20
        )
        if fallback_count:
            msg += f" | Generated {fallback_count} fallback evidence images."
    results["🚗 Traffic Detection"] = {"success": ok, "message": msg}

    # 3. Waterlogging
    progress.info("⏳ 🌊 Waterlogging Detection...")
    before = evidence_snapshot()
    ok, msg = _run_python_detector(
        "waterlogging_detector",
        "analyze_video",
        source,
        output_csv=str(DATA_DIR / "waterlogging_results.csv"),
        evidence_dir=str(EVIDENCE_DIR / "waterlogging_detections"),
    )
    capture_new_evidence("waterlogging_detections", before)
    extra_water = _waterlogging_fallback_scan(source)
    if extra_water:
        msg += f" | Second-pass scan added {extra_water} additional waterlogging detections."
        capture_new_evidence("waterlogging_detections", before)
    results["🌊 Waterlogging Detection"] = {"success": ok, "message": msg}

    # 4. Pedestrian
    progress.info("⏳ 🚶‍♂️ Pedestrian Safety Detection...")
    before = evidence_snapshot()
    ok, msg = _run_python_detector(
        "pedestrian_detector",
        "analyze_pedestrians",
        source,
        output_csv=str(DATA_DIR / "pedestrian_results.csv"),
        evidence_dir=str(EVIDENCE_DIR / "pedestrian_detections"),
    )
    capture_new_evidence("pedestrian_detections", before)
    results["🚶‍♂️ Pedestrian Safety"] = {"success": ok, "message": msg}

    # 5. ANPR
    progress.info("⏳ 🔍 ANPR & Offender Tracking...")
    before = evidence_snapshot()
    ok, msg = _run_python_detector(
        "anpr_detector",
        "analyze_anpr",
        source,
        output_csv=str(DATA_DIR / "anpr_results.csv"),
        evidence_dir=str(EVIDENCE_DIR / "anpr_detections"),
    )
    capture_new_evidence("anpr_detections", before)
    results["🔍 ANPR & Offenders"] = {"success": ok, "message": msg}

    # 6. Fleet
    progress.info("⏳ 🚌 Bus Fleet Aggregation...")
    try:
        if str(AI_DIR) not in sys.path:
            sys.path.append(str(AI_DIR))
        import fleet_aggregator
        res = fleet_aggregator.aggregate_fleet_data(
            output_csv=str(DATA_DIR / "fleet_summary.csv")
        )
        ok = bool(res.get("success", False))
        msg = str(res.get("message", "Completed"))
    except Exception as e:
        ok = False
        msg = str(e)
    results["🚌 Fleet Aggregation"] = {"success": ok, "message": msg}

    # 7. OD
    progress.info("⏳ 📈 Route Delay & OD Analytics...")
    try:
        if str(AI_DIR) not in sys.path:
            sys.path.append(str(AI_DIR))
        import od_analytics
        res = od_analytics.analyze_od_and_delays(
            output_csv=str(DATA_DIR / "od_delay_results.csv")
        )
        ok = bool(res.get("success", False))
        msg = str(res.get("message", "Completed"))
    except Exception as e:
        ok = False
        msg = str(e)
    results["📈 Route Delay & OD"] = {"success": ok, "message": msg}

    # 8. Central incidents — built ONLY from actual detector CSVs.
    progress.info("⏳ 🚨 Building Incident Report from real detections...")
    try:
        incident_df = build_real_incidents()
        ok = True
        msg = f"{len(incident_df)} real detector incidents compiled."
    except Exception as e:
        ok = False
        msg = str(e)
    results["🚨 Incident Generation"] = {"success": ok, "message": msg}

    apply_location = lambda: None  # kept as a no-op compatibility hook
    collect_csvs()
    refresh_evidence_from_disk()

    st.session_state["pipeline_results"] = results
    st.session_state["analysis_complete"] = True
    st.session_state["pipeline_running"] = False


def video_info(data):
    if not data: return None
    p = temp_source(data, "video.mp4")
    try:
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened(): return None
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return {"frames":frames,"fps":fps,"width":w,"height":h, "duration":frames/fps if fps else 0}
    finally:
        try: p.unlink()
        except Exception: pass


st.markdown("""<style>
.main-title{font-size:38px;font-weight:700;text-align:center}
.subtitle{text-align:center;color:#777;margin-bottom:25px}
.section-title{font-size:26px;font-weight:700;margin:10px 0 15px}
</style>""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">🛣️ AI-Powered Urban Road Intelligence Platform</div>'
    '<div class="subtitle">Smart Pothole • Traffic • Waterlogging • Pedestrian • ANPR • Fleet • Route Delay & OD • GIS Heatmap • Incidents</div>',
    unsafe_allow_html=True
)

st.sidebar.title("🛠️ Navigation")
page = st.sidebar.radio("Select Module", [
    "🏠 Dashboard",
    "🎥 Road Video",
    "⚡ Edge AI Analytics",
    "🤖 AI Detection",
    "🚗 Traffic Intelligence",
    "🚦 Live Traffic System",
    "🌊 Waterlogging Detection",
    "🚶‍♂️ Pedestrian Safety",
    "🔍 ANPR & Offenders",
    "🚌 Bus Fleet Aggregation",
    "📈 Route Delay & OD",
    "🗺️ GIS & Heatmap",
    "🚨 Incident Analysis",
    "📍 Location Intelligence",
    "📸 Evidence",
    "📊 Reports"
])

if st.session_state["active_video_bytes"] is not None:
    st.sidebar.success("🎥 Video active in this session")
    st.sidebar.caption(st.session_state.get("active_video_name","Current video"))
else:
    st.sidebar.info("🎥 No video uploaded")


if page == "🏠 Dashboard":
    st.markdown('<div class="section-title">Project Dashboard</div>', unsafe_allow_html=True)
    smart = load_csv(CSV_FILES["smart_detection_results.csv"])
    traffic = load_csv(CSV_FILES["traffic_results.csv"])
    water = load_csv(CSV_FILES["waterlogging_results.csv"])
    pedestrian = load_csv(CSV_FILES["pedestrian_results.csv"])
    anpr = load_csv(CSV_FILES["anpr_results.csv"])
    fleet = load_csv(CSV_FILES["fleet_summary.csv"])
    od_df = load_csv(CSV_FILES["od_delay_results.csv"])
    incidents = load_csv(CSV_FILES["incidents.csv"])
    refresh_evidence_from_disk()
    ev = st.session_state["evidence_memory"]
    
    a,b,c,d = st.columns(4)
    a.metric("🎥 Video", "Available" if st.session_state["active_video_bytes"] else "Not Uploaded")
    b.metric("🕳️ Potholes", len(smart))
    c.metric("🚗 Traffic", len(traffic))
    d.metric("🌊 Water", len(water))
    
    e,f,g,h = st.columns(4)
    e.metric("🚌 Fleet Buses", len(fleet))
    f.metric("📈 OD Routes", len(od_df))
    g.metric("🚨 Incidents", len(incidents))
    h.metric("📸 Total Evidence", sum(len(v) for v in ev.values()))
    
    loc = st.session_state.get("video_location")
    if loc:
        st.success(f'📍 {loc["latitude"]:.6f}, {loc["longitude"]:.6f} — {loc.get("label","GPS")}')
    elif st.session_state["active_video_bytes"]:
        st.warning("No GPS metadata found. Random location is NOT used.")

elif page == "🎥 Road Video":
    st.markdown('<div class="section-title">Road Video Input</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload Original Road Video", type=["mp4","avi","mov","mkv","m4v"], key="road_video")
    if uploaded is not None:
        data = uploaded.getvalue()
        sig = signature(uploaded.name, data)
        if sig != st.session_state["active_video_signature"]:
            st.session_state["active_video_bytes"] = data
            st.session_state["active_video_name"] = uploaded.name
            st.session_state["active_video_mime"] = "video/mp4"
            with st.spinner("🎬 Video playback prepare ho raha hai..."):
                browser_bytes, converted = prepare_browser_video(data, uploaded.name)
            st.session_state["browser_video_bytes"] = browser_bytes
            st.session_state["browser_video_converted"] = converted
            st.session_state["active_video_signature"] = sig
            st.session_state["analysis_results"] = {}
            st.session_state["evidence_memory"] = {}
            st.session_state["pipeline_results"] = {}
            st.session_state["analysis_complete"] = False
            st.session_state["video_location"] = None
            clear_runtime_outputs()
            clear_evidence_folders_for_new_video()
            src = temp_source(data, uploaded.name)
            try:
                loc = extract_gps(src)
                if loc:
                    loc["label"] = reverse_geocode(loc["latitude"], loc["longitude"]) or "Actual GPS coordinates"
                    st.session_state["video_location"] = loc
                    st.success(f'📍 Actual GPS found: {loc["latitude"]:.6f}, {loc["longitude"]:.6f}')
                else:
                    st.session_state["video_location"] = None
                    if find_ffprobe() is None: st.error("❌ FFprobe is not available. No fake location is used.")
                    else: st.warning("⚠️ Actual GPS metadata could not be read. No fake location is used.")
                progress = st.empty()
                with st.spinner("Edge AI Filtering → Pothole → Traffic → Waterlogging → Pedestrian → ANPR → Fleet → OD Analytics → Incidents..."):
                    run_pipeline(src, progress)
                st.success("🎉 Complete analysis and edge filtering finished.")
            except Exception as e: st.error(f"Pipeline error: {e}")
            finally:
                try: src.unlink()
                except Exception: pass
                cleanup_bridge()
    if st.session_state["active_video_bytes"]:
        st.subheader(f'🎬 {st.session_state.get("active_video_name","Current Video")}')
        playback_bytes = st.session_state.get("browser_video_bytes") or st.session_state["active_video_bytes"]
        st.video(playback_bytes, format="video/mp4")
        if st.session_state.get("browser_video_converted"): st.caption("🎬 Browser-compatible video prepared automatically.")
        info = video_info(st.session_state["active_video_bytes"])
        if info:
            a,b,c,d = st.columns(4)
            a.metric("Frames", info["frames"]); b.metric("FPS", f'{info["fps"]:.2f}')
            c.metric("Resolution", f'{info["width"]} × {info["height"]}')
            d.metric("Duration", f'{info["duration"]:.1f} sec')
    else: st.info("👆 Video upload karo. Upload ke baad automatic Edge AI pipeline chalega.")

elif page == "⚡ Edge AI Analytics":
    st.markdown('<div class="section-title">⚡ Edge AI Bandwidth & Data Filtering Metrics</div>', unsafe_allow_html=True)
    st.info("Local Edge processing metrics showcasing bandwidth optimization by discarding raw video streams and retaining only lightweight metadata & evidence snapshots.")
    
    edge_json_path = BASE_DIR / "edge_metadata_payload.json"
    
    raw_size_mb = 0.0
    if st.session_state.get("active_video_bytes") is not None:
        raw_size_mb = len(st.session_state["active_video_bytes"]) / (1024 * 1024)
    else:
        raw_size_mb = 0.0
        
    if edge_json_path.exists():
        try:
            with open(edge_json_path, "r") as f:
                edge_data = json.load(f)
            
            filtered_count = len(edge_data)
            payload_size_kb = filtered_count * 35.0  # Estimated KB per record + snapshot
            payload_size_mb = payload_size_kb / 1024.0
            
            savings_mb = max(0.0, raw_size_mb - payload_size_mb)
            savings_pct = (savings_mb / raw_size_mb) * 100 if raw_size_mb > 0 else 0.0
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📹 Total Raw Data (Avoided)", f"{raw_size_mb:.2f} MB")
            col2.metric("📦 Filtered Edge Payload", f"{payload_size_mb:.3f} MB")
            col3.metric("📉 Bandwidth Saved", f"{savings_pct:.1f}%")
            col4.metric("🚨 Filtered Incidents", filtered_count)
            
            st.success(f"✅ Edge AI successfully filtered and saved {filtered_count} valid incident frames while dropping redundant clean road frames.")
            
            if edge_data:
                st.subheader("📋 Filtered Edge Metadata Payload (Sent to Central Server)")
                st.dataframe(pd.DataFrame(edge_data), use_container_width=True)
            else:
                st.warning("⚠️ No hazardous incidents found in the current video frames to filter.")
        except Exception as e:
            st.error(f"Error reading edge metadata: {e}")
    else:
        st.warning("⚠️ Edge metadata payload not found. Please upload a road video in the 'Road Video' tab first to trigger edge processing.")

elif page == "🤖 AI Detection":
    st.markdown('<div class="section-title">🕳️ AI Pothole Detection</div>', unsafe_allow_html=True)
    df = load_csv(CSV_FILES["smart_detection_results.csv"])

    refresh_evidence_from_disk()
    pothole_ev = st.session_state["evidence_memory"].get("smart_detections", [])

    if df.empty and not pothole_ev:
        st.info("Pothole result available nahi hai. Road Video par video upload karo.")
    else:
        c1, c2 = st.columns(2)
        c1.metric("🕳️ Pothole Detection Records", len(df))
        c2.metric("📸 Pothole Evidence Frames", len(pothole_ev))

        if not df.empty:
            st.dataframe(df, use_container_width=True)

        if pothole_ev:
            st.subheader("📸 Detected Pothole Evidence")
            cols = st.columns(4)
            for i, item in enumerate(pothole_ev):
                with cols[i % 4]:
                    st.image(item["data"], caption=item["name"], use_container_width=True)

elif page == "🚗 Traffic Intelligence":
    st.markdown('<div class="section-title">🚗 Traffic Intelligence</div>', unsafe_allow_html=True)
    df = load_csv(CSV_FILES["traffic_results.csv"])
    if df.empty: st.info("Traffic result available nahi hai. Road Video par video upload karo.")
    else:
        st.success(f"✅ {len(df)} traffic records")
        st.dataframe(df, use_container_width=True)

elif page == "🚦 Live Traffic System":
    st.markdown('<div class="section-title">🚦 Real-Time Smart Signal Dashboard</div>', unsafe_allow_html=True)
    st.info(
        "Local webcam ko AI engine continuously read karega. "
        "Browser me live annotated frames aur vehicle statistics update honge."
    )

    if LIVE_STOP_FILE.exists():
        try:
            LIVE_STOP_FILE.unlink()
        except Exception:
            pass

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🚀 Start Live AI Camera", use_container_width=True):
            # Clean old live output.
            try:
                LIVE_FRAME_PATH.unlink(missing_ok=True)
            except Exception:
                pass
            try:
                CSV_FILES["traffic_results.csv"].unlink(missing_ok=True)
            except Exception:
                pass
            try:
                LIVE_STOP_FILE.unlink(missing_ok=True)
            except Exception:
                pass

            if not LIVE_ENGINE.exists():
                st.error("❌ live_traffic_engine.py project folder me nahi mila.")
            else:
                proc = subprocess.Popen(
                    [sys.executable, str(LIVE_ENGINE)],
                    cwd=str(BASE_DIR),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                st.session_state["live_process_pid"] = proc.pid
                st.session_state["live_mode"] = True
                st.rerun()

    with col2:
        if st.button("🛑 Stop Live View", use_container_width=True):
            st.session_state["live_mode"] = False
            try:
                LIVE_STOP_FILE.write_text("stop", encoding="utf-8")
            except Exception:
                pass
            st.rerun()

    if st.session_state.get("live_mode"):
        if LIVE_FRAME_PATH.exists():
            try:
                frame = cv2.imread(str(LIVE_FRAME_PATH))
                if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(frame, caption="🔴 LIVE — AI Annotated Camera Feed", use_container_width=True)
            except Exception:
                pass
        else:
            st.warning(
                "⏳ Camera engine start ho raha hai... "
                "Agar 5–10 seconds me frame na aaye, ensure karo ki local webcam available hai."
            )

        df = load_csv(CSV_FILES["traffic_results.csv"])
        if not df.empty:
            latest = df.iloc[-1]
            raw_emergency = str(latest.get("Emergency", "")).strip().lower()
            is_emergency = raw_emergency in {"true", "1", "yes", "y", "emergency"}

            if is_emergency:
                st.error("🚨 EMERGENCY VEHICLE DETECTED — SIGNAL OVERRIDE ACTIVE (FORCE GREEN)")
            else:
                st.success("🟢 Live traffic monitoring active")

            traffic_level = str(latest.get("Traffic_Level", "Low"))
            live_vehicles = latest.get("Live_Vehicles_in_Frame", 0)

            TIMINGS = {
                "Low": {"Green": 15, "Red": 45},
                "Medium": {"Green": 30, "Red": 30},
                "High": {"Green": 60, "Red": 10},
            }
            if is_emergency:
                green_time, red_time = 999, 0
            else:
                green_time = TIMINGS.get(traffic_level, TIMINGS["Low"])["Green"]
                red_time = TIMINGS.get(traffic_level, TIMINGS["Low"])["Red"]

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Traffic Density", traffic_level)
            mc2.metric("Live Vehicles", int(float(live_vehicles or 0)))
            mc3.metric("🟢 Green Light", f"{green_time} sec")
            mc4.metric("🔴 Red Light", f"{red_time} sec")

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("🚗 Cars", int(float(latest.get("Cars", latest.get("Total_Unique_Cars", 0)) or 0)))
            s2.metric("🏍️ Bikes", int(float(latest.get("Bikes", latest.get("Total_Unique_Bikes", 0)) or 0)))
            s3.metric("🚌 Buses", int(float(latest.get("Buses", latest.get("Total_Unique_Buses", 0)) or 0)))
            s4.metric("🚚 Trucks", int(float(latest.get("Trucks", latest.get("Total_Unique_Trucks", 0)) or 0)))
        else:
            st.info("⏳ Waiting for live traffic CSV...")

        # Streamlit rerun creates a real browser-visible live stream instead of
        # a blocking while-loop that often shows only one frame.
        time.sleep(0.35)
        if st.session_state.get("live_mode"):
            st.rerun()

elif page == "🌊 Waterlogging Detection":
    st.markdown('<div class="section-title">🌊 Waterlogging Detection</div>', unsafe_allow_html=True)
    refresh_evidence_from_disk()

    df = load_csv(CSV_FILES["waterlogging_results.csv"])
    source_name = "waterlogging_results.csv"
    if df.empty:
        fallback = load_csv(CSV_FILES["waterlogging_incidents.csv"])
        if not fallback.empty:
            df = fallback
            source_name = "waterlogging_incidents.csv"

    water_evidence = st.session_state["evidence_memory"].get("waterlogging_detections", [])

    if not df.empty or water_evidence:
        c1, c2 = st.columns(2)
        c1.metric("🌊 Detection Records", len(df))
        c2.metric("📸 Evidence Frames", len(water_evidence))

        if not df.empty:
            st.success(f"✅ {len(df)} waterlogging records")
            st.caption(f"Source: {source_name}")
            st.dataframe(df, use_container_width=True)

            risk_col = _first_existing_column(df, ["Risk_Level", "risk_level", "Risk", "risk"])
            if risk_col:
                st.subheader("🌊 Risk Summary")
                st.dataframe(
                    df[risk_col].astype(str).value_counts().rename("Count").to_frame(),
                    use_container_width=True
                )

        if water_evidence:
            st.subheader("📸 Waterlogging Evidence")
            cols = st.columns(4)
            for i, item in enumerate(water_evidence):
                with cols[i % 4]:
                    st.image(item["data"], caption=item["name"], use_container_width=True)
    else:
        st.info("👆 Road Video par video upload karo.")

elif page == "🚶‍♂️ Pedestrian Safety":
    st.markdown('<div class="section-title">🚶‍♂️ Pedestrian Safety & Vulnerable Zones</div>', unsafe_allow_html=True)
    df = load_csv(CSV_FILES["pedestrian_results.csv"])
    if not df.empty:
        st.success(f"✅ {len(df)} pedestrian records found")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("👆 Road Video par video upload karo.")


elif page == "🔍 ANPR & Offenders":
    st.markdown('<div class="section-title">🔍 Automatic Number Plate Recognition & Offender Tracking</div>', unsafe_allow_html=True)
    df = load_csv(CSV_FILES["anpr_results.csv"])
    if not df.empty:
        st.success(f"✅ {len(df)} license plate records detected")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("👆 Road Video par video upload karo.")


elif page == "🚌 Bus Fleet Aggregation":
    st.markdown('<div class="section-title">🚌 Public Transport Bus Fleet Aggregation</div>', unsafe_allow_html=True)
    st.info("Transforming public transport buses into mobile urban sensing units for centralized fleet intelligence.")
    df = load_csv(CSV_FILES["fleet_summary.csv"])
    if not df.empty:
        st.success(f"✅ {len(df)} Active Fleet Units Reporting")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("👆 Road Video upload karke pipeline run karein.")


elif page == "📈 Route Delay & OD":
    st.markdown('<div class="section-title">📈 Route Delay & Origin-Destination (OD) Analytics</div>', unsafe_allow_html=True)
    st.info("Analyzing travel corridors, public transit delays, and Origin-Destination (OD) traffic volume matrices.")
    df = load_csv(CSV_FILES["od_delay_results.csv"])
    if not df.empty:
        st.success(f"✅ {len(df)} Corridors Analyzed for OD & Delays")
        st.dataframe(df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("⏱️ Current Delay Breakdown (Minutes)")
            st.bar_chart(df.set_index("Route_ID")["Current_Delay_Min"])
    else:
        st.info("👆 Pipeline run karne ke baad OD Analytics data yahan show hoga.")


elif page == "🗺️ GIS & Heatmap":
    st.markdown('<div class="section-title">🗺️ GIS Spatial Heatmap & Urban Hazard Hotspots</div>', unsafe_allow_html=True)
    st.info(
        "GIS map uses only GPS metadata actually extracted from the uploaded video. "
        "No demo/random coordinates are used."
    )

    incidents = load_csv(CSV_FILES["incidents.csv"])

    if incidents.empty:
        st.warning("⚠️ No incident records are available yet. Upload a road video and run analysis.")
    elif "Location_Source" not in incidents.columns:
        st.warning("⚠️ Incident records do not contain a verified GPS source.")
    else:
        map_df = incidents[
            incidents["Location_Source"].astype(str).str.upper() == "VIDEO_GPS_METADATA"
        ].copy()

        if not map_df.empty and {"Latitude", "Longitude"}.issubset(map_df.columns):
            map_df["Latitude"] = pd.to_numeric(map_df["Latitude"], errors="coerce")
            map_df["Longitude"] = pd.to_numeric(map_df["Longitude"], errors="coerce")
            map_df = map_df.dropna(subset=["Latitude", "Longitude"])

        if map_df.empty:
            st.warning(
                "⚠️ This video does not contain usable GPS metadata. "
                "Therefore the GIS map is intentionally not plotted."
            )
        else:
            st.success(f"✅ Rendering GIS map for {len(map_df)} GPS-linked incidents")

            lat_center = float(map_df["Latitude"].mean())
            lon_center = float(map_df["Longitude"].mean())

            view_state = pdk.ViewState(
                latitude=lat_center,
                longitude=lon_center,
                zoom=12,
                pitch=40
            )

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position="[Longitude, Latitude]",
                get_radius=100,
                get_fill_color=[255, 69, 0, 180],
                pickable=True,
                auto_highlight=True,
            )

            deck = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={
                    "text": "Incident: {incident_type}\n"
                            "Severity: {severity}\n"
                            "Lat: {Latitude}\n"
                            "Lon: {Longitude}"
                },
            )

            st.pydeck_chart(deck, use_container_width=True)
            st.dataframe(map_df, use_container_width=True)

elif page == "🚨 Incident Analysis":
    st.markdown('<div class="section-title">🚨 Incident Analysis</div>', unsafe_allow_html=True)
    df = load_csv(CSV_FILES["incidents.csv"])

    if not df.empty:
        if "Location_Source" in df.columns:
            df = df[df["Location_Source"].astype(str).str.upper() != "DEMO_SIMULATED"].copy()

        if df.empty:
            st.warning("⚠️ No real incidents are available for this video.")
        else:
            st.success(f"✅ {len(df)} real incidents detected")
            st.caption(
                "Incident records are generated from the current video's detector outputs. "
                "Demo/simulated incidents are not displayed."
            )
            st.dataframe(df, use_container_width=True)
    else:
        st.info("👆 Road Video par video upload karke analysis run karo.")

elif page == "📍 Location Intelligence":
    st.markdown('<div class="section-title">📍 Location Intelligence</div>', unsafe_allow_html=True)
    loc = st.session_state.get("video_location")
    if loc:
        st.success(f'📍 Actual Video GPS: {loc["latitude"]:.6f}, {loc["longitude"]:.6f}')
        st.caption(loc.get("label","GPS location"))
        md = pd.DataFrame({"latitude":[loc["latitude"]],"longitude":[loc["longitude"]]})
        st.map(md, latitude="latitude", longitude="longitude", zoom=11)
    else: st.warning("⚠️ Actual GPS metadata could not be read from this video.")


elif page == "📸 Evidence":
    st.markdown('<div class="section-title">📸 AI Detection Evidence</div>', unsafe_allow_html=True)

    # Always refresh from disk so evidence does not depend on a single Streamlit session.
    refresh_evidence_from_disk()
    ev = st.session_state.get("evidence_memory", {})

    ordered_categories = [
        "smart_detections",
        "traffic_detections",
        "waterlogging_detections",
        "pedestrian_detections",
        "anpr_detections",
        "all_detections",
        "edge_detections",
    ]

    total = sum(len(ev.get(cat, [])) for cat in ordered_categories)

    if total == 0:
        st.info(
            "Evidence available nahi hai. Road Video upload karke analysis run karo. "
            "Detector ko evidence generate karne par yahan images automatically appear hongi."
        )
    else:
        st.success(f"✅ {total} unique evidence images available")

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for cat in ordered_categories:
                for item in ev.get(cat, []):
                    z.writestr(f"{cat}/{item['name']}", item["data"])

        st.download_button(
            "📦 Download All Evidence (ZIP)",
            buf.getvalue(),
            "urban_intelligence_evidence.zip",
            "application/zip",
            use_container_width=True,
        )

        for cat in ordered_categories:
            items = ev.get(cat, [])

            label = EVIDENCE_CATEGORY_LABELS.get(
                cat, cat.replace("_", " ").title()
            )
            st.subheader(f"{label} ({len(items)})")

            if not items:
                st.caption(
                    "No evidence image was generated by this detector for the current video."
                )
                continue

            cols = st.columns(4)
            for i, item in enumerate(items):
                with cols[i % 4]:
                    st.image(
                        item["data"],
                        caption=item["name"],
                        use_container_width=True
                    )

elif page == "📊 Reports":
    st.markdown('<div class="section-title">📊 Detection Reports</div>', unsafe_allow_html=True)
    for title, path in [
        ("🕳️ Pothole", CSV_FILES["smart_detection_results.csv"]), 
        ("🚗 Traffic", CSV_FILES["traffic_results.csv"]), 
        ("🌊 Waterlogging", CSV_FILES["waterlogging_results.csv"]), 
        ("🚶‍♂️ Pedestrian", CSV_FILES["pedestrian_results.csv"]), 
        ("🔍 ANPR & Offenders", CSV_FILES["anpr_results.csv"]),
        ("🚌 Fleet Summary", CSV_FILES["fleet_summary.csv"]),
        ("📈 Route Delay & OD", CSV_FILES["od_delay_results.csv"]),
        ("🚨 Incidents", CSV_FILES["incidents.csv"])
    ]:
        df = load_csv(path)
        with st.expander(f"{title} ({len(df)} records)", expanded=not df.empty):
            if df.empty: st.info("No report available.")
            else:
                st.dataframe(df, use_container_width=True)
                st.download_button(f"⬇️ Download {title} CSV", df.to_csv(index=False).encode("utf-8"), f"{path.stem}.csv", "text/csv", key=f"csv_{path.stem}")

st.divider()
st.markdown("<center><b>SIH26124 – AI-Powered Urban Intelligence Platform Using Public Transport Fleet</b><br>Prototype developed for Smart India Hackathon • Real-data only • No simulated GPS • Local live camera supported</center>", unsafe_allow_html=True)