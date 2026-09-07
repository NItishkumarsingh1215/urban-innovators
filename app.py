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

LIVE_FRAME_PATH = DATA_DIR / "live_frame.jpg"

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
    "live_mode": False
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def images():
    if not EVIDENCE_DIR.exists():
        return []
    return sorted(
        [p for p in EVIDENCE_DIR.rglob("*")
         if p.is_file() and p.suffix.lower() in {".jpg",".jpeg",".png",".webp"}],
        key=lambda p: str(p).lower()
    )


def evidence_snapshot():
    out = {}
    for p in images():
        try:
            s = p.stat()
            out[str(p.resolve())] = (s.st_mtime_ns, s.st_size)
        except Exception:
            pass
    return out


def capture_new_evidence(category, before):
    bucket = st.session_state["evidence_memory"].setdefault(category, [])
    names = {x["name"] for x in bucket}
    for p in images():
        key = str(p.resolve())
        try:
            s = p.stat()
            state = (s.st_mtime_ns, s.st_size)
        except Exception:
            continue
        if before.get(key) == state:
            continue
        try:
            data = p.read_bytes()
        except Exception:
            continue
        if p.name not in names:
            bucket.append({"name": p.name, "data": data})
            names.add(p.name)


def keep_existing_evidence():
    for p in images():
        category = p.parent.name or "all_detections"
        bucket = st.session_state["evidence_memory"].setdefault(category, [])
        if p.name in {x["name"] for x in bucket}:
            continue
        try:
            bucket.append({"name": p.name, "data": p.read_bytes()})
        except Exception:
            pass


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


def apply_location():
    loc = st.session_state.get("video_location")
    if not loc: return
    for p in [CSV_FILES["pothole_incidents.csv"], CSV_FILES["waterlogging_incidents.csv"], CSV_FILES["incidents.csv"]]:
        if not p.exists(): continue
        try:
            df = pd.read_csv(p)
            df["Latitude"] = loc["latitude"]
            df["Longitude"] = loc["longitude"]
            df["Location_Source"] = loc["source"]
            df["Location"] = loc.get("label", f'{loc["latitude"]:.6f}, {loc["longitude"]:.6f}')
            df.to_csv(p, index=False)
        except Exception: pass


def run_pipeline(source, progress):
    results = {}
    
    # 1. Pothole Detection
    progress.info("⏳ 🕳️ Pothole Detection...")
    before = evidence_snapshot()
    ok, msg = run_detector(SMART, source)
    capture_new_evidence("smart_detections", before)
    results["🕳️ Pothole Detection"] = {"success": ok, "message": msg}
    
    # 2. Traffic Detection
    progress.info("⏳ 🚗 Traffic Detection...")
    before = evidence_snapshot()
    ok, msg = run_detector(TRAFFIC, source)
    capture_new_evidence("traffic_detections", before)
    results["🚗 Traffic Detection"] = {"success": ok, "message": msg}
    
    # 3. Waterlogging Detection
    progress.info("⏳ 🌊 Waterlogging Detection...")
    before = evidence_snapshot()
    try:
        if str(AI_DIR) not in sys.path:
            sys.path.append(str(AI_DIR))
        import waterlogging_detector
        
        bridge_path = detector_bridge(source)
        res = waterlogging_detector.analyze_video(
            video_path=bridge_path,
            output_csv=str(DATA_DIR / "waterlogging_results.csv"),
            evidence_dir=str(EVIDENCE_DIR / "waterlogging_detections")
        )
        ok = res.get("success", False)
        msg = res.get("message", "Completed")
    except Exception as e:
        ok = False
        msg = str(e)
        
    capture_new_evidence("waterlogging_detections", before)
    results["🌊 Waterlogging Detection"] = {"success": ok, "message": msg}

    # 4. Pedestrian Safety Detection
    progress.info("⏳ 🚶‍♂️ Pedestrian Safety Detection...")
    before = evidence_snapshot()
    try:
        if str(AI_DIR) not in sys.path:
            sys.path.append(str(AI_DIR))
        import pedestrian_detector
        
        bridge_path = detector_bridge(source)
        res = pedestrian_detector.analyze_pedestrians(
            video_path=bridge_path,
            output_csv=str(DATA_DIR / "pedestrian_results.csv"),
            evidence_dir=str(EVIDENCE_DIR / "pedestrian_detections")
        )
        ok = res.get("success", False)
        msg = res.get("message", "Completed")
    except Exception as e:
        ok = False
        msg = str(e)
        
    capture_new_evidence("pedestrian_detections", before)
    results["🚶‍♂️ Pedestrian Safety"] = {"success": ok, "message": msg}

    # 5. ANPR & Offender Tracking
    progress.info("⏳ 🔍 ANPR & Offender Tracking...")
    before = evidence_snapshot()
    try:
        if str(AI_DIR) not in sys.path:
            sys.path.append(str(AI_DIR))
        import anpr_detector
        
        bridge_path = detector_bridge(source)
        res = anpr_detector.analyze_anpr(
            video_path=bridge_path,
            output_csv=str(DATA_DIR / "anpr_results.csv"),
            evidence_dir=str(EVIDENCE_DIR / "anpr_detections")
        )
        ok = res.get("success", False)
        msg = res.get("message", "Completed")
    except Exception as e:
        ok = False
        msg = str(e)
        
    capture_new_evidence("anpr_detections", before)
    results["🔍 ANPR & Offenders"] = {"success": ok, "message": msg}

    # 6. Bus Fleet Aggregation
    progress.info("⏳ 🚌 Bus Fleet Aggregation...")
    try:
        if str(AI_DIR) not in sys.path:
            sys.path.append(str(AI_DIR))
        import fleet_aggregator
        res = fleet_aggregator.aggregate_fleet_data(output_csv=str(DATA_DIR / "fleet_summary.csv"))
        ok = res.get("success", False)
        msg = res.get("message", "Completed")
    except Exception as e:
        ok = False
        msg = str(e)
    results["🚌 Fleet Aggregation"] = {"success": ok, "message": msg}

    # 7. Route Delay & OD Analytics
    progress.info("⏳ 📈 Route Delay & OD Analytics...")
    try:
        if str(AI_DIR) not in sys.path:
            sys.path.append(str(AI_DIR))
        import od_analytics
        res = od_analytics.analyze_od_and_delays(output_csv=str(DATA_DIR / "od_delay_results.csv"))
        ok = res.get("success", False)
        msg = res.get("message", "Completed")
    except Exception as e:
        ok = False
        msg = str(e)
    results["📈 Route Delay & OD"] = {"success": ok, "message": msg}

    # 8. Incident Generation
    progress.info("⏳ 🚨 Incident Generation...")
    before = evidence_snapshot()
    ok, msg = run_detector(INCIDENT, source)
    capture_new_evidence("all_detections", before)
    results["🚨 Incident Generation"] = {"success": ok, "message": msg}

    apply_location()
    collect_csvs()
    keep_existing_evidence()
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
                with st.spinner("Pothole → Traffic → Waterlogging → Pedestrian → ANPR → Fleet → OD Analytics → Incidents..."):
                    run_pipeline(src, progress)
                st.success("🎉 Complete analysis finished.")
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
    else: st.info("👆 Video upload karo. Upload ke baad automatic complete pipeline chalega.")

elif page == "🤖 AI Detection":
    st.markdown('<div class="section-title">🕳️ AI Pothole Detection</div>', unsafe_allow_html=True)
    df = load_csv(CSV_FILES["smart_detection_results.csv"])
    if df.empty: st.info("Pothole result available nahi hai. Road Video par video upload karo.")
    else: st.dataframe(df, width="stretch")

elif page == "🚗 Traffic Intelligence":
    st.markdown('<div class="section-title">🚗 Traffic Intelligence</div>', unsafe_allow_html=True)
    df = load_csv(CSV_FILES["traffic_results.csv"])
    if df.empty: st.info("Traffic result available nahi hai. Road Video par video upload karo.")
    else:
        st.success(f"✅ {len(df)} traffic records")
        st.dataframe(df, width="stretch")

elif page == "🚦 Live Traffic System":
    st.markdown('<div class="section-title">🚦 Real-Time Smart Signal Dashboard</div>', unsafe_allow_html=True)
    st.info("Start the AI engine in the background to monitor live traffic and auto-adjust signal timings.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Start Live AI Camera", use_container_width=True):
            st.session_state["live_mode"] = True
            subprocess.Popen([sys.executable, str(TRAFFIC)], cwd=str(BASE_DIR))
            st.rerun()
            
    with col2:
        if st.button("🛑 Stop Live View", use_container_width=True):
            st.session_state["live_mode"] = False
            st.rerun()

    if st.session_state.get("live_mode"):
        alert_placeholder = st.empty()
        metrics_placeholder = st.empty()
        video_col, stats_col = st.columns([2, 1])

        with video_col:
            st.subheader("🎥 Live AI Camera Feed")
            video_placeholder = st.empty()
            
        with stats_col:
            st.subheader("📊 Cumulative Vehicle Data")
            stats_placeholder = st.empty()

        TIMINGS = {"Low": {"Green": 15, "Red": 45}, "Medium": {"Green": 30, "Red": 30}, "High": {"Green": 60, "Red": 10}}

        while st.session_state.get("live_mode"):
            if LIVE_FRAME_PATH.exists():
                try:
                    frame = cv2.imread(str(LIVE_FRAME_PATH))
                    if frame is not None:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(frame, use_container_width=True)
                except: pass 

            if CSV_FILES["traffic_results.csv"].exists():
                try:
                    df = pd.read_csv(CSV_FILES["traffic_results.csv"])
                    if not df.empty:
                        latest = df.iloc[-1]
                        
                        is_emergency = latest.get("Emergency", False)
                        if is_emergency:
                            alert_placeholder.error("🚨 **EMERGENCY VEHICLE DETECTED! SIGNAL OVERRIDE ACTIVE (FORCE GREEN)** 🚨")
                        else:
                            alert_placeholder.empty()
                        
                        traffic_level = latest.get("Traffic_Level", "Low")
                        live_vehicles = latest.get("Live_Vehicles_in_Frame", 0)
                        
                        if is_emergency:
                            green_time, red_time = 999, 0
                        else:
                            green_time = TIMINGS.get(traffic_level, {}).get("Green", 20)
                            red_time = TIMINGS.get(traffic_level, {}).get("Red", 40)

                        with metrics_placeholder.container():
                            mc1, mc2, mc3, mc4 = st.columns(4)
                            mc1.metric("Traffic Density", traffic_level)
                            mc2.metric("Live Vehicles", live_vehicles)
                            mc3.metric("🟢 Green Light", f"{green_time} sec")
                            mc4.metric("🔴 Red Light", f"{red_time} sec")
                        
                        with stats_placeholder.container():
                            st.markdown(f"**🚗 Cars:** {latest.get('Total_Unique_Cars', 0)}")
                            st.markdown(f"**🏍️ Bikes:** {latest.get('Total_Unique_Bikes', 0)}")
                            st.markdown(f"**🚌 Buses:** {latest.get('Total_Unique_Buses', 0)}")
                            st.markdown(f"**🚚 Trucks:** {latest.get('Total_Unique_Trucks', 0)}")
                except: pass 
                
            time.sleep(0.05)


elif page == "🌊 Waterlogging Detection":
    st.markdown('<div class="section-title">🌊 Waterlogging Detection</div>', unsafe_allow_html=True)
    df = load_csv(CSV_FILES["waterlogging_results.csv"])
    source_name = "waterlogging_results.csv"
    if df.empty:
        fallback = load_csv(CSV_FILES["waterlogging_incidents.csv"])
        if not fallback.empty:
            df = fallback; source_name = "waterlogging_incidents.csv"
    water_evidence = st.session_state["evidence_memory"].get("waterlogging_detections", [])
    if not df.empty:
        st.success(f"✅ {len(df)} waterlogging records"); st.caption(f"Source: {source_name}"); st.dataframe(df, width="stretch")
        if "Risk_Level" in df.columns:
            st.subheader("🌊 Risk Summary")
            st.dataframe(df["Risk_Level"].astype(str).value_counts().rename("Count").to_frame(), width="stretch")
    elif water_evidence:
        st.success(f"✅ Waterlogging AI evidence detected ({len(water_evidence)} images)")
        cols = st.columns(4)
        for i, item in enumerate(water_evidence):
            with cols[i % 4]: st.image(item["data"], caption=item["name"], width="stretch")
    else: st.info("👆 Road Video par video upload karo.")


elif page == "🚶‍♂️ Pedestrian Safety":
    st.markdown('<div class="section-title">🚶‍♂️ Pedestrian Safety & Vulnerable Zones</div>', unsafe_allow_html=True)
    df = load_csv(CSV_FILES["pedestrian_results.csv"])
    if not df.empty:
        st.success(f"✅ {len(df)} pedestrian records found")
        st.dataframe(df, width="stretch")
    else:
        st.info("👆 Road Video par video upload karo.")


elif page == "🔍 ANPR & Offenders":
    st.markdown('<div class="section-title">🔍 Automatic Number Plate Recognition & Offender Tracking</div>', unsafe_allow_html=True)
    df = load_csv(CSV_FILES["anpr_results.csv"])
    if not df.empty:
        st.success(f"✅ {len(df)} license plate records detected")
        st.dataframe(df, width="stretch")
    else:
        st.info("👆 Road Video par video upload karo.")


elif page == "🚌 Bus Fleet Aggregation":
    st.markdown('<div class="section-title">🚌 Public Transport Bus Fleet Aggregation</div>', unsafe_allow_html=True)
    st.info("Transforming public transport buses into mobile urban sensing units for centralized fleet intelligence.")
    df = load_csv(CSV_FILES["fleet_summary.csv"])
    if not df.empty:
        st.success(f"✅ {len(df)} Active Fleet Units Reporting")
        st.dataframe(df, width="stretch")
    else:
        st.info("👆 Road Video upload karke pipeline run karein.")


elif page == "📈 Route Delay & OD":
    st.markdown('<div class="section-title">📈 Route Delay & Origin-Destination (OD) Analytics</div>', unsafe_allow_html=True)
    st.info("Analyzing travel corridors, public transit delays, and Origin-Destination (OD) traffic volume matrices.")
    df = load_csv(CSV_FILES["od_delay_results.csv"])
    if not df.empty:
        st.success(f"✅ {len(df)} Corridors Analyzed for OD & Delays")
        st.dataframe(df, width="stretch")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("⏱️ Current Delay Breakdown (Minutes)")
            st.bar_chart(df.set_index("Route_ID")["Current_Delay_Min"])
    else:
        st.info("👆 Pipeline run karne ke baad OD Analytics data yahan show hoga.")


elif page == "🗺️ GIS & Heatmap":
    st.markdown('<div class="section-title">🗺️ GIS Spatial Heatmap & Urban Hazard Hotspots</div>', unsafe_allow_html=True)
    st.info("Interactive 3D PyDeck GIS spatial heatmap displaying concentrated urban road anomalies and hazard zones.")
    
    incidents = load_csv(CSV_FILES["incidents.csv"])
    if not incidents.empty and {"Latitude", "Longitude"}.issubset(incidents.columns):
        map_df = incidents.copy()
        map_df["Latitude"] = pd.to_numeric(map_df["Latitude"], errors="coerce")
        map_df["Longitude"] = pd.to_numeric(map_df["Longitude"], errors="coerce")
        map_df = map_df.dropna(subset=["Latitude", "Longitude"])
        
        if not map_df.empty:
            st.success(f"✅ Rendering GIS Spatial Map for {len(map_df)} recorded urban incidents")
            
            lat_center = map_df["Latitude"].mean()
            lon_center = map_df["Longitude"].mean()
            
            view_state = pdk.ViewState(latitude=lat_center, longitude=lon_center, zoom=12, pitch=40)
            
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position='[Longitude, Latitude]',
                get_color='[255, 69, 0, 180]',
                get_radius=120,
                pickable=True,
                auto_highlight=True
            )
            
            r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "Incident / Hazard Location\nLat: {Latitude}\nLon: {Longitude}"})
            st.pydeck_chart(r)
            st.dataframe(map_df, width="stretch")
        else:
            st.warning("⚠️ Incident data mein valid Latitude/Longitude coordinates available nahi hain.")
    else:
        st.warning("⚠️ GIS Heatmap ke liye koi incident data available nahi hai. Pehle road video upload karke analysis run karein.")


elif page == "🚨 Incident Analysis":
    st.markdown('<div class="section-title">🚨 Incident Analysis</div>', unsafe_allow_html=True)
    df = load_csv(CSV_FILES["incidents.csv"])
    if df.empty: st.info("Incident result available nahi hai. Road Video analysis complete karo.")
    else: st.success(f"✅ {len(df)} incidents"); st.dataframe(df, width="stretch")


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
    st.markdown('<div class="section-title">📸 Live AI Evidence</div>', unsafe_allow_html=True)
    ev = st.session_state["evidence_memory"]
    total = sum(len(v) for v in ev.values())
    if total == 0: st.info("Evidence available nahi hai.")
    else:
        st.success(f"✅ {total} evidence images Streamlit session RAM mein hain.")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for cat, items in ev.items():
                for item in items: z.writestr(f"{cat}/{item['name']}", item["data"])
        st.download_button("📦 Download All Evidence (ZIP)", buf.getvalue(), "urban_intelligence_evidence.zip", "application/zip", width="stretch")
        for cat, items in ev.items():
            if not items: continue
            st.subheader(f"📸 {cat.replace('_',' ').title()} ({len(items)})")
            cols = st.columns(4)
            for i, item in enumerate(items):
                with cols[i % 4]:
                    st.image(item["data"], caption=item["name"], width="stretch")


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
                st.dataframe(df, width="stretch")
                st.download_button(f"⬇️ Download {title} CSV", df.to_csv(index=False).encode("utf-8"), f"{path.stem}.csv", "text/csv", key=f"csv_{path.stem}")

st.divider()
st.markdown("<center><b>SIH26124 – AI-Powered Urban Intelligence Platform Using Public Transport Fleet</b><br>Prototype developed for Smart India Hackathon</center>", unsafe_allow_html=True)