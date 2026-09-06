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
import urllib.parse
import urllib.request
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
INCIDENT = AI_DIR / "incident_generator.py"

CSV_FILES = {
    "smart_detection_results.csv": DATA_DIR / "smart_detection_results.csv",
    "traffic_results.csv": DATA_DIR / "traffic_results.csv",
    "waterlogging_results.csv": DATA_DIR / "waterlogging_results.csv",
    "pothole_incidents.csv": DATA_DIR / "pothole_incidents.csv",
    "waterlogging_incidents.csv": DATA_DIR / "waterlogging_incidents.csv",
    "incidents.csv": DATA_DIR / "incidents.csv",
}

for d in [UPLOAD_DIR, DATA_DIR, EVIDENCE_DIR, AI_DIR]:
    d.mkdir(parents=True, exist_ok=True)

defaults = {
    "active_video_bytes": None,
    "active_video_name": None,
    "active_video_mime": "video/mp4",
    "active_video_signature": None,
    "video_location": None,
    "analysis_results": {},
    "evidence_memory": {},
    "analysis_complete": False,
    "pipeline_running": False,
    "pipeline_results": {},
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
        try:
            p.unlink()
        except Exception:
            pass


def keep_existing_evidence():
    # Copy existing evidence to session RAM, but never delete it.
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
    x = shutil.which("ffprobe")
    if x:
        return x
    roots = [
        Path.home()/ "AppData/Local/Microsoft/WinGet/Packages",
        Path("C:/Program Files"),
        Path("C:/Program Files (x86)")
    ]
    for root in roots:
        if not root.exists():
            continue
        try:
            for p in root.rglob("ffprobe.exe"):
                return str(p)
        except Exception:
            pass
    return None


def parse_location(value):
    m = re.search(r"([+-])(\d{1,3}(?:\.\d+)?)([+-])(\d{1,3}(?:\.\d+)?)", str(value))
    if not m:
        return None
    lat = float(m.group(1)+m.group(2))
    lon = float(m.group(3)+m.group(4))
    if -90 <= lat <= 90 and -180 <= lon <= 180:
        return {"latitude": lat, "longitude": lon, "source": "VIDEO_GPS_METADATA"}
    return None


def extract_gps(video):
    ffprobe = find_ffprobe()
    if not ffprobe:
        return None
    cmd = [ffprobe, "-v", "quiet", "-show_entries",
           "format_tags=location,location-eng:stream_tags=location,location-eng",
           "-of", "default=noprint_wrappers=1:nokey=0", str(video)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        for line in r.stdout.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                if key.lower() in {"location", "location-eng"}:
                    loc = parse_location(value)
                    if loc:
                        return loc
    except Exception:
        pass
    return None


def reverse_geocode(lat, lon):
    try:
        q = urllib.parse.urlencode({"lat":lat,"lon":lon,"format":"jsonv2","zoom":18})
        req = urllib.request.Request(
            "https://nominatim.openstreetmap.org/reverse?"+q,
            headers={"User-Agent":"SIH26124-Urban-Intelligence/1.0"}
        )
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read().decode())
        a = data.get("address", {})
        vals = []
        for k in ["road","neighbourhood","suburb","city","town","village","state"]:
            if a.get(k) and a[k] not in vals:
                vals.append(a[k])
        return ", ".join(vals[:4]) or data.get("display_name")
    except Exception:
        return None


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
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass


def run_detector(script, source):
    if not script.exists():
        return False, f"{script.name} not found"
    before = evidence_snapshot()
    try:
        detector_bridge(source)
        r = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=900
        )
        msg = r.stdout.strip() or r.stderr.strip()
        return r.returncode == 0, msg
    except Exception as e:
        return False, str(e)
    finally:
        cleanup_bridge()


def collect_csvs():
    for name, path in CSV_FILES.items():
        if path.exists():
            try:
                st.session_state["analysis_results"][name] = pd.read_csv(path)
            except Exception:
                pass


def load_csv(path):
    name = Path(path).name
    if name in st.session_state["analysis_results"]:
        return st.session_state["analysis_results"][name].copy()
    if Path(path).exists():
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    return pd.DataFrame()


def apply_location():
    loc = st.session_state.get("video_location")
    if not loc:
        return
    for p in [CSV_FILES["pothole_incidents.csv"],
              CSV_FILES["waterlogging_incidents.csv"],
              CSV_FILES["incidents.csv"]]:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
            df["Latitude"] = loc["latitude"]
            df["Longitude"] = loc["longitude"]
            df["Location_Source"] = loc["source"]
            df["Location"] = loc.get("label", f'{loc["latitude"]:.6f}, {loc["longitude"]:.6f}')
            df.to_csv(p, index=False)
        except Exception:
            pass


def run_pipeline(source, progress):
    results = {}
    jobs = [
        ("🕳️ Pothole Detection", SMART, "smart_detections"),
        ("🚗 Traffic Detection", TRAFFIC, "traffic_detections"),
        ("🌊 Waterlogging Detection", WATER, "waterlogging_detections"),
        ("🚨 Incident Generation", INCIDENT, "all_detections"),
    ]
    for title, script, category in jobs:
        progress.info(f"⏳ {title}...")
        before = evidence_snapshot()
        ok, msg = run_detector(script, source)
        capture_new_evidence(category, before)
        results[title] = {"success":ok, "message":msg}
        (progress.success if ok else progress.error)(
            f'{"✅" if ok else "❌"} {title} {"completed." if ok else "failed; continuing."}'
        )
    apply_location()
    collect_csvs()
    keep_existing_evidence()
    st.session_state["pipeline_results"] = results
    st.session_state["analysis_complete"] = True
    st.session_state["pipeline_running"] = False


def video_info(data):
    if not data:
        return None
    p = temp_source(data, "video.mp4")
    try:
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            return None
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return {"frames":frames,"fps":fps,"width":w,"height":h,
                "duration":frames/fps if fps else 0}
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
    '<div class="subtitle">Smart Pothole • Traffic • Waterlogging • Location • Incidents • Evidence</div>',
    unsafe_allow_html=True
)

st.sidebar.title("🛠️ Navigation")
page = st.sidebar.radio("Select Module", [
    "🏠 Dashboard","🎥 Road Video","🤖 AI Detection",
    "🚗 Traffic Intelligence","🌊 Waterlogging Detection",
    "🚨 Incident Analysis","📍 Location Intelligence",
    "📸 Evidence","📊 Reports"
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
    incidents = load_csv(CSV_FILES["incidents.csv"])
    ev = st.session_state["evidence_memory"]
    a,b,c,d = st.columns(4)
    a.metric("🎥 Video", "Available" if st.session_state["active_video_bytes"] else "Not Uploaded")
    b.metric("🕳️ Pothole Records", len(smart))
    c.metric("🚗 Traffic Records", len(traffic))
    d.metric("🌊 Waterlogging Records", len(water))
    e,f,g,h = st.columns(4)
    e.metric("🚨 Incidents", len(incidents))
    f.metric("🕳️ Pothole Evidence", len(ev.get("smart_detections",[])))
    g.metric("🚗 Traffic Evidence", len(ev.get("traffic_detections",[])))
    h.metric("🌊 Water Evidence", len(ev.get("waterlogging_detections",[])))
    loc = st.session_state.get("video_location")
    if loc:
        st.success(f'📍 {loc["latitude"]:.6f}, {loc["longitude"]:.6f} — {loc.get("label","GPS")}')
    elif st.session_state["active_video_bytes"]:
        st.warning("No GPS metadata found. Random location is NOT used.")


elif page == "🎥 Road Video":
    st.markdown('<div class="section-title">Road Video Input</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload Original Road Video",
        type=["mp4","avi","mov","mkv","m4v"], key="road_video")
    if uploaded is not None:
        data = uploaded.getvalue()
        sig = signature(uploaded.name, data)
        if sig != st.session_state["active_video_signature"]:
            st.session_state["active_video_bytes"] = data
            st.session_state["active_video_name"] = uploaded.name
            st.session_state["active_video_mime"] = uploaded.type or "video/mp4"
            st.session_state["active_video_signature"] = sig
            st.session_state["analysis_results"] = {}
            st.session_state["evidence_memory"] = {}
            st.session_state["pipeline_results"] = {}
            st.session_state["analysis_complete"] = False
            src = temp_source(data, uploaded.name)
            try:
                loc = extract_gps(src)
                if loc:
                    loc["label"] = reverse_geocode(loc["latitude"], loc["longitude"]) or "GPS location"
                    st.session_state["video_location"] = loc
                    st.success(f'📍 GPS found: {loc["latitude"]:.6f}, {loc["longitude"]:.6f}')
                else:
                    st.session_state["video_location"] = None
                    st.warning("⚠️ Video GPS metadata nahi mila.")
                progress = st.empty()
                with st.spinner("Pothole → Traffic → Waterlogging → Incidents..."):
                    run_pipeline(src, progress)
                st.success("🎉 Complete analysis finished.")
            except Exception as e:
                st.error(f"Pipeline error: {e}")
            finally:
                try: src.unlink()
                except Exception: pass
                cleanup_bridge()
    if st.session_state["active_video_bytes"]:
        st.subheader(f'🎬 {st.session_state.get("active_video_name","Current Video")}')
        st.video(st.session_state["active_video_bytes"],
                 format=st.session_state.get("active_video_mime","video/mp4"))
        info = video_info(st.session_state["active_video_bytes"])
        if info:
            a,b,c,d = st.columns(4)
            a.metric("Frames", info["frames"]); b.metric("FPS", f'{info["fps"]:.2f}')
            c.metric("Resolution", f'{info["width"]} × {info["height"]}')
            d.metric("Duration", f'{info["duration"]:.1f} sec')
    else:
        st.info("👆 Video upload karo. Upload ke baad automatic complete pipeline chalega.")


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


elif page == "🌊 Waterlogging Detection":
    st.markdown('<div class="section-title">🌊 Waterlogging Detection</div>', unsafe_allow_html=True)
    df = load_csv(CSV_FILES["waterlogging_results.csv"])
    if df.empty: st.info("Waterlogging result available nahi hai. Road Video par video upload karo.")
    else:
        st.success(f"✅ {len(df)} waterlogging records")
        st.dataframe(df, width="stretch")
        if "Risk_Level" in df.columns:
            st.subheader("🌊 Risk Summary")
            st.dataframe(df["Risk_Level"].astype(str).value_counts().rename("Count").to_frame(), width="stretch")


elif page == "🚨 Incident Analysis":
    st.markdown('<div class="section-title">🚨 Incident Analysis</div>', unsafe_allow_html=True)
    df = load_csv(CSV_FILES["incidents.csv"])
    if df.empty: st.info("Incident result available nahi hai. Road Video analysis complete karo.")
    else:
        st.success(f"✅ {len(df)} incidents")
        st.dataframe(df, width="stretch")


elif page == "📍 Location Intelligence":
    st.markdown('<div class="section-title">📍 Location Intelligence</div>', unsafe_allow_html=True)
    loc = st.session_state.get("video_location")
    if loc:
        st.success(f'📍 Actual Video GPS: {loc["latitude"]:.6f}, {loc["longitude"]:.6f}')
        st.caption(loc.get("label","GPS location"))
        md = pd.DataFrame({"latitude":[loc["latitude"]],"longitude":[loc["longitude"]]})
        st.map(md, latitude="latitude", longitude="longitude", zoom=11)
    else:
        st.warning("No actual GPS metadata found in this video.")
    incidents = load_csv(CSV_FILES["incidents.csv"])
    if not incidents.empty and {"Latitude","Longitude"}.issubset(incidents.columns):
        x = incidents.copy()
        x["Latitude"] = pd.to_numeric(x["Latitude"], errors="coerce")
        x["Longitude"] = pd.to_numeric(x["Longitude"], errors="coerce")
        x = x.dropna(subset=["Latitude","Longitude"])
        if not x.empty:
            st.subheader("🚨 Incident Map")
            md = x[["Latitude","Longitude"]].rename(columns={"Latitude":"latitude","Longitude":"longitude"})
            st.map(md, latitude="latitude", longitude="longitude", zoom=11)
            st.dataframe(x, width="stretch")


elif page == "📸 Evidence":
    st.markdown('<div class="section-title">📸 Live AI Evidence</div>', unsafe_allow_html=True)
    ev = st.session_state["evidence_memory"]
    total = sum(len(v) for v in ev.values())
    if total == 0:
        st.info("Evidence available nahi hai. Road Video par video upload karo.")
    else:
        st.success(f"✅ {total} evidence images Streamlit session RAM mein hain.")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for cat, items in ev.items():
                for item in items:
                    z.writestr(f"{cat}/{item['name']}", item["data"])
        st.download_button("📦 Download All Evidence (ZIP)", buf.getvalue(),
                           "urban_intelligence_evidence.zip", "application/zip",
                           width="stretch")
        for cat, items in ev.items():
            if not items: continue
            st.subheader(f"📸 {cat.replace('_',' ').title()} ({len(items)})")
            cols = st.columns(4)
            for i, item in enumerate(items):
                with cols[i % 4]:
                    st.image(item["data"], caption=item["name"], width="stretch")
                    st.download_button("⬇️ Download", item["data"], item["name"],
                                       "image/jpeg", key=f"ev_{cat}_{i}", width="stretch")


elif page == "📊 Reports":
    st.markdown('<div class="section-title">📊 Detection Reports</div>', unsafe_allow_html=True)
    for title, path in [
        ("🕳️ Pothole", CSV_FILES["smart_detection_results.csv"]),
        ("🚗 Traffic", CSV_FILES["traffic_results.csv"]),
        ("🌊 Waterlogging", CSV_FILES["waterlogging_results.csv"]),
        ("🚨 Incidents", CSV_FILES["incidents.csv"]),
    ]:
        df = load_csv(path)
        with st.expander(f"{title} ({len(df)} records)", expanded=not df.empty):
            if df.empty:
                st.info("No report available.")
            else:
                st.dataframe(df, width="stretch")
                st.download_button(f"⬇️ Download {title} CSV",
                    df.to_csv(index=False).encode("utf-8"),
                    f"{path.stem}.csv", "text/csv", key=f"csv_{path.stem}")

st.divider()
st.markdown("<center><b>SIH26124 – AI-Powered Urban Intelligence Platform</b><br>Prototype developed for Smart India Hackathon</center>", unsafe_allow_html=True)
