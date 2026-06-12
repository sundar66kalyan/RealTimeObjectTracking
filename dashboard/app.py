"""
dashboard/app.py
----------------
Real-Time Intelligent Object Tracking and Video Analytics System
Created by: KalyanaSundar — AI Engineer | Computer Vision Engineer | Data Scientist

Run:
    streamlit run dashboard/app.py
"""

import os
import sys
import time
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Traffic Analytics System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { padding: 0 2rem 2rem 2rem; max-width: 1440px; }

    .project-header {
        background: linear-gradient(135deg, #0a0f1e 0%, #0f172a 50%, #0a1628 100%);
        border-bottom: 1px solid #1e3a5f;
        padding: 1.4rem 2rem 1.2rem 2rem;
        margin: -1rem -2rem 1.5rem -2rem;
    }
    .project-title { font-size: 1.45rem; font-weight: 700; color: #e2e8f0; letter-spacing: -0.02em; margin: 0 0 4px 0; }
    .project-subtitle { font-size: 0.8rem; color: #475569; letter-spacing: 0.06em; text-transform: uppercase; margin: 0 0 8px 0; }
    .creator-line { font-size: 0.78rem; color: #334155; }
    .creator-name { color: #3b82f6; font-weight: 600; }
    .creator-roles { color: #06b6d4; }
    .tech-pill { display: inline-block; background: #0f2235; border: 1px solid #1e3a5f; color: #60a5fa; font-size: 10px; font-weight: 500; padding: 2px 8px; border-radius: 10px; margin: 0 3px 0 0; }

    .metric-card { background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; padding: 1.1rem 1rem; text-align: center; }
    .metric-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.09em; color: #475569; margin-bottom: 6px; }
    .metric-value { font-size: 2rem; font-weight: 700; line-height: 1; color: #f1f5f9; }
    .metric-unit  { font-size: 11px; color: #334155; margin-top: 4px; }

    .badge-high   { background:#7f1d1d; color:#fca5a5; padding:4px 14px; border-radius:20px; font-size:12px; font-weight:600; }
    .badge-medium { background:#78350f; color:#fcd34d; padding:4px 14px; border-radius:20px; font-size:12px; font-weight:600; }
    .badge-low    { background:#14532d; color:#86efac; padding:4px 14px; border-radius:20px; font-size:12px; font-weight:600; }

    .section-title { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; color: #334155; border-bottom: 1px solid #1e293b; padding-bottom: 7px; margin-bottom: 14px; margin-top: 6px; }
    .feature-card { background: #0f172a; border: 1px solid #1e293b; border-radius: 10px; padding: 0.85rem 1.1rem; margin-bottom: 9px; }
    .step-wait { color: #334155; font-size: 12px; padding: 2px 0; }
    .step-run  { color: #f59e0b; font-size: 12px; padding: 2px 0; }
    .step-done { color: #22c55e; font-size: 12px; padding: 2px 0; }

    [data-testid="stSidebar"] { background: #020617 !important; border-right: 1px solid #0f172a; }
    [data-testid="stSidebar"] .stMarkdown p { color: #94a3b8; }

    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stImage"] img { border-radius: 8px; }
    .stProgress > div > div { background: linear-gradient(90deg,#3b82f6,#06b6d4); }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

VEHICLE_NAMES  = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
FRAME_SIZE     = (640, 360)
PIXEL_TO_KMH   = 0.05
SPEED_HISTORY  = 5
LINE_RATIO     = 0.65

# Model options — n=fastest/least accurate, s=good balance, m=best accuracy
MODEL_OPTIONS = {
    "YOLOv8n — Fast (less accurate)": "yolov8n.pt",
    "YOLOv8s — Balanced ✓ Recommended": "yolov8s.pt",
    "YOLOv8m — Accurate (slower)": "yolov8m.pt",
}

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def ensure_dirs() -> None:
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/history", exist_ok=True)
    os.makedirs("data", exist_ok=True)


def ensure_model(model_name: str) -> bool:
    """Download YOLOv8 model if not already present. Returns True on success."""
    try:
        from ultralytics import YOLO
        with st.spinner(f"Downloading {model_name} model (first run only)…"):
            YOLO(model_name)
        return True
    except Exception as e:
        st.error(f"Failed to download model: {e}")
        return False


def alert_badge(total: int) -> tuple[str, str]:
    if total >= 20:
        return '<span class="badge-high">🔴 HIGH TRAFFIC</span>', "HIGH"
    elif total >= 10:
        return '<span class="badge-medium">🟡 MEDIUM TRAFFIC</span>', "MEDIUM"
    return '<span class="badge-low">🟢 LOW TRAFFIC</span>', "LOW"


def save_history(counts: dict, total: int) -> pd.DataFrame:
    record = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **counts,
        "Total": total,
    }
    path = "outputs/history/traffic_history.csv"
    df   = pd.concat(
        [pd.read_csv(path), pd.DataFrame([record])], ignore_index=True
    ) if os.path.exists(path) else pd.DataFrame([record])
    df.to_csv(path, index=False)
    return df


def save_alert(total: int) -> None:
    _, level = alert_badge(total)
    record   = {
        "Timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Total Vehicles": total,
        "Alert":          level,
    }
    path = "outputs/traffic_alerts.csv"
    df   = pd.concat(
        [pd.read_csv(path), pd.DataFrame([record])], ignore_index=True
    ) if os.path.exists(path) else pd.DataFrame([record])
    df.to_csv(path, index=False)


def metric_card(label: str, value, unit: str, color: str = "#f1f5f9") -> str:
    return (
        f"<div class='metric-card'>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='metric-value' style='color:{color}'>{value}</div>"
        f"<div class='metric-unit'>{unit}</div>"
        f"</div>"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PROJECT HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="project-header">
    <div class="project-title">
        🚦 Real-Time Intelligent Object Tracking and Video Analytics System
    </div>
    <div class="project-subtitle">YOLOv8 &nbsp;·&nbsp; ByteTrack &nbsp;·&nbsp; Computer Vision &nbsp;·&nbsp; Traffic Analytics</div>
    <div class="creator-line">
        Created by &nbsp;<span class="creator-name">KalyanaSundar</span>
        &nbsp;—&nbsp;
        <span class="creator-roles">AI Engineer &nbsp;/&nbsp; Computer Vision Engineer &nbsp;/&nbsp; Data Scientist</span>
    </div>
    <div style="margin-top:10px">
        <span class="tech-pill">YOLOv8</span>
        <span class="tech-pill">ByteTrack</span>
        <span class="tech-pill">OpenCV</span>
        <span class="tech-pill">Streamlit</span>
        <span class="tech-pill">Plotly</span>
        <span class="tech-pill">Pandas</span>
        <span class="tech-pill">Python</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CORE VIDEO PROCESSING  (cloud-safe — no cv2.imshow)
# ══════════════════════════════════════════════════════════════════════════════

def process_video(
    video_path: str,
    features: set,
    model_name: str,
    frame_ph,
    progress_bar,
    status_text,
) -> dict | None:

    try:
        from ultralytics import YOLO
    except ImportError:
        st.error("ultralytics not installed — run: pip install ultralytics")
        return None

    model = YOLO(model_name)
    cap   = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error(f"Cannot open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fw, fh       = FRAME_SIZE
    line_y       = int(fh * LINE_RATIO)

    counted_ids:    set[int]         = set()
    vehicle_counts: dict             = {v: 0 for v in VEHICLE_NAMES.values()}
    prev_pos:       dict             = {}
    speed_hist:     dict             = defaultdict(list)
    speed_now:      dict             = {}
    all_detections: list             = []

    writer = None
    if "export" in features:
        writer = cv2.VideoWriter(
            "outputs/traffic_analytics_output.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            int(fps_video),
            FRAME_SIZE,
        )

    do_track = bool(features & {"tracking", "counting", "speed", "classification"})
    display_every = max(1, int(fps_video / 8))
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, FRAME_SIZE)
        t_now = time.perf_counter()

        if do_track:
            results   = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
                classes=list(VEHICLE_NAMES.keys()),
            )
            annotated = results[0].plot()
        else:
            annotated = frame.copy()
            results   = None

        if features & {"counting", "classification"}:
            cv2.line(annotated, (0, line_y), (fw, line_y), (0, 0, 255), 2)
            cv2.putText(annotated, "COUNTING LINE",
                        (fw // 2 - 65, line_y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)

        if results and results[0].boxes.id is not None:
            boxes   = results[0].boxes.xyxy.cpu().numpy()
            ids     = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, tid, cls in zip(boxes, ids, classes):
                if cls not in VEHICLE_NAMES:
                    continue

                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                if features & {"counting", "classification"}:
                    if abs(cy - line_y) < 12 and tid not in counted_ids:
                        counted_ids.add(tid)
                        vehicle_counts[VEHICLE_NAMES[cls]] += 1
                        all_detections.append({
                            "Frame":        frame_idx,
                            "Time":         datetime.now().strftime("%H:%M:%S"),
                            "Vehicle Type": VEHICLE_NAMES[cls],
                            "Track ID":     int(tid),
                        })

                if "speed" in features:
                    if tid in prev_pos:
                        px, py, pt = prev_pos[tid]
                        dt = t_now - pt
                        if dt > 0:
                            dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                            spd  = (dist / dt) * PIXEL_TO_KMH
                            speed_hist[tid].append(spd)
                            if len(speed_hist[tid]) > SPEED_HISTORY:
                                speed_hist[tid].pop(0)
                            speed_now[tid] = round(
                                sum(speed_hist[tid]) / len(speed_hist[tid]), 1
                            )
                            col = (0, 255, 255) if speed_now[tid] < 60 else (0, 100, 255)
                            cv2.putText(
                                annotated, f"{speed_now[tid]:.0f} km/h",
                                (int(cx) - 28, int(y1) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1,
                            )
                    prev_pos[tid] = (cx, cy, t_now)

        if features & {"counting", "classification"}:
            cv2.rectangle(annotated, (6, 6), (175, 125), (0, 0, 0), -1)
            yp = 22
            for vname, cnt in vehicle_counts.items():
                cv2.putText(annotated, f"{vname}: {cnt}", (11, yp),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                yp += 20
            cv2.putText(annotated, f"TOTAL: {len(counted_ids)}", (11, yp + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 0), 2)

        cv2.putText(
            annotated,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            (fw - 220, fh - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1,
        )

        if writer:
            writer.write(annotated)

        # UI update — convert BGR→RGB for Streamlit display (no cv2.imshow needed)
        if frame_idx % display_every == 0:
            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_ph.image(frame_rgb, use_column_width=True)
            progress_bar.progress(min(frame_idx / max(total_frames, 1), 1.0))
            status_text.markdown(
                f"**Processing** — frame **{frame_idx}** / {total_frames} &nbsp;|&nbsp; "
                f"vehicles counted: **{len(counted_ids)}**"
            )

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()

    # Save outputs
    if features & {"counting", "classification", "export"}:
        rows = list(vehicle_counts.items()) + [("Total", len(counted_ids))]
        pd.DataFrame(rows, columns=["Vehicle Type", "Count"]).to_csv(
            "outputs/vehicle_report.csv", index=False
        )

    history_df = save_history(vehicle_counts, len(counted_ids))
    save_alert(len(counted_ids))

    speeds    = list(speed_now.values())
    avg_speed = round(sum(speeds) / len(speeds), 1) if speeds else 0
    max_speed = round(max(speeds), 1) if speeds else 0

    return {
        "vehicle_counts":  vehicle_counts,
        "total":           len(counted_ids),
        "speeds":          speed_now,
        "avg_speed":       avg_speed,
        "max_speed":       max_speed,
        "history_df":      history_df,
        "detections":      all_detections,
        "frames":          frame_idx,
        "video_exported":  "export" in features,
        "model_used":      model_name,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        "<div style='height:3px;background:linear-gradient(90deg,"
        "#3b82f6,#06b6d4,#10b981);border-radius:2px;margin-bottom:18px'></div>",
        unsafe_allow_html=True,
    )

    st.markdown("### 📤 Upload Video")
    uploaded = st.file_uploader(
        "Drop traffic/surveillance video",
        type=["mp4", "avi", "mov", "mkv", "wmv"],
        help="MP4, AVI, MOV, MKV, WMV supported",
    )

    st.markdown("---")
    st.markdown("### 🤖 Model")
    model_label = st.selectbox(
        "Detection model",
        options=list(MODEL_OPTIONS.keys()),
        index=1,  # default to yolov8s
        help="Larger models detect more accurately but take longer to process.",
    )
    selected_model = MODEL_OPTIONS[model_label]

    st.markdown("---")
    st.markdown("### ⚙️ Features")

    feat_tracking  = st.checkbox("🎯 Object Tracking",         value=True)
    feat_count     = st.checkbox("🔢 Vehicle Counting",        value=True)
    feat_classify  = st.checkbox("🚗 Vehicle Classification",  value=True)
    feat_speed     = st.checkbox("⚡ Speed Estimation",         value=True)
    feat_export    = st.checkbox("💾 Export Annotated Video",  value=False,
                                  help="Writes every frame — slower processing")

    features: set = set()
    if feat_tracking: features.add("tracking")
    if feat_count:    features.add("counting")
    if feat_classify: features.add("classification")
    if feat_speed:    features.add("speed")
    if feat_export:   features.add("export")

    st.markdown("")
    run_btn = st.button("▶ Run Analysis", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("### 🗂 Navigation")
    page = st.radio(
        "",
        ["📊 Dashboard", "📜 History", "🚨 Alerts", "📁 Downloads"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:11px;color:#1e3a5f;text-align:center;line-height:1.8'>"
        "YOLOv8 · ByteTrack · OpenCV<br>"
        "<span style='color:#0f2235'>KalyanaSundar © 2025</span>"
        "</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

ensure_dirs()

if "results" not in st.session_state:
    st.session_state["results"] = None
if "video_path" not in st.session_state:
    st.session_state["video_path"] = None


# ══════════════════════════════════════════════════════════════════════════════
#  RUN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

if run_btn:
    if uploaded is None:
        st.warning("⬆️ Please upload a video in the sidebar first.")
    elif not features:
        st.warning("Select at least one feature to run.")
    else:
        # Ensure model is downloaded before processing
        if not ensure_model(selected_model):
            st.stop()

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(uploaded.read())
        tmp.close()
        st.session_state["video_path"] = tmp.name

        st.markdown("---")
        st.markdown("## ⚙️ Processing")
        st.markdown(
            f"**File:** `{uploaded.name}` &nbsp;|&nbsp; "
            f"**Model:** `{selected_model}` &nbsp;|&nbsp; "
            f"**Features:** {', '.join(sorted(features))}"
        )

        col_vid, col_steps = st.columns([3, 1])

        with col_vid:
            frame_ph = st.empty()

        with col_steps:
            st.markdown("<div class='section-title'>Pipeline</div>", unsafe_allow_html=True)
            step_list = []
            if "tracking"       in features: step_list += ["YOLOv8 detection", "ByteTrack tracking"]
            if "counting"       in features: step_list.append("Line crossing count")
            if "classification" in features: step_list.append("Vehicle classification")
            if "speed"          in features: step_list.append("Speed estimation")
            step_list += ["CSV export", "History update", "Alert generation"]
            if "export"         in features: step_list.append("Video file export")

            steps_ph = st.markdown(
                "".join(f"<div class='step-wait'>○ {s}</div>" for s in step_list),
                unsafe_allow_html=True,
            )

        prog  = st.progress(0)
        stat  = st.empty()

        results = process_video(
            st.session_state["video_path"],
            features,
            selected_model,
            frame_ph,
            prog,
            stat,
        )

        if results:
            steps_ph.markdown(
                "".join(f"<div class='step-done'>✓ {s}</div>" for s in step_list),
                unsafe_allow_html=True,
            )
            prog.progress(1.0)
            stat.markdown("✅ **Analysis complete!**")
            st.session_state["results"] = results
            st.success(
                f"Done! Detected **{results['total']} vehicles** "
                f"across **{results['frames']}** frames."
            )
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGES
# ══════════════════════════════════════════════════════════════════════════════

results = st.session_state.get("results")

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: Dashboard
# ─────────────────────────────────────────────────────────────────────────────
if page == "📊 Dashboard":
    st.markdown("<div class='section-title'>Live Analytics Dashboard</div>", unsafe_allow_html=True)

    if results is None:
        st.markdown("""
        <div style='background:#0a0f1e;border:1px dashed #1e3a5f;border-radius:14px;
                    padding:3.5rem 2rem;text-align:center;margin-top:1rem'>
            <div style='font-size:3rem;margin-bottom:14px'>🚦</div>
            <div style='font-size:1.1rem;font-weight:600;color:#94a3b8;margin-bottom:8px'>
                No analysis run yet
            </div>
            <div style='font-size:13px;color:#334155'>
                Upload a video in the sidebar and click
                <strong style='color:#3b82f6'>▶ Run Analysis</strong> to begin
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        r      = results
        counts = r["vehicle_counts"]
        total  = r["total"]
        badge, _ = alert_badge(total)

        cols = st.columns(6)
        cards = [
            ("Total Vehicles",  total,               "detected",    "#60a5fa"),
            ("Cars",            counts["Car"],        "vehicles",    "#f1f5f9"),
            ("Motorcycles",     counts["Motorcycle"], "vehicles",    "#f1f5f9"),
            ("Buses",           counts["Bus"],        "vehicles",    "#f1f5f9"),
            ("Avg Speed",       r["avg_speed"],       "km/h",        "#34d399"),
            ("Max Speed",       r["max_speed"],       "km/h",        "#f87171"),
        ]
        for col, (lbl, val, unit, clr) in zip(cols, cards):
            col.markdown(metric_card(lbl, val, unit, clr), unsafe_allow_html=True)

        st.markdown("")

        st.markdown(
            f"<div style='background:#0a0f1e;border:1px solid #1e293b;"
            f"border-radius:10px;padding:12px 18px;display:flex;"
            f"align-items:center;gap:14px;margin:0.5rem 0 1.2rem 0'>"
            f"<div style='font-size:12px;color:#475569;font-weight:500'>Traffic Status</div>"
            f"<div>{badge}</div>"
            f"<div style='font-size:11px;color:#1e3a5f;margin-left:auto'>"
            f"Analysed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp; "
            f"Model: {r.get('model_used', 'yolov8n.pt')}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        try:
            import plotly.graph_objects as go
            import plotly.express as px

            CHART_BG   = "#0a0f1e"
            PAPER_BG   = "#0a0f1e"
            GRID_COLOR = "#0f172a"
            FONT_COLOR = "#94a3b8"
            PALETTE    = ["#3b82f6", "#06b6d4", "#10b981", "#f59e0b"]

            col_bar, col_pie = st.columns(2)

            with col_bar:
                fig = go.Figure(go.Bar(
                    x=list(counts.keys()),
                    y=list(counts.values()),
                    marker_color=PALETTE,
                    text=list(counts.values()),
                    textposition="outside",
                    textfont=dict(color=FONT_COLOR, size=12),
                ))
                fig.update_layout(
                    title=dict(text="Vehicle Count by Type", font=dict(color="#e2e8f0", size=14)),
                    plot_bgcolor=CHART_BG, paper_bgcolor=PAPER_BG,
                    font=dict(color=FONT_COLOR, size=12),
                    xaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=FONT_COLOR)),
                    yaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=FONT_COLOR)),
                    margin=dict(t=44, b=16, l=16, r=16), height=280,
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_pie:
                non_zero = {k: v for k, v in counts.items() if v > 0}
                if non_zero:
                    fig2 = go.Figure(go.Pie(
                        labels=list(non_zero.keys()),
                        values=list(non_zero.values()),
                        marker_colors=PALETTE[:len(non_zero)],
                        hole=0.48,
                        textfont=dict(size=12),
                    ))
                    fig2.update_layout(
                        title=dict(text="Fleet Composition", font=dict(color="#e2e8f0", size=14)),
                        plot_bgcolor=CHART_BG, paper_bgcolor=PAPER_BG,
                        font=dict(color=FONT_COLOR, size=12),
                        legend=dict(font=dict(color=FONT_COLOR)),
                        margin=dict(t=44, b=10, l=10, r=10), height=280,
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("No vehicle classifications yet.")

            if r["speeds"]:
                spd_vals = list(r["speeds"].values())
                fig3 = px.histogram(
                    x=spd_vals, nbins=25,
                    labels={"x": "Speed (km/h)", "y": "Vehicle Count"},
                    title="Speed Distribution",
                    color_discrete_sequence=["#3b82f6"],
                )
                fig3.update_layout(
                    plot_bgcolor=CHART_BG, paper_bgcolor=PAPER_BG,
                    font=dict(color=FONT_COLOR, size=12),
                    title_font=dict(color="#e2e8f0", size=14),
                    xaxis=dict(gridcolor=GRID_COLOR),
                    yaxis=dict(gridcolor=GRID_COLOR),
                    margin=dict(t=44, b=16, l=16, r=16), height=220,
                )
                st.plotly_chart(fig3, use_container_width=True)

        except ImportError:
            st.warning("Install plotly for charts: pip install plotly")
            st.dataframe(
                pd.DataFrame(list(counts.items()), columns=["Type", "Count"]),
                use_container_width=True,
            )

        if r["detections"]:
            st.markdown("<div class='section-title' style='margin-top:1rem'>Detection Log</div>", unsafe_allow_html=True)
            df_det = pd.DataFrame(r["detections"])
            st.dataframe(df_det.tail(100), use_container_width=True, height=230)

            csv_det = df_det.to_csv(index=False).encode()
            st.download_button(
                "⬇ Download Detection Log CSV",
                csv_det, "detection_log.csv", "text/csv",
            )


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: History
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📜 History":
    st.markdown("<div class='section-title'>Historical Traffic Trends</div>", unsafe_allow_html=True)

    path = "outputs/history/traffic_history.csv"
    if not os.path.exists(path):
        st.info("No historical data yet. Run an analysis first.")
    else:
        df = pd.read_csv(path)

        cols = st.columns(4)
        summary = [
            ("Sessions",      len(df),              "",       "#60a5fa"),
            ("Total Vehicles",df["Total"].sum(),     "all time","#34d399"),
            ("Avg / Session", round(df["Total"].mean(), 1), "vehicles", "#f1f5f9"),
            ("Peak Session",  int(df["Total"].max()), "vehicles", "#f87171"),
        ]
        for col, (lbl, val, unit, clr) in zip(cols, summary):
            col.markdown(metric_card(lbl, val, unit, clr), unsafe_allow_html=True)

        st.markdown("")

        try:
            import plotly.express as px
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            fig = px.line(
                df, x="Timestamp",
                y=["Car", "Motorcycle", "Bus", "Truck", "Total"],
                title="Traffic Trends Over Sessions",
                markers=True,
                color_discrete_sequence=["#3b82f6","#f59e0b","#10b981","#ef4444","#a78bfa"],
            )
            fig.update_layout(
                plot_bgcolor="#0a0f1e", paper_bgcolor="#0a0f1e",
                font=dict(color="#94a3b8"), title_font=dict(color="#e2e8f0", size=14),
                xaxis=dict(gridcolor="#0f172a"), yaxis=dict(gridcolor="#0f172a"),
                legend=dict(font=dict(color="#94a3b8")),
                height=360, margin=dict(t=44, b=16, l=16, r=16),
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            pass

        st.dataframe(df, use_container_width=True)
        st.download_button(
            "⬇ Download History CSV",
            df.to_csv(index=False).encode(),
            "traffic_history.csv", "text/csv",
        )


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: Alerts
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🚨 Alerts":
    st.markdown("<div class='section-title'>Traffic Congestion Alerts</div>", unsafe_allow_html=True)

    path = "outputs/traffic_alerts.csv"
    if not os.path.exists(path):
        st.info("No alerts generated yet. Run an analysis first.")
    else:
        df = pd.read_csv(path)

        high   = (df["Alert"] == "HIGH").sum()
        medium = (df["Alert"] == "MEDIUM").sum()
        low    = (df["Alert"] == "LOW").sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card("Total Alerts",  len(df), "sessions",  "#60a5fa"), unsafe_allow_html=True)
        c2.markdown(metric_card("High",    high,   "sessions",  "#f87171"), unsafe_allow_html=True)
        c3.markdown(metric_card("Medium",  medium, "sessions",  "#fbbf24"), unsafe_allow_html=True)
        c4.markdown(metric_card("Low",     low,    "sessions",  "#34d399"), unsafe_allow_html=True)

        st.markdown("")

        try:
            import plotly.express as px
            counts_alert = df["Alert"].value_counts().reset_index()
            counts_alert.columns = ["Level", "Count"]
            fig = px.bar(
                counts_alert, x="Level", y="Count",
                title="Alert Level Distribution",
                color="Level",
                color_discrete_map={"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#22c55e"},
            )
            fig.update_layout(
                plot_bgcolor="#0a0f1e", paper_bgcolor="#0a0f1e",
                font=dict(color="#94a3b8"), title_font=dict(color="#e2e8f0", size=14),
                xaxis=dict(gridcolor="#0f172a"), yaxis=dict(gridcolor="#0f172a"),
                showlegend=False,
                height=240, margin=dict(t=44, b=16, l=16, r=16),
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            pass

        def style_alert(val: str) -> str:
            return {
                "HIGH":   "background-color:#7f1d1d;color:#fca5a5",
                "MEDIUM": "background-color:#78350f;color:#fcd34d",
                "LOW":    "background-color:#14532d;color:#86efac",
            }.get(val, "")

        st.dataframe(
            df.style.applymap(style_alert, subset=["Alert"]),
            use_container_width=True,
        )
        st.download_button(
            "⬇ Download Alerts CSV",
            df.to_csv(index=False).encode(),
            "traffic_alerts.csv", "text/csv",
        )


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: Downloads
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📁 Downloads":
    st.markdown("<div class='section-title'>Export & Downloads</div>", unsafe_allow_html=True)

    OUTPUT_FILES = {
        "outputs/vehicle_report.csv":           ("📊 Vehicle Report CSV",        "text/csv"),
        "outputs/history/traffic_history.csv":  ("📜 Historical Analytics CSV",  "text/csv"),
        "outputs/traffic_alerts.csv":           ("🚨 Traffic Alerts CSV",        "text/csv"),
        "outputs/traffic_analytics_output.mp4": ("🎬 Annotated Video (MP4)",     "video/mp4"),
    }

    for fpath, (label, mime) in OUTPUT_FILES.items():
        col_info, col_btn = st.columns([3, 1])
        exists = os.path.exists(fpath)
        size   = f"{os.path.getsize(fpath) / 1024:.0f} KB" if exists else "—"
        status = "✅" if exists else "⬜"

        with col_info:
            st.markdown(
                f"<div class='feature-card'>"
                f"<span style='font-size:13px;color:#e2e8f0'>{status} {label}</span>"
                f"<span style='font-size:11px;color:#334155;margin-left:10px'>{fpath}</span>"
                f"<span style='font-size:11px;color:#1e3a5f;float:right'>{size}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col_btn:
            if exists:
                with open(fpath, "rb") as f:
                    st.download_button(
                        "⬇ Download", f.read(),
                        file_name=Path(fpath).name,
                        mime=mime, key=fpath,
                        use_container_width=True,
                    )
            else:
                st.markdown(
                    "<div style='padding-top:10px;font-size:12px;color:#1e3a5f'>"
                    "Not generated</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    st.markdown("<div class='section-title'>Analytics Report</div>", unsafe_allow_html=True)

    if results:
        r = results
        report = "\n".join([
            "=" * 46,
            "  REAL-TIME INTELLIGENT TRAFFIC ANALYTICS",
            "  VEHICLE MONITORING SYSTEM",
            f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Created by: KalyanaSundar",
            f"  Role      : AI Engineer | CV Engineer | Data Scientist",
            f"  Model     : {r.get('model_used', 'yolov8n.pt')}",
            "=" * 46,
            "",
            f"  {'Cars':<16}: {r['vehicle_counts']['Car']}",
            f"  {'Motorcycles':<16}: {r['vehicle_counts']['Motorcycle']}",
            f"  {'Buses':<16}: {r['vehicle_counts']['Bus']}",
            f"  {'Trucks':<16}: {r['vehicle_counts']['Truck']}",
            "-" * 46,
            f"  {'Total Vehicles':<16}: {r['total']}",
            f"  {'Avg Speed':<16}: {r['avg_speed']} km/h",
            f"  {'Max Speed':<16}: {r['max_speed']} km/h",
            f"  {'Frames Processed':<16}: {r['frames']}",
            "=" * 46,
        ])
        st.code(report, language="text")
        st.download_button(
            "⬇ Download Report TXT",
            report.encode(),
            "analytics_report.txt", "text/plain",
        )
    else:
        st.info("Run an analysis to generate the report.")
