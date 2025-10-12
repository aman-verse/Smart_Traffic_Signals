import os
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO
import json

# --- SORT tracker ---
try:
    from sort import Sort
    SORT_AVAILABLE = True
except Exception:
    SORT_AVAILABLE = False

# --- Streamlit Page Setup ---
st.set_page_config(layout="wide", page_title="AI Traffic Signals")
st.title("🚦 Smart Traffic Signals: Vehicle Density & Adaptive Timing")
st.markdown(
    "Upload a traffic video or use live webcam, define lanes, detect vehicles, "
    "compute metrics, and display adaptive signal timing dynamically."
)

# --- Sidebar Controls ---
st.sidebar.header("Configuration")
video_source_option = st.sidebar.selectbox("Video Source", ["Upload Video", "Webcam"])
model_option = st.sidebar.selectbox("YOLO Model", ["yolov8n.pt", "yolov8s.pt"])
conf_threshold = st.sidebar.slider("Detection Confidence", 0.1, 0.9, 0.45)
use_sort = st.sidebar.checkbox("Use SORT tracker", value=SORT_AVAILABLE and True)
min_green, max_green = st.sidebar.slider("Adaptive Green Time Range (s)", 5, 120, (10, 30))
sample_frame_interval = st.sidebar.number_input("Record counts every N frames", min_value=1, value=5)
gt_file = st.sidebar.file_uploader("Optional: Upload Ground-Truth CSV (frame_index,total_count)", type=["csv"])

# --- Robustness options ---
st.sidebar.markdown("---")
st.sidebar.markdown("**Robustness Tests**")
robust_dark = st.sidebar.checkbox("Simulate Dark Lighting")
robust_rain = st.sidebar.checkbox("Simulate Rain Noise")

# --- Load YOLO Model ---
@st.cache_resource
def load_yolo_model(model_name):
    return YOLO(model_name)

try:
    model = load_yolo_model(model_option)
except Exception as e:
    st.error(f"Error loading YOLO model `{model_option}`: {e}")
    st.stop()

# --- Video Input ---
if video_source_option == "Upload Video":
    video_file = st.file_uploader("Upload traffic video", type=["mp4","avi","mov"])
    if video_file is None:
        st.info("Upload a video to proceed.")
        st.stop()
    tfile_path = "temp_video.mp4"
    with open(tfile_path, "wb") as f:
        f.write(video_file.read())
    cap = cv2.VideoCapture(tfile_path)
    video_name = os.path.splitext(video_file.name)[0]
else:
    cap = cv2.VideoCapture(0)
    video_name = "webcam"

ret, first_frame = cap.read()
if not ret:
    st.error("Could not read first frame.")
    st.stop()
frame_h, frame_w = first_frame.shape[:2]

# --- ROI / Lanes ---
st.subheader("Lane ROIs")
roi_json_path = f"data/roi_config_{video_name}.json"
lanes = []

if os.path.exists(roi_json_path):
    try:
        with open(roi_json_path, "r") as f:
            data = json.load(f)
            for coords in data.get("roi_polygon", []):
                if len(coords) == 4:
                    lanes.append(tuple(coords))
        st.success(f"Loaded {len(lanes)} lanes from saved ROI config.")
    except:
        st.warning("Failed to read ROI JSON, using manual input.")

manual_rois = st.text_input("Enter lanes manually as x1,y1,x2,y2|x1,y1,x2,y2", "")
if manual_rois:
    try:
        lanes = []
        for p in manual_rois.split("|"):
            coords = tuple(map(int, p.split(",")))
            if len(coords) == 4:
                lanes.append(coords)
        st.success(f"{len(lanes)} lanes parsed from manual input.")

        os.makedirs("data", exist_ok=True)
        with open(roi_json_path, "w") as f:
            json.dump({"roi_polygon": lanes}, f)
        st.info(f"Saved ROI config for {video_name}.")
    except:
        st.error("Failed to parse manual ROIs. Check format.")

if not lanes:
    st.warning("No lanes defined. Counting total vehicles only.")

# --- Ground Truth Parsing using pandas ---
gt_df, gt_map = None, {}
if gt_file is not None:
    try:
        gt_df = pd.read_csv(gt_file)
        if "frame_index" in gt_df.columns and "total_count" in gt_df.columns:
            gt_map = dict(zip(gt_df["frame_index"], gt_df["total_count"]))
            st.sidebar.success(f"Loaded ground truth for {len(gt_map)} frames using pandas.")
            if st.sidebar.checkbox("Preview Ground Truth Data"):
                st.sidebar.dataframe(gt_df.head())
        else:
            st.sidebar.error("CSV must contain 'frame_index' and 'total_count' columns.")
    except Exception as e:
        st.sidebar.error(f"Failed to parse GT CSV: {e}")

# --- Helper: robustness transforms ---
def apply_robustness(frame, dark=False, rain=False):
    out = frame.copy()
    if dark:
        out = cv2.convertScaleAbs(out, alpha=0.6, beta=0)
    if rain:
        noise = np.random.normal(0, 15, out.shape).astype(np.uint8)
        out = cv2.add(out, noise)
    return out

# --- Start Processing ---
if st.button("Start Processing"):
    tracker = Sort(max_age=8, min_hits=2, iou_threshold=0.3) if use_sort and SORT_AVAILABLE else None
    sampled_frame_indices, predicted_total_counts, predicted_per_lane, inference_times = [], [], [], []
    total_detected_vehicles, frame_idx = 0, 0
    start_time_global = time.time()
    display_frame = st.empty()
    metrics_col1, metrics_col2 = st.columns(2)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        proc_frame = apply_robustness(frame, dark=robust_dark, rain=robust_rain)

        t0 = time.time()
        results = model(proc_frame, verbose=False)[0]
        t1 = time.time()
        inference_times.append(t1 - t0)

        detection_boxes = []
        for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            if int(cls) in [2,3,5,7] and float(conf) >= conf_threshold:
                x1,y1,x2,y2 = map(int, box)
                detection_boxes.append([x1,y1,x2,y2,float(conf)])

        dets_np = np.array(detection_boxes) if detection_boxes else np.empty((0,5))
        tracks = tracker.update(dets_np) if tracker is not None and dets_np.shape[0]>0 else \
                 np.array([[x1,y1,x2,y2,frame_idx*1000+i] for i,(x1,y1,x2,y2,_) in enumerate(detection_boxes)]) if dets_np.shape[0]>0 else np.empty((0,5))

        lane_counts = [0]*len(lanes) if lanes else []
        total_count_frame = 0

        for tr in tracks:
            x1,y1,x2,y2,tid = [int(v) for v in tr]
            cx,cy = (x1+x2)//2,(y1+y2)//2
            total_count_frame += 1
            for i,(lx1,ly1,lx2,ly2) in enumerate(lanes):
                if lx1<=cx<=lx2 and ly1<=cy<=ly2:
                    lane_counts[i]+=1
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.putText(frame,f"ID {tid}",(x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1)

        dominant_idx = np.argmax(lane_counts) if lanes else -1
        for i,(lx1,ly1,lx2,ly2) in enumerate(lanes):
            color = (0,255,0) if i==dominant_idx else (0,0,255)
            cv2.rectangle(frame,(lx1,ly1),(lx2,ly2),color,2)
            cv2.putText(frame,f"Lane {i+1}: {lane_counts[i]}",(lx1,ly1-8),cv2.FONT_HERSHEY_SIMPLEX,1.2,color,2)

        if frame_idx % sample_frame_interval==0:
            sampled_frame_indices.append(frame_idx)
            predicted_total_counts.append(total_count_frame)
            predicted_per_lane.append(lane_counts.copy() if lanes else [])

        total_detected_vehicles += total_count_frame
        elapsed_global = time.time()-start_time_global
        current_fps = frame_idx/elapsed_global if elapsed_global>0 else 0

        metrics_col1.metric("Frames processed", frame_idx)
        metrics_col1.metric("FPS (approx)", f"{current_fps:.2f}")
        metrics_col1.metric("Avg inference (ms)", f"{(np.mean(inference_times)*1000):.1f}")

        metrics_col2.metric("Total detected vehicles", total_detected_vehicles)
        if lanes:
            metrics_col2.write(f"Per-lane latest counts: {lane_counts}")

        display_frame.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB), channels="RGB")

    # --- Video Processing Complete ---
    cap.release()

    # --- Results & Metrics (After Video Stops) ---
    st.subheader("📊 Processing Complete — Results & Counting Accuracy")
    throughput_vpm = total_detected_vehicles / (elapsed_global / 60.0)
    st.write(f"- Total frames processed: **{frame_idx}**")
    st.write(f"- Total detected vehicles: **{total_detected_vehicles}**")
    st.write(f"- Throughput (vehicles/min): **{throughput_vpm:.2f}**")
    st.write(f"- Avg inference per frame (ms): **{(np.mean(inference_times)*1000):.1f}**")
    st.write(f"- Avg FPS: **{current_fps:.2f}**")

    # Counting Accuracy
    if gt_df is not None and not gt_df.empty:
        pred_df = pd.DataFrame({
            "frame_index": sampled_frame_indices,
            "pred_count": predicted_total_counts
        })
        merged_df = pd.merge(gt_df, pred_df, on="frame_index", how="inner")
        
        if not merged_df.empty:
            merged_df["abs_error"] = (merged_df["pred_count"] - merged_df["total_count"]).abs()
            merged_df["pct_error"] = merged_df["abs_error"] / merged_df["total_count"].replace(0, 1) * 100

            mae = merged_df["abs_error"].mean()
            mape = merged_df["pct_error"].mean()

            st.markdown("### 🧮 Counting Accuracy")
            st.write(f"- MAE (Mean Absolute Error): **{mae:.2f} vehicles**")
            st.write(f"- MAPE (Mean Absolute Percentage Error): **{mape:.2f}%**")

            with st.expander("View Ground Truth Comparison Table"):
                st.dataframe(merged_df.head(20))
        else:
            st.warning("No matching frames found between predictions and ground truth.")
    else:
        st.info("No ground truth CSV uploaded. MAE/MAPE cannot be computed.")

    # Adaptive green summary
    if predicted_per_lane and lanes:
        avg_adaptive_green_per_frame=[]
        for pl in predicted_per_lane:
            s = sum(pl)
            greens = [max(min_green, int((c/s)*max_green)) if s>0 else min_green for c in pl]
            avg_adaptive_green_per_frame.append(np.mean(greens))
        avg_adaptive_green = float(np.mean(avg_adaptive_green_per_frame))
        st.write(f"- Avg adaptive green (per-lane avg): **{avg_adaptive_green:.2f}s**")

    st.success("✅ Processing complete.")
