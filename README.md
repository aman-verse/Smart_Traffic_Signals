# 🚦 Smart Traffic Signals: Vehicle Density Estimation & Dynamic Timing  

## 🎯 Objective  
The project aims to **estimate real-time vehicle density** from a camera feed or recorded traffic video and **dynamically adjust traffic signal durations**.  
This helps **reduce average waiting time**, improve traffic throughput, and demonstrate an **AI-based traffic management system**.

---

## ⚙️ How It Works  

1. **Video Input:**  
   Accepts either a **live webcam feed** or **uploaded traffic video**.  

2. **Vehicle Detection:**  
   Vehicles (cars, buses, trucks, bikes) are detected using **YOLOv8**, a real-time object detection model.  

3. **Lane Definition:**  
   Users define **Regions of Interest (ROIs)** corresponding to traffic lanes for accurate counting.  

4. **Counting & Density Estimation:**  
   Vehicles are counted per lane in each frame to estimate **lane-wise traffic density**.  

5. **Rule-Based Signal Logic:**  
   Green signal duration is adjusted proportionally to lane density — **higher density = longer green time**.  

6. ## 🖥️ Dashboard Features
- Upload video or use webcam feed
- Live vehicle detection and counting
- ROI (lane) management
- Real-time metrics: FPS, vehicle counts, throughput
- Optional robustness tests (dark/rain simulation)
- Auto-saves ROI configuration for each video
- Adaptive signal timing visualization

---

## 🧩 Methodology  

1. **Data Acquisition:**  
   - Video input from **webcam** or pre-recorded traffic footage.  
   - Optional ground-truth CSV to evaluate counting accuracy.

2. **Preprocessing:**  
   - Define **lane ROIs** manually or load from saved configuration.  
   - Optional simulation of **low-light or rain** conditions for robustness testing.

3. **Vehicle Detection:**  
   - YOLOv5 model detects vehicles in each frame.  
   - Only relevant vehicle classes (car, bus, truck, motorbike) are counted.  

4. **Vehicle Tracking (Optional):**  
   - SORT tracker can be used to maintain unique IDs across frames.  

5. **Lane-Wise Counting & Density Calculation:**  
   - Centroid of each detected vehicle is checked against lane ROIs.  
   - Counts per lane are used for traffic density estimation.

6. **Adaptive Signal Timing:**  
   - Lane green time is **proportional to vehicle density** within limits defined in the dashboard.  
   - Average green time is calculated dynamically for each frame.

7. **Visualization & Metrics:**  
   - Streamlit displays live video, lane counts, adaptive green times, and FPS metrics.  
   - Ground-truth comparison provides **MAE and MAPE** for evaluation.

---

## 🧠 Model Used  

| **Component** | **Model / Method** | **Purpose** |
|----------------|--------------------|--------------|
| Vehicle Detection | **YOLOv5** | Real-time detection of cars, buses, trucks, bikes |
| Control Logic | **Rule-Based** | Adaptive signal timing based on lane density |
| Interface | **Streamlit** | Live dashboard for visualization and monitoring |

---

## 🧩 Setup Instructions  

### 1. Clone the repository
```bash
git clone https://github.com/aman-verse/Smart_Traffic_Signals.git
cd Smart_Traffic_Signals

2. Install dependencies
    pip install -r requirements.txt

3. Run the dashboard
    streamlit run app.py

4. Choose input
    Upload traffic video (.mp4, .avi, .mov)
    or Use webcam feed for live detection
