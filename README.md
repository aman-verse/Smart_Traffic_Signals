# 🚦 Smart Traffic Signals: Adaptive Signal Timing Based on Vehicle Density

This project implements a real-time traffic signal control system that adjusts signal timing based on the number of vehicles detected in each lane. Utilizing YOLOv8 for vehicle detection and OpenCV for image processing, the system dynamically allocates green signal durations to optimize traffic flow.

---

## 🧠 Features

- **Real-Time Vehicle Detection**: Uses YOLOv8 for accurate vehicle detection in video streams.
- **Lane-wise Vehicle Counting**: Counts vehicles per lane to assess traffic density.
- **Adaptive Signal Timing**: Adjusts green signal durations based on lane density.
- **Streamlit Dashboard**: Provides a user-friendly interface for real-time monitoring and control.
- **Robustness Testing**: Simulates dark and rainy conditions to evaluate system performance.

---

## 📥 Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/aman-verse/Smart_Traffic_Signals.git
   cd Smart_Traffic_Signals

2. **Set up a virtual environment (optional but recommended):**
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. **Install dependencies:**
    pip install -r requirements.txt

4. 🚀** Usage**

Run the Streamlit app:
    
    streamlit run app.py


.
**# 
📂 Project Structure**

Smart_Traffic_Signals/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── data/                  # Lane ROI configurations
├── models/                # YOLOv8 model weights
│   ├── yolov8n.pt
├── README.md              # Project documentation


