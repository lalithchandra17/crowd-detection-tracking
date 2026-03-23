# Crowd Detection and Tracking System

## Overview
This project implements a **real-time smart surveillance system** that detects and tracks multiple people using **YOLOv8** and **BoTSORT**.  

It assigns **unique IDs** to individuals, monitors crowd movement, detects entry into restricted zones, and logs tracking data for analysis.

---

## Key Features
-  Real-time person detection using YOLOv8  
-  Multi-object tracking with unique IDs (BoTSORT)  
-  People counting in each frame  
-  Zone-based alert system (restricted area detection)  
-  Multi-camera simulation support  
-  CSV data logging for analytics  
-  Output video generation with tracking visualization  

---

##  Tech Stack
- **Programming Language:** Python  
- **Libraries:** OpenCV, Ultralytics YOLOv8  
- **Tracking Algorithm:** BoTSORT  
- **Tools:** NumPy  

---

##  How It Works
1. The system reads video frames from input sources  
2. YOLOv8 detects people in each frame  
3. BoTSORT assigns and maintains unique IDs across frames  
4. Bounding box centers are used to detect zone entry  
5. Alerts are triggered when a person enters the defined zone  
6. Tracking data is saved into a CSV file  
7. Output video is generated with all visualizations  

---

##  Installation & Setup

### Clone the repository
```bash
git clone https://github.com/yourusername/crowd-detection-tracking.git
cd crowd-detection-tracking
