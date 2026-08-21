## NeuroTraffic Ops: AI Traffic Police System

## 👁️ Project Overview
NeuroTraffic Ops is an intelligent, low-latency urban surveillance suite designed to automate traffic monitoring. Built for the MIC AIML Department Recruitment Challenge (Computer Vision Track), this platform ingests raw street-level visual feeds and converts them into structured, actionable telemetry data using a dual-node deep learning architecture.

## ⚠️ Problem Statement
Modern traffic authorities require automated, real-time systems to monitor roads, classify vehicles, and detect incidents without relying on manual observation. The challenge is to build a reliable computer vision pipeline that can accurately isolate targets in dense traffic and provide fine-grained classification to improve overall traffic management and law enforcement.

## ⚙️ Installation Instructions
To run this dashboard locally, ensure you have Python 3.8+ installed.

1. **Clone the repository:**
```bash
git clone [https://github.com/Anshuuu0635/neurotraffic-ops.git](https://github.com/Anshuuu0635/neurotraffic-ops.git)
cd neurotraffic-ops
```
## 🗄️ Dataset Used
This project strictly utilizes the Mandatory Traffic Image Dataset provided by the MIC AIML department for Part 1 (Foundations). This dataset was critical for evaluating the baseline accuracy of vehicle classification (cars, buses, bikes, and trucks) and validating spatial counts.

## 🧠 Methodology
The system utilizes a Dual-Node architecture to balance speed and accuracy:

Part 1: Spatial Detection (YOLOv8 Core): The primary network scans the raw frame to isolate moving targets, extracting high-precision spatial bounding boxes in real-time.

Part 2: Deep Classification (Crop Router): Detected targets are cropped and routed through a secondary custom-trained neural network (CNN). This categorizes vehicles into specific classes (Car, Motorcycle, Bus, Truck) regardless of angle or scale. It is also architected to detect emergency vehicles and basic traffic violations.

## 💻 Technologies & AI Architecture Stack
* **Python:** Core backend logic.

* **Ultralytics YOLOv8:** Primary spatial object detection.

* **OpenCV & NumPy:** Image tensor routing and matrix manipulation.

* **NiceGUI:** Asynchronous, low-latency tactical UI rendering.

* **Tailwind CSS:** Edge-to-edge custom frontend styling.
* **Core Backend:** Python, OpenCV, NumPy
* **Deep Learning Engine:** Ultralytics YOLOv8 (Dual-Node Spatial & Classification)
* **Frontend Dashboard:** NiceGUI, Tailwind CSS (Asynchronous Tactical UI)
* **AI Collaborative Engineering:**
  * **Gemini:** Process alignment and workflow optimization
  * **GPT:** Precision prompt engineering and logic structuring
  * **Claude:** Training model deep-dive and architectural evaluation

## 📊 Results
The system successfully processes high-density street images, yielding accurate vehicle counts across 4 distinct classes. By optimizing confidence thresholds (0.18) and Intersection over Union (IoU) overlap parameters, the system successfully detects up to 30-40+ overlapping vehicles in dense traffic frames while maintaining a processing latency of under 1 second.

## 🚧 Challenges Faced
Occlusion in Dense Traffic: Vehicles blocking each other resulted in merged bounding boxes. This was mitigated by fine-tuning the model's IoU threshold to allow for overlapping detection.!
Latency vs. Accuracy: Balancing a high-resolution image size (imgsz=1280) for better accuracy against processing speed required careful decoupling of the detection and classification matrices.

## 🚀 Future Improvements
Multi-Camera Support (Stretch Goal): Expanding the async dashboard to handle parallel video streams from multiple CCTVs simultaneously.

Traffic Density Heatmaps: Storing bounding box coordinate data over time to generate thermal representations of traffic bottlenecks.

Temporal Tracking: Implementing DeepSORT to track identical vehicles across multiple frames to calculate average speed and detect speeding violations.

## 📸 Dashboard Screenshot
<img width="1704" height="985" alt="Screenshot 2026-08-21 at 11 13 26 PM" src="https://github.com/user-attachments/assets/fdece6ea-6501-46c7-8edd-2ac0cab8c6ec" />
