# Technical Documentation: NeuroTraffic Ops

## 1. System Architecture & Workflow
NeuroTraffic Ops implements a decoupled, dual-node computer vision architecture designed for real-time urban surveillance telemetry:
* **Incision Node (YOLOv8 Core):** Ingests raw high-resolution street images (`imgsz=1280`) to isolate spatial regions of interest and generate precise bounding boxes for vehicles.
* **Classification Router:** Extracts cropped bounding box tensors from the primary node and passes them through a secondary deep classification network to categorize them into exact classes (Car, Motorcycle, Bus, Truck).

## 2. Codebase Structure
* `app.py`: The main NiceGUI asynchronous server handling frontend rendering, state management, and backend inference orchestration.
* `requirements.txt`: Contains pinned package versions for Ultralytics, OpenCV, NumPy, and NiceGUI.
* `yolov8s.pt`: Pre-trained YOLO weights optimized for spatial object detection.
* `vehicle_model.pt`: Specialized secondary classification weights trained on the mandatory dataset.

## 3. Performance & Optimization
* **Inference Latency:** Optimized to execute full-frame detection and classification cycles in under 1 second.
* **Overlap Handling:** Configured with custom Intersection over Union (IoU) thresholds to ensure high accuracy and prevent missed detections in dense, congested traffic conditions.
