# Vision Guided Robotic Assistance for Spinal Interventions
Lumbar spinal stenosis is a degenerative condition that frequently affects the L4–L5 segment of the spine, often requiring decompression surgery such as laminotomy. Surgical precision in this context is critical: small deviations in tool positioning may compromise decompression quality or damage surrounding anatomical structures.

This project presents a vision-guided robotic assistance framework designed to support L4–L5 laminotomy through the integration of real-time computer vision, spatial calibration, and robotic simulation.

The proposed system combines:
- RGB-D image acquisition
- AI-based anatomical localization (YOLOv8)
- 3D centroid estimation
- Hand–eye calibration
- Coordinate transformation into the robot reference frame
- Simulation of robot-assisted motion planning
 
<p align="center"> <img src="workflow.png" width="800"> </p>

The entire workflow has been validated in a simulated environment using a KUKA LBR iiwa 7 manipulator. Images were acquired using an Intel RealSense D435 camera (acqimm.py), with resolution of 640x480. This resolution was used as a trade off between image detail and a fast processing for a realtime detection. These images formed the raw dataset. The overall dataset was created through Robotflow. 

## Object Detection 
The L4-L5 laminae region is detected using YOLOv8. 

```python
from ultralytics import YOLO
model = YOLO("yolov8s.pt")
model.train(data="data.yaml", epochs=50, batch=16)


