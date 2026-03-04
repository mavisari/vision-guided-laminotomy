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
 
<p align="center"> <img src="images/workflow.png" width="500"> </p>
<p align="center">
  <em>Figure 1 – Workflow.</em>
</p>

The entire workflow has been validated in a simulated environment using a KUKA LBR iiwa 7 manipulator. Images were acquired using an Intel RealSense D435 camera (acqimm.py), with resolution of 640x480. This resolution was used as a trade off between image detail and a fast processing for a realtime detection. These images formed the raw dataset. The overall dataset was created through Robotflow. 

## Object Detection and Centroid Calculation
The L4-L5 laminae region is detected using YOLOv8.

```python
from ultralytics import YOLO
model = YOLO("yolov8s.pt")
model.train(data="data.yaml", epochs=50, batch=16)
```

After detection:
1. Bounding box center is computed
2. Pixel coordinates are deprojected using depth
3. 3D centroid is stabilized

<p align="center"> <img src="images/centroid_calculation.png" width="500"> </p>
<p align="center">
  <em>Figure 2 – Centroid Calculation.</em>
</p>

As a result, a realtime detection with centroid calculation is obtained. Stability is given by a Kalman filter application.

<p align="center"> <img src="images/objectdetection.PNG" width="500"> </p>
<p align="center">
  <em>Figure 3 – Realtime detection and centroid calcuation.</em>
</p>

## Hand-eye Calibration
Camera Intrinsic Calibration is perfomed usign OpenCV and images with reprojection error > 0.50 px were discarded. The rigid trasformation between camera and end effector is estimated solving:

$$
A_i X = X B_i
$$

where:

- $A_i$ represents the motion of the end-effector in the base frame  
- $B_i$ represents the motion of the calibration target in the camera frame  
- $X$ is the unknown camera-to-end-effector transformation

<table align="center">
  <tr>
    <td align="center">
      <img src="images/transformation chain .png" width="450"><br>
      <em>Transformation chain from camera to robot base frame</em>
    </td>
    <td align="center">
      <img src="images/imm_iiwa7.png" width="450"><br>
      <em>KUKA LBR iiwa 7 simulation environment</em>
    </td>
  </tr>
</table>

## Handeye Calibration Validation 
Validation was performed by:
- Reconstructing checkerboard 3D points
- Comparing them with ground-truth robot frame points
- Computing Euclidean distance
The quantitative results are mean error along axis and rmse.
<table align="center">
  <tr>
    <td align="center">
      <img src="images/validation_method.png" width="300"><br>
      <em>Hand–eye validation methodology</em>
    </td>
    <td align="center">
      <img src="images/validation.png" width="300"><br>
      <em>Validation results in RViz</em>
    </td>
  </tr>
</table>

## Robot simulation (ROS2 + Movelt2)
The simulation is performed using:
- ROS2 Humble
- MoveIt2
- lbr-stack
- RViz
Motion strategy consists in a planar traslation in xy followed by a perpendicular descent along Z.

## Limitations 
- Depth instability at close range
- Systematic Z-axis calibration error
- Simulation-only validation
- Static centroid assumption

## Future Work
- Improved depth-axis excitation
- Closed-lopp visual servoing
- Respiratory motion compensation
- Real robot deployment 

# Installation 
## Python Dependencies 
```bash
pip install ultralytics
pip install opencv-python
pip install numpy
pip install pyrealsense2
pip install pandas
```
## ROS2 SetUp 
Install: 
- ROS2 Humble
- Movelt2
- lbr-stack
  
Then build:

```bash
colcon build
source install/setup.bash
```
# Authors 
Medical Robotics and Technologies for computer aided surgery laboratory 2025 - 2026
Elena Blaco, Beatrice Andrea Mantegari, Maria Vittoria Sari 
