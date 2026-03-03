## THIS CODE TAKES MULTIPLE IMAGES & POSES AND FILTERS THEM ##
# Minimal modification version

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import cv2
import pyrealsense2 as rs
import os
import csv
import numpy as np
import time

# =======================
# Configuration
# =======================
save_dir = "validation22dec"
os.makedirs(save_dir, exist_ok=True)
csv_file = os.path.join(save_dir, "poses.csv")

N_SAMPLES = 20     # << Number of samples to average per capture
SAMPLE_INTERVAL = 0.03  # seconds between samples

# Init RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)

# =======================
# ROS2 Node to get latest EE pose
# =======================
latest_pose = None

class PoseListener(Node):
    def __init__(self):
        super().__init__('pose_listener')
        self.subscription = self.create_subscription(
            PoseStamped,
            '/lbr/state/pose',
            self.pose_callback,
            10
        )

    def pose_callback(self, msg):
        global latest_pose
        latest_pose = msg

# =======================
# Main loop
# =======================
rclpy.init()
pose_listener = PoseListener()
pose_counter = 1
# CSV setup
csv_fields = ['image_filename', 'px', 'py', 'pz', 'qx', 'qy', 'qz', 'qw']
csv_writer = open(csv_file, 'w', newline='')
writer = csv.DictWriter(csv_writer, fieldnames=csv_fields)
writer.writeheader()

try:
    print("Press SPACE to capture filtered pose, ESC to quit...")
    while True:

        rclpy.spin_once(pose_listener, timeout_sec=0)

        # Get RealSense frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        color_image = np.asanyarray(color_frame.get_data())

        # Show image
        cv2.imshow("RealSense RGB", color_image)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break

        elif key == 32:  # SPACE
            if latest_pose is None:
                print("No pose received yet, wait a moment...")
                continue

            print(f"\n--- Capturing {N_SAMPLES} samples for filtering ---")
            poses = []

            # COLLECT MULTIPLE SAMPLES
            for i in range(N_SAMPLES):
                rclpy.spin_once(pose_listener, timeout_sec=0)

                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                color_image = np.asanyarray(color_frame.get_data())

                if latest_pose is not None:
                    poses.append([
                        latest_pose.pose.position.x,
                        latest_pose.pose.position.y,
                        latest_pose.pose.position.z,
                        latest_pose.pose.orientation.x,
                        latest_pose.pose.orientation.y,
                        latest_pose.pose.orientation.z,
                        latest_pose.pose.orientation.w,
                    ])

                time.sleep(SAMPLE_INTERVAL)

            poses = np.array(poses)
            mean_pose = poses.mean(axis=0)

            # Save last image
            img_filename = f"pose_{pose_counter:03d}.png"
            img_path = os.path.join(save_dir, img_filename)
            cv2.imwrite(img_path, color_image)

            # Save mean pose to CSV
            pose_data = {
                'image_filename': img_filename,
                'px': mean_pose[0],
                'py': mean_pose[1],
                'pz': mean_pose[2],
                'qx': mean_pose[3],
                'qy': mean_pose[4],
                'qz': mean_pose[5],
                'qw': mean_pose[6]
            }
            writer.writerow(pose_data)
            csv_writer.flush()

            print(f"✓ Saved filtered pose for {img_filename}")
            pose_counter += 1

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    pose_listener.destroy_node()
    rclpy.shutdown()
    csv_writer.close()

