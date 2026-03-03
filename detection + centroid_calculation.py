import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import time
import os

# cartella screenshot
os.makedirs("captures", exist_ok=True)

detect_model = YOLO(r"C:\\Users\\Tori\\Desktop\\Medical Robotics lab\\progetto\\best_detect.pt")
print("Classi:", detect_model.names)

# RealSense 
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

print("RealSense avviata.")

# Calcolo centroide 2D 
def get_bbox_centroid(x1, y1, x2, y2):
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

# Calcolo profondità z usando profondità camera 
def get_depth_smooth(depth_frame, cx, cy, k=3):
    values = []
    H, W = depth_frame.get_height(), depth_frame.get_width()

    for dx in range(-k, k+1):
        for dy in range(-k, k+1):
            x = min(max(cx + dx, 0), W - 1)
            y = min(max(cy + dy, 0), H - 1)

            d = depth_frame.get_distance(x, y)
            if d > 0:
                values.append(d)

    if len(values) == 0:
        return None

    return np.median(values)

# KALMAN FILTER per stabilizzare le coordinate 
class Kalman3D:
    def __init__(self):
        self.x = np.zeros((3, 1))
        self.F = np.eye(3)
        self.P = np.eye(3) * 0.01
        self.Q = np.eye(3) * 0.0001
        self.R = np.eye(3) * 0.005

    def update(self, z):
        z = np.array(z).reshape((3, 1))

        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        S = P_pred + self.R
        K = P_pred @ np.linalg.inv(S)
        self.x = x_pred + K @ (z - x_pred)
        self.P = (np.eye(3) - K) @ P_pred

        return self.x.flatten()

kalman = Kalman3D()

# salvataggio coordinate 
last_Xf = last_Yf = last_Zf = None

try:
    while True:

        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        intr = depth_frame.profile.as_video_stream_profile().intrinsics

        # DETECTION 
        results = detect_model(frame, conf=0.3, verbose=False)
        detections = []

        for b in results[0].boxes:
            label = detect_model.names[int(b.cls[0])]
            conf = float(b.conf[0])
            if conf < 0.3:
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            detections.append((x1, y1, x2, y2, label, conf))

        if len(detections) > 0:

            x1, y1, x2, y2, label, conf = detections[0]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, "Laminae", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

            cx, cy = get_bbox_centroid(x1, y1, x2, y2)
            cv2.circle(frame, (cx, cy), 6, (0,255,255), -1)

            depth = get_depth_smooth(depth_frame, cx, cy)
            if depth is None:
                continue

            X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [cx, cy], depth)
            Xf, Yf, Zf = kalman.update([X, Y, Z])

            # SALVIAMO LE COORDINATE FILTRATE PER LO SCREENSHOT
            last_Xf, last_Yf, last_Zf = Xf, Yf, Zf

            cv2.putText(frame,
                        f"({Xf:.3f}, {Yf:.3f}, {Zf:.3f})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,255,255), 2)

        # SHOW
        cv2.imshow("Detection + 3D Kalman", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        # ========== SALVATAGGIO IMMAGINE + TXT ===============
        if key == ord('s'):
            timestamp = int(time.time())
            img_name = f"captures/capture_{timestamp}.png"
            txt_name = f"captures/capture_{timestamp}.txt"

            # Salva immagine
            cv2.imwrite(img_name, frame)
            print(f"Immagine salvata: {img_name}")

            # Salva SOLO coordinate Kalman
            if last_Xf is not None:
                with open(txt_name, "w") as f:
                    f.write(f"{last_Xf:.6f}, {last_Yf:.6f}, {last_Zf:.6f}\n")

                print(f"Coordinate Kalman salvate in: {txt_name}")
            else:
                print("Nessuna coordinata Kalman disponibile!")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
