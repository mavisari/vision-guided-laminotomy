#this code was used to acquire the images with realsense camera

import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime

base_path = r"C:\Users\Tori\Desktop\Medical Robotics lab\immagini"
os.makedirs(base_path, exist_ok=True)

#realsense configuration 
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

#start of the stream 
profile = pipeline.start(config)
print("RealSense D456 avviata.")


device = profile.get_device()
color_sensor = device.query_sensors()[1]

#esposure and luminosity modulation 
if color_sensor.supports(rs.option.enable_auto_exposure):
    color_sensor.set_option(rs.option.enable_auto_exposure, 1)
    print("Auto-exposure attivata.")
else:
    color_sensor.set_option(rs.option.exposure, 400)
    print("Esposizione manuale impostata su 400 µs.")

if color_sensor.supports(rs.option.brightness):
    color_sensor.set_option(rs.option.brightness, 1)
    print("Luminosità aumentata.")

print("Premi 's' per scattare una foto, 'q' per uscire.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        color_image = cv2.filter2D(color_image, -1, kernel)

        cv2.imshow('Intel RealSense D456', color_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            filename = datetime.now().strftime("%Y%m%d_%H%M%S.jpg")
            filepath = os.path.join(base_path, filename)
            cv2.imwrite(filepath, color_image)
            print(f"Immagine salvata in: {filepath}")
        elif key == ord('q'):
            print("Uscita dal programma.")
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
