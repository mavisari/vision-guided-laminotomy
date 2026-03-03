# this code was used for L4-L5 Detection - YOLOv8 Object Detection 

from ultralytics import YOLO
import os
from pathlib import Path

project_dir = Path(r"C:\Users\Tori\Desktop\Medical Robotics lab\progetto\dataset")

data_yaml = project_dir / "data.yaml"

if not data_yaml.exists():
    raise FileNotFoundError(f"File data.yaml non trovato in: {data_yaml}")

print(f"[INFO] Dataset YAML trovato: {data_yaml}")

model = YOLO("yolov8s.pt")

print("\n[INFO] Inizio addestramento...\n")

#model hyperparameters --> we tried different values (25 and 50) epochs (50 epochs achieved the best results)
model.train(
    data=str(data_yaml),       
    epochs=50,                
    imgsz=640,                 
    batch=16,                  
    name="L4L5_detection_v2i", 
    project=str(project_dir / "runs"),  
    exist_ok=True              
)

print("\n[INFO] Addestramento completato")

print("\n[INFO] Inizio validazione...\n")
metrics = model.val()
print("\n[INFO] Validazione completata")
print(metrics)

print("\n[INFO] Inizio testing...\n")
results = model.predict(
    source=str(project_dir / "test" / "images"),
    save=True,
    imgsz=640,
    conf=0.5  
)

print("\n[INFO] Testing completato")
print(f"Risultati salvati in: {project_dir / 'runs/detect/predict'}")

print("\n Fine script.")
