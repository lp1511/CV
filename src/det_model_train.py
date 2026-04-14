from ultralytics import YOLO
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = YOLO('yolov8n.pt')
results = model.train(
    data= os.path.join(BASE_DIR, 'detection_data','dataset.yaml'),
    epochs=50,
    imgsz=640,
    batch=8,
    save=True,         # Сохранение моделей
    verbose=True
)

detection_model_path = os.path.join(BASE_DIR, 'models', 'label_detection_model.pt')
model.save(detection_model_path)
