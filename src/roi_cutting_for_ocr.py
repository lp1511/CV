import cv2
from ultralytics import YOLO
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def detect_object(image_path):
    # Запуск детекции
    results = detection_model(image_path)
    result = results[0]
    image = cv2.imread(image_path)
    # Извлекаем bounding box (на изображении только один объект)
    if result.boxes:
        box = result.boxes.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
        confidence = result.boxes.conf[0].cpu().numpy()
        return image, box, confidence
    else:
        return image, None, 0


def extract_roi(image, box):
    x1, y1, x2, y2 = map(int, box)
    roi = image[y1:y2, x1:x2]
    return roi


def process_image(folder_path, image_path):
    # детекция таблички
    image, box, detection_confidence = detect_object(image_path)
    file_name = os.path.basename(image_path)
    output_path = folder_path + '/output/' + file_name

    if box is None:
        return {"image": file_name, "detection_confidence": 0}
    else:
        # предобработка изображения
        x1, y1, x2, y2 = map(int, box)
        roi = image[y1:y2, x1:x2]
        h, w = roi.shape[0:2]

        if h > w:
            roi = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(output_path, roi)


        return {
            "file_name": file_name,
            "detection_confidence": float(detection_confidence),
        }


def process_folder(folder_path):
    results = []

    image_extensions = {'.jpg', '.jpeg'}

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_extensions:
                result = process_image(folder_path, file_path)

                results.append(result)

    df = pd.DataFrame(results)

    return df


detection_model_path = os.path.join(BASE_DIR, 'models', 'label_detection_model.pt')
detection_model = YOLO(detection_model_path)

# путь к папке с исходными изображениями
folder_path = 'D:/datasets/data/data_all'
process_folder(folder_path)
