import cv2
from ultralytics import YOLO
import os
import pandas as pd
from tensorflow.keras.models import load_model, Model
import ocr_func as ocrf
import result_processing as rp

img_width, img_height = 128, 40
def detect_object(image_path, model):
    # Запуск детекции
    results = model(image_path)
    result = results[0]
    image = cv2.imread(image_path)

    # Извлекаем bounding box (на изображении только один объект)
    if result.boxes:
        box = result.boxes.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
        confidence = result.boxes.conf[0].cpu().numpy()
        return image, box, round(float(confidence), 3)
    else:
        return image, None, 0


def extract_roi(image, box):
    x1, y1, x2, y2 = map(int, box)
    roi = image[y1:y2, x1:x2]
    h, w = roi.shape[0:2]
    if h > w:
        roi = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)
    return roi

def preprocess_roi(image, target_size=(img_width, img_height)):
    """Загрузка и предобработка изображения"""
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    resized_image = cv2.resize(gray_image, target_size, interpolation=cv2.INTER_AREA)
    normalized_image = resized_image.astype('float32') / 255.0
    return normalized_image

def process_image(image_path, detection_model, rec_model):
    # детекция таблички
    image, box, detection_confidence = detect_object(image_path, detection_model)
    if box is None:
        return {
            "file_name": os.path.basename(image_path),
            "file_ext": os.path.splitext(image_path)[1],
            "text": "no detection",
            "detection_confidence": float(0),
            "recognition_confidence": float(0)
        }
    else:
        roi = extract_roi(image, box)
        # распознаём текст
        text, recognition_confidence = ocrf.predict_on_image(rec_model, roi)
        print(text)
        if text == '0':
            result_text = 'no text'
        else:
            result_text = text
        return {
            "file_name": os.path.basename(image_path),
            "file_ext": os.path.splitext(image_path)[1],
            "text": result_text,
            "detection_confidence": float(detection_confidence),
            "recognition_confidence": float(recognition_confidence)
        }

def process_folder(folder_path, prefix, det_model, rec_model, confidence_threshold):
    results = []

    image_extensions = {'.jpg', '.jpeg'}

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_extensions:
                result = process_image(file_path, det_model, rec_model)
                results.append(result)

    df = pd.DataFrame(results)
    print(df)

    result_df = rp.create_result_df(prefix, confidence_threshold, df).sort_values(by='confidence')
    print(result_df)

    rp.rename_files_from_dataframe(folder_path, result_df)

    rp.print_to_xls(result_df, confidence_threshold, folder_path)
    rp.print_to_csv(result_df, folder_path)

    return result_df


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    pd.set_option('display.max_columns', None)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Расположение моделей
    detection_model_path = os.path.join(BASE_DIR, 'models', 'label_detection_model.pt')
    ocr_model_path = os.path.join(BASE_DIR, 'models', 'best_ocr_model.keras')

    detection_model = YOLO(detection_model_path)
    ocr_loaded_model = load_model(ocr_model_path,
                           custom_objects={'ctc_lambda_func': ocrf.ctc_lambda_func},
                           safe_mode=False)
    # модель для предсказаний (без CTC-слоя)
    input_img = ocr_loaded_model.get_layer('input_image').output
    y_pred = ocr_loaded_model.get_layer('output').output
    recognition_model = Model(inputs=input_img, outputs=y_pred)

    folder_path = input("Введите путь до папки с изображениями: ")
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Папка не найдена: {folder_path}")
    prefix = input('Введите префикс (новые имена файлов будут в формате ''префиксимяфайла''. '
                   'Если префикс не требуется, нажмите enter.')

    process_folder(folder_path, prefix, detection_model, recognition_model, 0.6)

    print('Обработка завершена! Нажмите любую клавишу.')
    input()
