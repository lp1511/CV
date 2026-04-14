import cv2
import numpy as np
from tensorflow.keras import backend as K


# Параметры (как при обучении)
img_width, img_height = 128, 40
char_list = '0123456789'
n_classes = 11  # 0–9 + blank
blank_index = 10

# Словарь: индекс → символ для декодирования
index_to_char = {idx: char for idx, char in enumerate(char_list)}
index_to_char[blank_index] = ''  # blank-символ


def preprocess_image(image, target_size=(img_width, img_height)):
    """Загрузка и предобработка изображения"""
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    resized_image = cv2.resize(gray_image, target_size, interpolation=cv2.INTER_AREA)
    normalized_image = resized_image.astype('float32') / 255.0
    return normalized_image

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    labels = K.cast(labels, 'int32')
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def ctc_decode_single_prediction(prediction, input_length, blank_index=10):
    """
    Декодирование предсказания для одного изображения
    """
    # Обрезаем до реальной длины
    pred_slice = prediction[:input_length]

    # Greedy декодирование
    most_likely = np.argmax(pred_slice, axis=-1)
    # Удаляем повторяющиеся символы и blank
    cleaned = []
    prev_class = -1
    for cls in most_likely:
        if cls != prev_class and cls != blank_index:
            cleaned.append(cls)
        prev_class = cls
    decoded = cleaned

    return decoded

def calculate_confidence(prediction_sequence, decoded_indices, blank_index=10):
    """
    Расчёт уверенности распознавания для декодированной последовательности.

    Args:
        prediction_sequence: предсказанные вероятности (T, n_classes)
        decoded_indices: декодированные индексы символов
        blank_index: индекс blank-символа

    Returns:
        confidence: средняя уверенность для распознанных символов
    """
    if not decoded_indices:
        return 0.0

    # Находим позиции, где были выбраны распознанные символы
    positions = []
    current_pos = 0

    for symbol_idx in decoded_indices:
        # Ищем первую позицию после current_pos, где символ имеет максимальный вес
        # и не является повторением предыдущего символа
        found = False
        for pos in range(current_pos, len(prediction_sequence)):
            max_prob_idx = np.argmax(prediction_sequence[pos])
            if (max_prob_idx == symbol_idx and
                (not positions or max_prob_idx != np.argmax(prediction_sequence[positions[-1]]))):
                positions.append(pos)
                current_pos = pos + 1
                found = True
                break
        if not found:
            # Если символ не найден, берём позицию с максимальной вероятностью этого символа
            symbol_probs = prediction_sequence[:, symbol_idx]
            best_pos = np.argmax(symbol_probs)
            positions.append(best_pos)
            current_pos = best_pos + 1

    # Рассчитываем уверенность как среднее значение вероятностей для выбранных позиций
    confidences = [prediction_sequence[pos, symbol_idx]
                   for pos, symbol_idx in zip(positions, decoded_indices)]
    confidence = np.mean(confidences)

    return confidence

def predict_on_image(model, image):
    """ Предсказание на конкретном изображении"""
    # Загружаем и предобрабатываем изображение
    processed_image = preprocess_image(image)

    # Добавляем размерности для batch и channels
    input_image = np.expand_dims(processed_image, axis=0)  # batch dimension
    input_image = np.expand_dims(input_image, axis=-1)  # channels dimension

    # Делаем предсказание
    prediction = model.predict(input_image, verbose=0)

    # Определяем реальную длину последовательности (после CNN)
    input_length = img_width // 4  # или точное значение после CNN

    # Декодируем предсказание
    decoded_indices = ctc_decode_single_prediction(
        prediction[0],  # берём первый элемент из batch
        input_length,
        blank_index
    )

    # Рассчитываем уверенность распознавания
    confidence = calculate_confidence(
        prediction[0],
        decoded_indices,
        blank_index
    )

    # Конвертируем индексы в строку
    predicted_text = ''.join([index_to_char[idx] for idx in decoded_indices if idx != blank_index])

    return predicted_text, confidence
