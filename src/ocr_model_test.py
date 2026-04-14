import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from Levenshtein import distance as levenshtein_distance
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Параметры
char_list = '0123456789'
blank_index = 10
img_width, img_height = 128, 40
n_classes = 11  # 0–9 + blank

# Словарь: символ → индекс
char_to_index = {char: idx for idx, char in enumerate(char_list)}
# Добавляем blank-символ
char_to_index[''] = blank_index

def text_to_labels(text):
    # Гарантируем, что text — строка
    if not isinstance(text, str):
        text = str(text)
    # Преобразуем каждый символ в индекс
    return [char_to_index[char] for char in text]

def load_and_preprocess_image(image_path, target_size=(img_width, img_height)):
    """Загрузка и предобработка изображения"""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, target_size, interpolation=cv2.INTER_AREA)
    normalized_image = resized_image.astype('float32') / 255.0
    return normalized_image

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    labels = K.cast(labels, 'int32')
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def prepare_ctc_inputs(images, texts, img_width, img_height):
    batch_size = len(images)
    X_data = np.zeros((batch_size, img_height, img_width, 1), dtype=np.float32)


    # Преобразуем тексты в строки
    texts_as_strings = [str(text) for text in texts]
    max_label_len = max(len(text) for text in texts_as_strings)

    labels = np.ones([batch_size, max_label_len], dtype=np.int32) * blank_index  # инициализируем blank-символом
    input_lengths = np.zeros([batch_size], dtype=np.int32)
    label_lengths = np.zeros([batch_size], dtype=np.int32)

    for i, (img, text) in enumerate(zip(images, texts_as_strings)):
        X_data[i] = np.expand_dims(img, axis=-1)

        # Преобразуем строку в последовательность индексов
        label = text_to_labels(text)
        actual_len = len(label)
        labels[i, :actual_len] = label
        label_lengths[i] = actual_len

        # Реальная длина последовательности после CNN
        input_lengths[i] = img_width // 4  # или точное значение после CNN

    return X_data, labels, input_lengths, label_lengths

def calculate_metrics(true_labels, predicted_labels):
    """Расчёт метрик качества OCR."""
    # Конвертируем индексы в строки для сравнения
    index_to_char = {v: k for k, v in char_to_index.items()}

    true_strings = []
    pred_strings = []

    for true_seq, pred_seq in zip(true_labels, predicted_labels):
        true_str = ''.join([index_to_char[idx] for idx in true_seq if idx != blank_index])
        pred_str = ''.join([index_to_char[idx] for idx in pred_seq if idx != blank_index])
        true_strings.append(true_str)
        pred_strings.append(pred_str)

    # accuracy
    exact_matches = sum(1 for t, p in zip(true_strings, pred_strings) if t == p)
    accuracy = exact_matches / len(true_strings)

    # Среднее расстояние Левенштейна
    avg_levenshtein = sum(
        levenshtein_distance(t, p) for t, p in zip(true_strings, pred_strings)
    ) / len(true_strings)

    print(f"Accuracy (точное совпадение): {accuracy:.4f}")
    print(f"Среднее расстояние Левенштейна: {avg_levenshtein:.4f}")

    return accuracy, avg_levenshtein

def calculate_prediction_confidence(prediction_sequence, decoded_indices, blank_index=10):
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

    confidences = []
    current_pos = 0

    for symbol_idx in decoded_indices:
        # Ищем позицию с максимальной вероятностью нужного символа после current_pos
        found = False
        for pos in range(current_pos, len(prediction_sequence)):
            if np.argmax(prediction_sequence[pos]) == symbol_idx:
                confidences.append(prediction_sequence[pos, symbol_idx])
                current_pos = pos + 1
                found = True
                break

        if not found:
            # Если символ не найден в «правильном» порядке, берём максимальную вероятность этого символа
            symbol_probs = prediction_sequence[:, symbol_idx]
            best_pos = np.argmax(symbol_probs)
            confidences.append(symbol_probs[best_pos])

    # Средняя уверенность по всем символам
    confidence = np.mean(confidences)
    return confidence


def apply_length_penalty(logits, penalty=0.0):
    """
    Применяет штраф за длину
    Args:
        logits: логиты формы (T, n_classes)
        penalty: штраф за каждый временной шаг (положительный — штрафует длину)
    Returns:
        скорректированные логиты
    """
    T, n_classes = logits.shape
    length_correction = np.arange(T)[:, np.newaxis] * penalty
    return logits + length_correction


def ctc_decode_predictions(predictions, input_lengths, blank_index=10, beam_search=False, beam_width=10, length_penalty=0.0):
    """CTC‑декодирование с поддержкой Beam Search и расчётом уверенности."""
    decoded_sequences = []
    confidences = []

    for i, pred in enumerate(predictions):
        length = input_lengths[i]
        pred_slice = pred[:length]  # Определяем pred_slice до использования

        if beam_search:
            logits = np.log(pred_slice + 1e-8)

            # Применяем штраф за длину
            if length_penalty != 0.0:
                logits = apply_length_penalty(logits, length_penalty)

            logits_tf = tf.constant(logits[np.newaxis, ...])
            logits_tf = tf.transpose(logits_tf, perm=[1, 0, 2])

            decoded_sparse = tf.nn.ctc_beam_search_decoder(
                inputs=logits_tf,
                sequence_length=[length],
                beam_width=beam_width,
                top_paths=1
            )
            decoded_dense = tf.sparse.to_dense(decoded_sparse[0][0])
            decoded = decoded_dense.numpy()[0].tolist()
        else:
            # Greedy декодирование
            most_likely = np.argmax(pred_slice, axis=-1)
            cleaned = []
            prev_class = -1
            for cls in most_likely:
                if cls != prev_class and cls != blank_index:
                    cleaned.append(cls)
                prev_class = cls
            decoded = cleaned

        confidence = calculate_prediction_confidence(pred_slice, decoded, blank_index)
        decoded_sequences.append(decoded)
        confidences.append(confidence)

    return decoded_sequences, confidences
def analyze_errors(cm, class_names):
    """Анализирует наиболее частые ошибки классификации"""
    print("\nАНАЛИЗ ОШИБОК:")
    errors = []

    # Ищем ячейки вне диагонали с наибольшим количеством ошибок
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            if i != j and cm[i][j] > 0:
                errors.append((class_names[i], class_names[j], cm[i][j]))

    # Сортируем по количеству ошибок (убывание)
    errors.sort(key=lambda x: x[2], reverse=True)

    print("ТОП‑10 наиболее частых ошибок:")
    for true_class, pred_class, count in errors[:10]:
        print(f"Символ '{true_class}' ошибочно распознан как '{pred_class}': {count} раз")


def flatten_sequences(true_sequences, pred_sequences, blank_index=10):
    """
    Преобразует последовательности символов в плоские списки для confusion matrix.
    Удаляет blank‑символы из истинных меток.
    """
    y_true_flat = []
    y_pred_flat = []

    for true_seq, pred_seq in zip(true_sequences, pred_sequences):
        # Удаляем blank‑символы из истинной последовательности
        true_clean = [idx for idx in true_seq if idx != blank_index]

        # Дополняем предсказание до длины истинной последовательности или обрезаем
        if len(pred_seq) < len(true_clean):
            # Дополняем нулями (или другим padding‑символом)
            pred_padded = pred_seq + [0] * (len(true_clean) - len(pred_seq))
        elif len(pred_seq) > len(true_clean):
            # Обрезаем до длины истинной последовательности
            pred_padded = pred_seq[:len(true_clean)]
        else:
            pred_padded = pred_seq

        y_true_flat.extend(true_clean)
        y_pred_flat.extend(pred_padded)

    return y_true_flat, y_pred_flat

# Загрузка исходных данных
images = []
texts = []
labels_df = pd.read_csv('D:/datasets/data/data_all/output/label_list.csv')
image_folder = 'D:/datasets/data/data_all/output'

for _, row in labels_df.iterrows():
    filename = row['file_name']
    text = row['label2']
    image_path = os.path.join(image_folder, filename)
    try:
        processed_image = load_and_preprocess_image(image_path)
        images.append(processed_image)
        texts.append(text)
    except FileNotFoundError as e:
        print(f"Пропущено изображение: {e}")

# Конвертация в numpy
X_original = np.array(images)
y_original = np.array(texts)  # Сохраняем строки

# Разделение на обучающую и тестовую выборки
X_train_prep, X_test, y_train_prep, y_test = train_test_split(
    X_original,
    y_original,  # y_train и y_test — строки
    test_size=0.2,
    random_state=42#,
    #stratify=y_original
)

# Преобразуем тексты в индексы для test
X_test_ctc, y_test_labels, test_input_lengths, test_label_lengths = prepare_ctc_inputs(
    X_test, y_test, img_width, img_height
)

try:
    loaded_model = load_model('best_ocr_model.keras', custom_objects={'ctc_lambda_func': ctc_lambda_func}, safe_mode=False)
    print("Модель загружена успешно!")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    raise

# модель для предсказаний (без CTC-слоя)
input_img = loaded_model.get_layer('input_image').output
y_pred = loaded_model.get_layer('output').output
prediction_model = Model(inputs=input_img, outputs=y_pred)

# Предсказание на тестовой выборке
test_predictions = prediction_model.predict(X_test_ctc, verbose=0)

# Реальные длины последовательностей
test_input_lengths_actual = test_input_lengths

# Декодирование
decoded_beam, confidences_beam = ctc_decode_predictions(
    test_predictions, test_input_lengths_actual, beam_search=True)

decoded_greedy, confidences_greedy = ctc_decode_predictions(
    test_predictions, test_input_lengths_actual, beam_search=False)

# Расчёт метрик
print("\nМЕТРИКИ КАЧЕСТВА:")
accuracy, avg_levenshtein = calculate_metrics(y_test_labels, decoded_greedy)



class_names = [str(i) for i in range(10)] + ['blank']
# Преобразуем последовательности в плоские массивы
y_true_flat, y_pred_flat = flatten_sequences(y_test_labels, decoded_greedy, blank_index)
y_true_flat_filtered = [y for y in y_true_flat if y != blank_index]
y_pred_flat_filtered = [p for p, y in zip(y_pred_flat, y_true_flat) if y != blank_index]

labels_filtered = list(range(10))
target_names_filtered = [str(i) for i in range(10)]

# Рассчитываем confusion matrix
cm = confusion_matrix(y_true_flat, y_pred_flat)

# Визуализируем
plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='YlGnBu',
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title('Confusion Matrix для OCR‑модели (Greedy Decoding)')
plt.xlabel('Предсказанные символы')
plt.ylabel('Истинные символы')
plt.tight_layout()
plt.show()

# классификация
print("\nCLASSIFICATION REPORT (Greedy, без blank):")
print(classification_report(
    y_true_flat_filtered,
    y_pred_flat_filtered,
    labels=labels_filtered,
    target_names=target_names_filtered
))

# Анализ ошибок
analyze_errors(cm, class_names)

# Точность по символам
per_class_accuracy = np.diagonal(cm) / cm.sum(axis=1)
print("\nТОЧНОСТЬ ПО СИМВОЛАМ (Greedy):")
for i, acc in enumerate(per_class_accuracy):
    symbol = class_names[i]
    support = cm.sum(axis=1)[i]
    print(f"Символ '{symbol}' (n={support}): {acc:.4f} ({acc*100:.2f}%)")