import cv2
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import layers, models, regularizers, backend as K
from Levenshtein import distance as levenshtein_distance
import os
import albumentations as A
from collections import Counter
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model, Model


# Параметры
char_list = '0123456789'
blank_index = 10
img_width, img_height = 128, 40
n_classes = 11  # 0–9 + blank

# Создаём словарь: символ → индекс
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

# Аугментация с albumentations
augmentation_transform = A.Compose([
    A.GaussianBlur(blur_limit=(1, 3), p=0.2),
    A.MotionBlur(p=0.1),
    A.GaussNoise(var_limit=(10.0, 25.0), p=0.2),
    A.RandomBrightnessContrast(p=0.2),
    A.Affine(
        translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
        scale=(0.95, 1.05),
        rotate=(-3, 3),
        p=0.2
    ),
    A.GridDistortion(p=0.1)  # новая аугментация
])
def apply_augmentation(images, texts, transform, factor=1, add_180_rotation=False):
    """Применяет аугментацию к набору изображений"""
    augmented_images = []
    augmented_texts = []

    for img, text in zip(images, texts):
        for _ in range(factor):
            img_uint8 = (img * 255).astype(np.uint8)
            augmented = transform(image=img_uint8)['image']
            augmented_float = augmented.astype(np.float32) / 255.0
            augmented_images.append(augmented_float)
            augmented_texts.append(text)

        # Дополнительная аугментация: поворот на 180°
        if add_180_rotation:
            img_uint8 = (img * 255).astype(np.uint8)
            # Поворот на 180° через OpenCV
            rotated_180 = cv2.rotate(img_uint8, cv2.ROTATE_180)
            rotated_float = rotated_180.astype(np.float32) / 255.0

            augmented_images.append(rotated_float)
            augmented_texts.append(text)  # текст остаётся тем же

    return augmented_images, augmented_texts



def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # Убедимся, что метки имеют правильный тип
    labels = K.cast(labels, 'int32')
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def create_ocr_model(img_width, img_height, n_classes=11):
    input_img = layers.Input(shape=(img_height, img_width, 1), name='input_image')
    labels = layers.Input(shape=[None], dtype='int32', name='labels')
    input_length = layers.Input(shape=[1], dtype='int32', name='input_length')
    label_length = layers.Input(shape=[1], dtype='int32', name='label_length')

    # CNN часть с L2‑регуляризацией и увеличенным Dropout
    x = layers.Conv2D(
        32,
        (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=regularizers.l2(1e-4)
    )(input_img)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # (None, 75, 256, 32)

    x = layers.Conv2D(
        64,
        (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)  # Увеличили Dropout
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # (None, 37, 128, 64)

    x = layers.Conv2D(
        128,
        (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)  # (None, 37, 128, 128)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)  # Максимальный Dropout для глубокого слоя

    # Рассчитываем новые размеры после CNN
    new_height = x.shape[1]  # 37
    new_width = x.shape[2]   # 128

    # Reshape для преобразования в последовательность
    x = layers.Reshape(target_shape=(new_width, new_height * 128))(x)  # (None, 128, 4736)


    # Нормализация и Dropout перед LSTM
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)  # Увеличили с 0.3

    # LSTM часть с регуляризацией
    x = layers.Bidirectional(layers.LSTM(
        128,
        return_sequences=True,
        dropout=0.4,  # Увеличили с 0.25
        recurrent_dropout=0.3,  # Увеличили с 0.25
        kernel_regularizer=regularizers.l2(5e-4),
        recurrent_regularizer=regularizers.l2(1e-4)
    ))(x)  # (None, 128, 256)

    x = layers.Bidirectional(layers.LSTM(
        64,
        return_sequences=True,
        dropout=0.4,  # Увеличили с 0.25
        recurrent_dropout=0.3,  # Увеличили с 0.25
        kernel_regularizer=regularizers.l2(5e-4),
        recurrent_regularizer=regularizers.l2(5e-4)
    ))(x)   # (None, 128, 128)

    # TimeDistributed Dense с регуляризацией
    y_pred = layers.TimeDistributed(layers.Dense(
        n_classes,
        activation='softmax',
        kernel_regularizer=regularizers.l2(1e-4)
    ), name='output')(x)  # (None, 128, 11)

    # CTC слой
    loss_out = layers.Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    model = models.Model(
        inputs=[input_img, labels, input_length, label_length],
        outputs=loss_out
    )
    return model

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


def ctc_decode_predictions(predictions, input_lengths, blank_index=10, beam_search=False):
    """CTC‑декодирование с поддержкой Beam Search."""
    decoded_sequences = []


    for i, pred in enumerate(predictions):
        # Обрезаем до реальной длины
        length = input_lengths[i]
        pred_slice = pred[:length]


        if beam_search:
            # Beam Search декодирование
            logits = np.log(pred_slice + 1e-8)
            logits_tf = tf.constant(logits[np.newaxis, ...])  # добавляем batch dim
            logits_tf = tf.transpose(logits_tf, perm=[1, 0, 2])

            decoded_sparse = tf.nn.ctc_beam_search_decoder(
                inputs=logits_tf,
                sequence_length=[length],
                beam_width=10,
                top_paths=1
            )
            # Преобразуем EagerTensor в список через .numpy().tolist()
            decoded_dense = tf.sparse.to_dense(decoded_sparse[0][0])
            decoded = decoded_dense.numpy()[0].tolist()  # исправлено: .numpy() перед .tolist()
        else:
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

        decoded_sequences.append(decoded)

    return decoded_sequences


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

    # Точность (точное совпадение)
    exact_matches = sum(1 for t, p in zip(true_strings, pred_strings) if t == p)
    accuracy = exact_matches / len(true_strings)

    # Среднее расстояние Левенштейна
    avg_levenshtein = sum(
        levenshtein_distance(t, p) for t, p in zip(true_strings, pred_strings)
    ) / len(true_strings)

    print(f"Accuracy (точное совпадение): {accuracy:.4f}")
    print(f"Среднее расстояние Левенштейна: {avg_levenshtein:.4f}")

    return accuracy, avg_levenshtein

# 1. Загрузка исходных данных
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

# Конвертация в numpy массивы
X_original = np.array(images)
y_original = np.array(texts)  # Сохраняем строки

# 1. Загрузка исходных данных (без изменений)
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


# Анализ распределения классов (без изменений)
class_counts = Counter(y_original)
print(f"Всего уникальных классов: {len(class_counts)}")
print(f"Классы с 1 примером: {sum(1 for c in class_counts.values() if c == 1)}")

# 2. Разделение на обучающую и тестовую выборки
X_train_prep, X_test, y_train_prep, y_test = train_test_split(
    X_original,
    y_original,  # y_train и y_test — строки
    test_size=0.2,
    random_state=42#,
    #stratify=y_original
)

print(f"Размер обучающей выборки до аугментации: {X_train_prep.shape[0]}")
print(f"Размер тестовой выборки: {X_test.shape[0]}")

# 3. Аугментация ТОЛЬКО тренировочной выборки
print("Начинаем аугментацию тренировочной выборки...")
X_aug_list, y_aug_list = apply_augmentation(
    X_train_prep,
    y_train_prep,  # передаём строки из обучающей выборки
    augmentation_transform,
    factor=4,  # создаём 4 копии каждого изображения
    add_180_rotation=True
)

X_aug = np.array(X_aug_list)
y_aug = np.array(y_aug_list)  # y_aug — строки

print(f"Размер аугментированных данных: {X_aug.shape[0]}")

# 4. Объединение исходных тренировочных данных с аугментированными
X_train = np.concatenate([X_train_prep, X_aug], axis=0)
y_train = np.concatenate([y_train_prep, y_aug], axis=0)  # y_train_final — строки

print(f"Итоговый размер обучающей выборки после аугментации: {X_train.shape[0]}")
print(f"Размер тестовой выборки (без изменений): {X_test.shape[0]}")



# 5. Подготовка данных для CTC loss — преобразуем тексты в индексы ПОСЛЕ разделения
print("Подготовка данных для обучения...")

# Преобразуем тексты в индексы только для train
X_train_ctc, y_train_labels, train_input_lengths, train_label_lengths = prepare_ctc_inputs(
    X_train, y_train, img_width, img_height
)

# Преобразуем тексты в индексы для test
X_test_ctc, y_test_labels, test_input_lengths, test_label_lengths = prepare_ctc_inputs(
    X_test, y_test, img_width, img_height
)

y_train_labels_fixed = np.where(y_train_labels == -1, 10, y_train_labels)
y_test_labels_fixed = np.where(y_test_labels == -1, 10, y_test_labels)

y_train_labels = y_train_labels_fixed
y_test_labels = y_test_labels_fixed

# 6. Создание модели
print("Создание модели OCR...")
model = create_ocr_model(img_width, img_height, n_classes)

# Компиляция модели
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss={'ctc': lambda y_true, y_pred: y_pred}
)


# 7. Настройка callback-ов
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,  # ждём 10 эпох без улучшений
        restore_best_weights=True,  # восстанавливаем веса эпохи 25
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # уменьшаем LR в 2 раза
        patience=5,  # ждём 5 эпох без улучшений
        min_lr=1e-7,  # минимальный LR
        verbose=1
    ),
    ModelCheckpoint(
        'best_ocr_model.keras',  # новый формат
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
]

# 8. Обучение модели
print("Начало обучения...")
history = model.fit(
    x=[X_train_ctc, y_train_labels, train_input_lengths, train_label_lengths],
    y=np.zeros(len(X_train_ctc)),  # фиктивные метки для CTC
    batch_size=32,
    epochs=100,  # увеличиваем количество эпох
    validation_data=(
        [X_test_ctc, y_test_labels, test_input_lengths, test_label_lengths],
        np.zeros(len(X_test_ctc))
    ),
    callbacks=callbacks,
    verbose=1,
    shuffle=True  # перемешиваем данные между эпохами
)

#  Сохранение модели
model.save('best_ocr_model.keras')
print("Модель сохранена как 'best_ocr_model.keras'")


# Визуализация

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Проверка баланса классов
train_class_counts = Counter(y_train)
test_class_counts = Counter(y_test)
print("Баланс классов в train:", train_class_counts)
print("Баланс классов в test:", test_class_counts)

# Визуализация распределения val_loss
plt.plot(history.history['val_loss'], label='Val Loss')
plt.axhline(y=2.0057, color='r', linestyle='--', label='Best Val Loss')
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.legend()
plt.title('Динамика val_loss')
plt.show()


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

def ctc_decode_predictions(predictions, input_lengths, blank_index=10, beam_search=False):
    """CTC‑декодирование с поддержкой Beam Search и расчётом уверенности."""
    decoded_sequences = []
    confidences = []

    for i, pred in enumerate(predictions):
        # Обрезаем до реальной длины
        length = input_lengths[i]
        pred_slice = pred[:length]

        if beam_search:
            # Beam Search декодирование
            logits = np.log(pred_slice + 1e-8)
            logits_tf = tf.constant(logits[np.newaxis, ...])
            logits_tf = tf.transpose(logits_tf, perm=[1, 0, 2])

            decoded_sparse = tf.nn.ctc_beam_search_decoder(
                inputs=logits_tf,
                sequence_length=[length],
                beam_width=10,
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

        # Рассчитываем уверенность для этой последовательности
        confidence = calculate_prediction_confidence(pred_slice, decoded, blank_index)

        decoded_sequences.append(decoded)
        confidences.append(confidence)

    return decoded_sequences, confidences

def apply_length_penalty(logits, penalty=0.0):
    """
    Применяет штраф за длину к логитам.
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
        pred_slice = pred[:length]  # Определяем pred_slice ДО использования

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

def calculate_cer(decoded_sequences, true_labels):
    """Упрощённая функция расчёта CER. Замените на свою реализацию."""
    total_errors = 0
    total_chars = 0

    for decoded, true in zip(decoded_sequences, true_labels):
        # Здесь должна быть логика расчёта ошибок (замены, вставки, удаления)
        # Для примера считаем количество несовпадающих символов
        max_len = max(len(decoded), len(true))
        errors = sum(1 for a, b in zip(decoded, true) if a != b) + abs(len(decoded) - len(true))
        total_errors += errors
        total_chars += max_len

    return total_errors / total_chars if total_chars > 0 else 0.0

def evaluate_ocr_performance(predictions, true_labels, input_lengths, beam_widths, penalties):
    results = []
    for bw in beam_widths:
        for penalty in penalties:
            decoded, confidences = ctc_decode_predictions(
                predictions,
                input_lengths,
                beam_search=True,
                beam_width=bw,
                length_penalty=penalty
            )
            cer = calculate_cer(decoded, true_labels)
            avg_confidence = np.mean(confidences)
            results.append({
                'beam_width': bw,
                'length_penalty': penalty,
                'cer': cer,
                'avg_confidence': avg_confidence
            })
    return results


try:
    loaded_model = load_model('best_ocr_model.keras', custom_objects={'ctc_lambda_func': ctc_lambda_func}, safe_mode=False)
    print("Модель загружена успешно!")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    raise

# 2. Создаём модель для предсказаний (без CTC-слоя)
input_img = loaded_model.get_layer('input_image').output
y_pred = loaded_model.get_layer('output').output
prediction_model = Model(inputs=input_img, outputs=y_pred)
print("Модель для предсказаний создана!")

# 3. Предсказание на тестовой выборке
test_predictions = prediction_model.predict(X_test_ctc, verbose=0)

beam_widths = [5, 7, 10]
penalties = [-0.1, 0.0, 0.1]
results = evaluate_ocr_performance(test_predictions, y_test_labels, test_input_lengths, beam_widths, penalties)

# 5. Анализ результатов
for res in results:
    print(f"Beam width: {res['beam_width']}, Penalty: {res['length_penalty']:.2f} -> CER: {res['cer']:.4f}, Confidence: {res['avg_confidence']:.4f}")
    




print(f"Форма предсказаний: {test_predictions.shape}")

# 4. Диагностика: проверка сумм вероятностей
print("\nДИАГНОСТИКА:")
for i in range(min(3, len(test_predictions))):
    pred_seq = test_predictions[i]
    row_sums = np.sum(pred_seq, axis=1)
    print(f"Образец {i} - сумма вероятностей: {row_sums[:5].round(3)}")

# Дополнительная диагностика распределения
print("\nДЕТАЛЬНАЯ ДИАГНОСТИКА РАСПРЕДЕЛЕНИЯ ВЕРОЯТНОСТЕЙ:")
for i in range(min(3, len(test_predictions))):
    pred_seq = test_predictions[i]
    print(f"\nОбразец {i}:")
    for t in range(5):
        probs = pred_seq[t]
        print(f"  Временной шаг {t}: max={np.max(probs):.3f}, "
              f"min={np.min(probs):.3f}, sum={np.sum(probs):.3f}")

# 5. Реальные длины последовательностей
test_input_lengths_actual = test_input_lengths

# 6. Декодирование с расчётом уверенности
print("\nРЕЗУЛЬТАТЫ BEAM SEARCH ДЕКОДИРОВАНИЯ:")
try:
    decoded_beam, confidences_beam = ctc_decode_predictions(
        test_predictions, test_input_lengths_actual, beam_search=True
    )
    for i, (pred, true, conf) in enumerate(zip(decoded_beam, y_test_labels, confidences_beam)):
        true_list = [idx for idx in true if idx != blank_index]
        print(f"Образец {i}: декодировано {pred}, истинная метка {true_list}, "
              f"уверенность: {conf:.4f} ({conf*100:.2f}%)")
except Exception as e:
    print(f"Ошибка Beam Search: {e}")

print("\nРЕЗУЛЬТАТЫ GREEDY ДЕКОДИРОВАНИЯ:")
decoded_greedy, confidences_greedy = ctc_decode_predictions(
    test_predictions, test_input_lengths_actual, beam_search=False
)
for i, (pred, true, conf) in enumerate(zip(decoded_greedy, y_test_labels, confidences_greedy)):
    true_list = [idx for idx in true if idx != blank_index]
    print(f"Образец {i}: декодировано {pred}, истинная метка {true_list}, "
          f"уверенность: {conf:.4f} ({conf*100:.2f}%)")

# 7. Расчёт метрик с учётом уверенности
print("\nМЕТРИКИ КАЧЕСТВА:")
accuracy, avg_levenshtein = calculate_metrics(y_test_labels, decoded_greedy)

# Дополнительная метрика: средняя уверенность
avg_confidence_greedy = np.mean(confidences_greedy)
avg_confidence_beam = np.mean(confidences_beam) if 'confidences_beam' in locals() else None

print(f"\nДОПОЛНИТЕЛЬНЫЕ МЕТРИКИ:")
print(f"Средняя уверенность (Greedy): {avg_confidence_greedy:.4f} ({avg_confidence_greedy*100:.2f}%)")
if avg_confidence_beam is not None:
    print(f"Средняя уверенность (Beam Search): {avg_confidence_beam:.4f} ({avg_confidence_beam*100:.2f}%)")


fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.ravel()

for i in range(10):
    axes[i].imshow(X_test[i].squeeze(), cmap='gray')
    axes[i].set_title(f'Img {i+1}')
    axes[i].axis('off')

plt.suptitle('Первые 10 изображений обучающей выборки', fontsize=14)
plt.tight_layout()
plt.show()


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

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# После декодирования последовательностей (в вашем коде после шага 6)
class_names = [str(i) for i in range(10)] + ['blank']
# 1. Преобразуем последовательности в плоские массивы
y_true_flat, y_pred_flat = flatten_sequences(y_test_labels, decoded_greedy, blank_index)
y_true_flat_filtered = [y for y in y_true_flat if y != blank_index]
y_pred_flat_filtered = [p for p, y in zip(y_pred_flat, y_true_flat) if y != blank_index]

labels_filtered = list(range(10))
target_names_filtered = [str(i) for i in range(10)]

# 2. Рассчитываем confusion matrix
cm = confusion_matrix(y_true_flat, y_pred_flat)

# 3. Визуализируем
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

# 4. Отчёт по классификации
print("\nCLASSIFICATION REPORT (Greedy, без blank):")
print(classification_report(
    y_true_flat_filtered,
    y_pred_flat_filtered,
    labels=labels_filtered,
    target_names=target_names_filtered
))

# 5. Анализ ошибок
analyze_errors(cm, class_names)

# 6. Точность по символам
per_class_accuracy = np.diagonal(cm) / cm.sum(axis=1)
print("\nТОЧНОСТЬ ПО СИМВОЛАМ (Greedy):")
for i, acc in enumerate(per_class_accuracy):
    symbol = class_names[i]
    support = cm.sum(axis=1)[i]
    print(f"Символ '{symbol}' (n={support}): {acc:.4f} ({acc*100:.2f}%)")
