import cv2
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import layers, models, regularizers, backend as K
import os
import albumentations as A
from collections import Counter
import matplotlib.pyplot as plt

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
    # text — строка
    if not isinstance(text, str):
        text = str(text)
    # каждый символ в индекс
    return [char_to_index[char] for char in text]

def load_and_preprocess_image(image_path, target_size=(img_width, img_height)):
    """Загрузка и предобработка изображения"""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")
    h, w = image.shape[0:2]

    if h > w:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, target_size, interpolation=cv2.INTER_AREA)
    normalized_image = resized_image.astype('float32') / 255.0
    return normalized_image

# Аугментации
augmentation_transform = A.Compose([
    A.GaussianBlur(blur_limit=(1, 2), p=0.1),  # уменьшаем интенсивность
    A.RandomBrightnessContrast(
        brightness_limit=0.1,
        contrast_limit=0.1,
        p=0.2
    ),
    A.Affine(
        translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
        scale=(0.98, 1.02),
        rotate=(-2, 2),
        p=0.15
    ),
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

        #  поворот на 180°
        if add_180_rotation:
            img_uint8 = (img * 255).astype(np.uint8)

            rotated_180 = cv2.rotate(img_uint8, cv2.ROTATE_180)
            rotated_float = rotated_180.astype(np.float32) / 255.0

            augmented_images.append(rotated_float)
            augmented_texts.append(text)

    return augmented_images, augmented_texts

def balance_dataset(X_train, y_train, min_samples=30):
    """Балансировка классов - oversampling редких классов"""
    class_counts = Counter(y_train)
    X_balanced, y_balanced = [], []

    for x, y in zip(X_train, y_train):
        X_balanced.append(x)
        y_balanced.append(y)

        # Если класс редкий, добавляем дополнительные копии
        if class_counts[y] < min_samples:
            # Добавляем 1–2 дополнительные копии для редких классов
            aug_factor = min(2, max(1, min_samples // class_counts[y]))
            for _ in range(aug_factor):
                X_balanced.append(x)
                y_balanced.append(y)
    return np.array(X_balanced), np.array(y_balanced)

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    labels = K.cast(labels, 'int32')
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def create_ocr_model(img_width, img_height, n_classes=11):
    input_img = layers.Input(shape=(img_height, img_width, 1), name='input_image')
    labels = layers.Input(shape=[None], dtype='int32', name='labels')
    input_length = layers.Input(shape=[1], dtype='int32', name='input_length')
    label_length = layers.Input(shape=[1], dtype='int32', name='label_length')

    # CNN часть с L2‑регуляризацией и Dropout
    x = layers.Conv2D(
        32,
        (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=regularizers.l2(5e-4)
    )(input_img)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  

    x = layers.Conv2D(
        64,
        (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=regularizers.l2(5e-4)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x) 

    x = layers.Conv2D(
        128,
        (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=regularizers.l2(5e-4)
    )(x)  # (None, 37, 128, 128)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    # размеры после CNN
    new_height = x.shape[1]
    new_width = x.shape[2]   # 128

    # Reshape для преобразования в последовательность
    x = layers.Reshape(target_shape=(new_width, new_height * 128))(x) 

    # Нормализация и Dropout перед LSTM
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)  #

    # LSTM с регуляризацией
    x = layers.Bidirectional(layers.LSTM(
        128,
        return_sequences=True,
        dropout=0.5,
        recurrent_dropout=0.4,
        kernel_regularizer=regularizers.l2(1e-3),
        recurrent_regularizer=regularizers.l2(5e-4)
    ))(x)  # (None, 128, 256)

    x = layers.Bidirectional(layers.LSTM(
        64,
        return_sequences=True,
        dropout=0.5,
        recurrent_dropout=0.4,
        kernel_regularizer=regularizers.l2(1e-3),
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


# Загрузка данных
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

# в numpy массивы
X_original = np.array(images)
y_original = np.array(texts)  # Сохраняем строки

# разделение выборок
X_train_prep, X_test, y_train_prep, y_test = train_test_split(
    X_original,
    y_original,  # y_train и y_test — строки
    test_size=0.2,
    random_state=42#,
    #stratify=y_original
)
print('train_prep', X_train_prep.shape, y_train_prep.shape)
# аугментация  тренировочной выборки
X_aug_list, y_aug_list = apply_augmentation(
    X_train_prep,
    y_train_prep,
    augmentation_transform,
    factor=4,  # 4 копии каждого изображения
    add_180_rotation=True
)

X_aug = np.array(X_aug_list)
y_aug = np.array(y_aug_list)  # y_aug — строки
print('aug', X_aug.shape, y_aug.shape)

# балансировка классов
X_aug_balanced, y_aug_balanced = balance_dataset(X_aug, y_aug)
print('aug_balanced', X_aug_balanced.shape, y_aug_balanced.shape)

X_train = X_aug_balanced
y_train = y_aug_balanced
print('train', X_train.shape, y_train.shape)

# подготовка данных для CTC loss — преобразуем тексты в индексы после разделения
X_train_ctc, y_train_labels, train_input_lengths, train_label_lengths = prepare_ctc_inputs(
    X_train, y_train, img_width, img_height
)
X_test_ctc, y_test_labels, test_input_lengths, test_label_lengths = prepare_ctc_inputs(
    X_test, y_test, img_width, img_height
)

# cоздание модели

print(y_train_labels.shape, y_test_labels.shape)
model = create_ocr_model(img_width, img_height, n_classes)

# компиляция модели
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss={'ctc': lambda y_true, y_pred: y_pred}
)

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        'best_ocr_model.keras',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
]

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



