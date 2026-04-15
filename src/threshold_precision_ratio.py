import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score

df = pd.read_excel('D:/datasets/data/test_ocr/logs.xlsx')
true_labels = pd.read_csv('D:/datasets/data/test_ocr/label_list.csv')
df = df.merge(true_labels, on='file_name', how='left')

thresholds = np.arange(0.2, 1.0, 0.05)

# Создаем пустой список для хранения значений precision
precision_values = []

# Вычисляем precision для каждого порога
for threshold in thresholds:
    # Фильтруем данные по текущему порогу
    filtered_df = df[df['confidence'] >= threshold]

    # Если после фильтрации нет данных - пропускаем
    if filtered_df.empty:
        continue

    # Вычисляем precision как долю верных предсказаний
    precision = precision_score(
        filtered_df['label2'],
        filtered_df['text'],
        average='weighted'
    )

    precision_values.append(precision)

# Построение графика
plt.figure(figsize=(12, 6))
plt.plot(thresholds[:len(precision_values)], precision_values, marker='o')

plt.title('Зависимость Precision от порога уверенности')
plt.xlabel('Порог уверенности (confidence threshold)')
plt.ylabel('Precision')
plt.grid(True)
plt.xlim(0.2, 1.0)
plt.ylim(0.0, 1.1)
plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)  # Линия 100% точности
plt.axvline(x=0.8, color='g', linestyle='--', alpha=0.5)  # Рекомендуемый порог
plt.text(0.81, 0.95, 'Рекомендуемый порог', color='g')
plt.tight_layout()
plt.show()


print(df)