import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv('results-to-visual.csv')

# Построение графиков потерь
plt.figure(figsize=(12, 6))
plt.plot(data['epoch'], data['train/box_loss'], label='Train Box Loss')
plt.plot(data['epoch'], data['train/cls_loss'], label='Train Class Loss')
plt.plot(data['epoch'], data['train/dfl_loss'], label='Train DFL Loss')
plt.plot(data['epoch'], data['val/box_loss'], label='Validation Box Loss', linestyle='--')
plt.plot(data['epoch'], data['val/cls_loss'], label='Validation Class Loss', linestyle='--')
plt.plot(data['epoch'], data['val/dfl_loss'], label='Validation DFL Loss', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.grid()
plt.show()

# Построение графиков метрик
plt.figure(figsize=(12, 6))
plt.plot(data['epoch'], data['metrics/precision(B)'], label='Precision (B)')
plt.plot(data['epoch'], data['metrics/recall(B)'], label='Recall (B)')
plt.plot(data['epoch'], data['metrics/mAP50(B)'], label='mAP50 (B)')
plt.plot(data['epoch'], data['metrics/mAP50-95(B)'], label='mAP50-95 (B)')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.title('Metrics Over Epochs')
plt.legend()
plt.grid()
plt.show()

# Построение графика скорости обучения
plt.figure(figsize=(12, 6))
plt.plot(data['epoch'], data['lr/pg0'], label='Learning Rate pg0')
plt.plot(data['epoch'], data['lr/pg1'], label='Learning Rate pg1')
plt.plot(data['epoch'], data['lr/pg2'], label='Learning Rate pg2')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Over Epochs')
plt.legend()
plt.grid()
plt.show()