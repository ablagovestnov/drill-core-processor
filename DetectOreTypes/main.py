from ultralytics import YOLO

# Создаем модель (можно использовать предобученную или пустую)
model = YOLO('yolov8m.pt')  # Используем самую легкую модель YOLOv8n

# Тренировка модели
model.train(data='data.yaml', epochs=50, imgsz=1024, batch=16, project="DetectOreTypes", name="model")

# Сохранение обученной модели
model.export(format='onnx')  # Экспорт в ONNX для использования в других системах
