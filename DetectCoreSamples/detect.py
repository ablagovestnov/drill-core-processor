from ultralytics import YOLO

# Загрузить обученную модель
model = YOLO('../runs/detect/train/weights/best.pt')

# Проведение предсказаний на изображении
results = model.predict(source='../ProcessorData/input/S-DD-200_BOX(30)82.3-85.2.JPG', save=True, imgsz=1024)

# Визуализация результатов
results.show()
