from ultralytics import YOLO

# Загрузить обученную модель
model = YOLO('../runs/detect/train/weights/best.pt')

# Проведение предсказаний на изображении
results = model.predict(source='../ProcessorData/drill_core_samples/SRT/S-DD-217/Core_Fotos_S-DD-217/S-DD-217_BOX (07) 16.8-19.5.JPG', save=True, imgsz=1024)

# Визуализация результатов
print(results)
