from ultralytics import YOLO

# Загрузить обученную модель
model = YOLO('yolov8m/weights/best/weights/best.pt')

# Проведение предсказаний на изображении
results = model.predict(source='../ProcessorData/drill_core_samples/UNK/U-DD-156/Core_Fotos_U-DD-156/U-DD-156_BOX(18) 46.7-49.5.JPG', save=True, imgsz=1024)

