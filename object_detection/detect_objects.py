from ultralytics import YOLO
import argparse
import os

# Константы
MODEL_NAME = "yolo11n-seg.pt"

def main():
    """Детекция объектов на изображении"""
    parser = argparse.ArgumentParser(description='Детектирование объектов с помощью YOLO')
    parser.add_argument('--image', required=True, help='Путь к входному изображению')
    parser.add_argument('--class', required=True, help='Класс для поиска (через запятую)')
    parser.add_argument('--out_name', required=True, help='Имя выходного изображения')
    
    args = parser.parse_args()
    
    # Загружаем модель
    model = YOLO(MODEL_NAME)
    
    # Детектируем объекты
    results = model(args.image)
    
    # Сохраняем результат
    results[0].save(args.out_name)
    
    # Считаем объекты нужного класса
    target_classes = [cls.strip() for cls in getattr(args, 'class').split(',')]
    
    for target_class in target_classes:
        count = sum(1 for box in results[0].boxes if results[0].names[int(box.cls)] == target_class)
        print(f"Найдено {count} объектов класса '{target_class}'")

if __name__ == "__main__":
    main()