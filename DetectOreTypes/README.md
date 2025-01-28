# Результаты обучения модели

В этом документе представлены результаты обучения модели на основе данных из файла `results.csv`.

## Обзор обучения

Модель обучалась в течение 69 эпох. Ниже приведены ключевые метрики и потери, которые были зафиксированы в процессе обучения.

## Графики и анализ

### 1. Потери (Losses)

#### Тренировочные потери:
- **Train Box Loss**: Потери, связанные с предсказанием ограничивающих рамок (bounding boxes). Значение уменьшается с увеличением количества эпох, что указывает на улучшение точности предсказания рамок.
- **Train Class Loss**: Потери, связанные с классификацией объектов. Также наблюдается снижение, что говорит о улучшении классификации.
- **Train DFL Loss**: Потери, связанные с распределением вероятностей. Уменьшение этого значения указывает на улучшение предсказания распределения.

#### Валидационные потери:
- **Validation Box Loss**, **Validation Class Loss**, **Validation DFL Loss**: Аналогичные потери для валидационного набора данных. Их уменьшение свидетельствует о том, что модель не переобучается и хорошо обобщает данные.

### 2. Метрики

- **Precision (B)**: Точность (precision) для класса B. В целом, точность колеблется, но в конце обучения достигает значений около 0.5-0.6.
- **Recall (B)**: Полнота (recall) для класса B. Значение recall также колеблется, но в конце обучения достигает значений около 0.4-0.5.
- **mAP50 (B)**: Средняя точность (mean Average Precision) при IoU=0.5. Значение mAP50 увеличивается с течением времени, достигая значений около 0.4-0.45.
- **mAP50-95 (B)**: Средняя точность при различных порогах IoU (от 0.5 до 0.95). Это значение также увеличивается, но остается ниже, чем mAP50.

### 3. Скорость обучения (Learning Rate)

- **Learning Rate pg0, pg1, pg2**: Скорость обучения для разных групп параметров. Значение learning rate уменьшается с течением времени, что является стандартной практикой для обеспечения более стабильного обучения на поздних этапах.

## Выводы

- Модель демонстрирует устойчивое улучшение по всем метрикам и потерям на протяжении всего обучения.
- Потери на тренировочном и валидационном наборах данных уменьшаются, что указывает на то, что модель успешно обучается и не переобучается.
- Значения mAP50 и mAP50-95 увеличиваются, что свидетельствует о улучшении качества детекции объектов.
- Однако, значения precision и recall остаются умеренными, что может указывать на то, что модель все еще может пропускать некоторые объекты или ошибаться в их классификации.

## Рекомендации

- Если точность и полнота остаются на низком уровне, можно рассмотреть возможность увеличения количества данных, улучшения аннотаций или использования более сложной архитектуры модели.
- Также можно попробовать настроить гиперпараметры, такие как learning rate, или использовать методы аугментации данных для улучшения обобщающей способности модели.

## Графики

Для визуализации результатов обучения были построены следующие графики:

1. **Графики потерь**:
   - Показывают, как уменьшаются потери на тренировочном и валидационном наборах данных.

2. **Графики метрик**:
   - Показывают, как изменяются точность, полнота и средняя точность (mAP) с каждой эпохой.

3. **График скорости обучения**:
   - Показывает, как изменяется скорость обучения с каждой эпохой.

Эти графики помогли визуализировать процесс обучения и сделать выводы о том, как модель улучшается с каждой эпохой.