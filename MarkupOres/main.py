import pandas as pd
import os
import re
import random
from ultralytics import YOLO
import cv2

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
rock_classes_set = set()
rock_classes_array = []


def process_geological_table(file_path, skip_rows):
    """
    Processes a geological log Excel file to extract intervals and rock types.

    Parameters:
        file_path (str): Path to the Excel file.
        skip_rows (int): Number of rows to skip (header rows).

    Returns:
        list of dict: Cleaned list of intervals with rock types, structured as:
                      [{'start': float, 'end': float, 'rock': str}, ...]
    """
    # Load the Excel file, using the first sheet by index, skipping the specified number of rows
    df = pd.read_excel(file_path, sheet_name=0, skiprows=skip_rows)

    # Check if default columns contain valid rock type
    df_cleaned = df.rename(columns={
        df.columns[1]: 'End',  # Second column (index 1) -> 'End'
        df.columns[5]: 'Rock'  # Sixth column (index 5) -> 'Rock'
    })
    df_cleaned[['End', 'Rock']] = df_cleaned[['End', 'Rock']].dropna()
    df_cleaned['End'] = pd.to_numeric(df_cleaned['End'], errors='coerce')
    df_cleaned = df_cleaned.dropna(subset=['End'])

    if not all(df_cleaned['Rock'].astype(str).str.match(r'^[A-Za-z]+$')):
        print(f"Invalid rock type detected in file: {file_path}, switching to alternative columns.")
        df_cleaned = df.rename(columns={
            df.columns[3]: 'End',  # Fourth column (index 3) -> 'End'
            df.columns[7]: 'Rock'  # Eighth column (index 7) -> 'Rock'
        })
        df_cleaned[['End', 'Rock']] = df_cleaned[['End', 'Rock']].dropna()
        df_cleaned['End'] = pd.to_numeric(df_cleaned['End'], errors='coerce')
        df_cleaned = df_cleaned.dropna(subset=['End'])

    # Prepare the structured list of intervals with rock types
    result = []
    previous_end = 0.0  # Initialize the starting point

    for _, row in df_cleaned.iterrows():
        rock_type = str(row['Rock'])
        if not re.match(r'^[A-Za-z]+$', rock_type):
            print(f"Invalid rock type '{rock_type}' in file: {file_path}")
        result.append({
            'start': previous_end,
            'end': row['End'],
            'rock': rock_type  # Ensure rock type is always a string
        })
        previous_end = row['End']  # Update the starting point for the next interval
        rock_classes_set.add(rock_type)

    return result


def find_rock_by_interval(intervals, value):
    """
    Finds the rock type for a given depth value.

    Parameters:
        intervals (list of dict): List of intervals with rock types.
        value (float): The depth value to search for.

    Returns:
        str: Rock type if found, otherwise 'Unknown'.
    """
    for interval in intervals:
        if interval['start'] <= value < interval['end']:
            return interval['rock']
    return 'Unknown'


def process_all_logs(base_directory, skip_rows):
    """
    Processes all *_Log.xlsm files in a directory structure and extracts intervals.

    Parameters:
        base_directory (str): Base directory to search for log files.
        skip_rows (int): Number of rows to skip (header rows).

    Returns:
        dict: A dictionary where keys are file paths and values are interval data.
    """
    result = {}

    for root, _, files in os.walk(base_directory):
        for file in files:
            if file.endswith("_Log.xlsm"):
                file_path = os.path.join(root, file)
                try:
                    intervals = process_geological_table(file_path, skip_rows)
                    result[file_path] = intervals
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
    return result

def draw_boxes_on_image(image_path, original_boxes, annotated_boxes, output_directory, file_index):
    """
    Draws original and annotated bounding boxes with labels on the image and saves the result.
    Resizes images to 1024px on the longer side and saves YOLO annotations in appropriate directories.

    Parameters:
        image_path (str): Path to the input image.
        original_boxes (list of dict): List of original bounding boxes.
        annotated_boxes (list of dict): List of annotated bounding boxes with coordinates and labels.
        output_directory (str): Path to the directory where annotated images will be saved.
        file_index (int): Index of the file, used to determine whether it goes into 'train' or 'val'.
    """
    # Load image
    image = cv2.imread(image_path)
    original_height, original_width = image.shape[:2]

    # Determine resize scale
    long_side = 1024
    scale = long_side / max(original_width, original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Resize image
    resized_image = cv2.resize(image, (new_width, new_height))

    # Scale bounding boxes to resized image
    scale_x = new_width / original_width
    scale_y = new_height / original_height

    # Draw original bounding boxes in red
    # for box in original_boxes:
    #     x_min, y_min, x_max, y_max = int(box['x_min'] * scale_x), int(box['y_min'] * scale_y), int(
    #         box['x_max'] * scale_x), int(box['y_max'] * scale_y)
    #     cv2.rectangle(resized_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 4)

    # Prepare YOLO annotations
    yolo_annotations = []

    # Draw annotated bounding boxes in green with labels
    for box in annotated_boxes:
        x_min, y_min, x_max, y_max = int(box['x_min'] * scale_x), int(box['y_min'] * scale_y), int(
            box['x_max'] * scale_x), int(box['y_max'] * scale_y)
        rock_type = str(box.get('rock', 'Unknown'))  # Ensure rock type is always a string
        # start_depth = box.get('start_depth', '')
        # end_depth = box.get('end_depth', '')
        # label = f"{rock_type} ({start_depth}-{end_depth}m)"

        # Draw box and label on the image
        # cv2.rectangle(resized_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        # cv2.putText(resized_image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert bounding box to YOLO format
        # print(f'{rock_classes_array}')
        class_id = rock_classes_array.index(rock_type) if rock_classes_array.index(rock_type) else 9999  # Assign a class ID (default 0 if non-numeric)
        x_center = (x_min + x_max) / 2 / new_width
        y_center = (y_min + y_max) / 2 / new_height
        box_width = (x_max - x_min) / new_width
        box_height = (y_max - y_min) / new_height

        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

    # Determine if the file goes to train or val
    dataset_type = 'val' if file_index % 5 == 0 else 'train'

    # Ensure the output directories exist
    image_output_dir = f'../ore_markup/images/{dataset_type}'
    label_output_dir = f'../ore_markup/labels/{dataset_type}'
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(label_output_dir, exist_ok=True)

    # Save the resized and annotated image
    image_filename = os.path.basename(image_path).replace('.JPG', '_annotated.JPG')
    output_image_path = os.path.join(image_output_dir, image_filename)
    cv2.imwrite(output_image_path, resized_image)
    print(f"Resized and annotated image saved to {output_image_path}")

    # Save YOLO annotations
    yolo_annotation_path = os.path.join(label_output_dir, os.path.basename(image_path).replace('.JPG', '_annotated.txt'))
    with open(yolo_annotation_path, 'w') as yolo_file:
        yolo_file.write("\n".join(yolo_annotations))

    print(f"YOLO annotation saved to {yolo_annotation_path}")


def process_photos_for_drill_hole(file_path, intervals, debug=False):
    """
    Processes all core box photos for a given drill hole.

    Parameters:
        file_path (str): Path to the Excel log file.
        intervals (list of dict): List of intervals extracted from the log file.
        debug (bool): If True, process only 3 random photos per drill hole.

    Prints:
        Normalized results with file names, corresponding intervals, rock type breakdowns,
        and bounding boxes for detected core regions.
    """
    # Load the trained YOLO model
    model = YOLO('../runs/detect/train/weights/best.pt')

    # Derive drill hole name from file path
    drill_hole_name = os.path.basename(file_path).replace("_Log.xlsm", "")

    # Construct path to the photo directory
    photo_directory = os.path.join(os.path.dirname(file_path), f"Core_Fotos_{drill_hole_name}")

    if not os.path.exists(photo_directory):
        print(f"Photo directory {photo_directory} does not exist.")
        return

    # Create output directory for annotated images
    output_directory = os.path.join(os.path.dirname(photo_directory), "Annotated")

    # Get the list of photo files
    photo_files = [f for f in os.listdir(photo_directory) if f.endswith(".JPG") and "annotate" not in f.lower()]

    # If debug mode is enabled
    if debug:
        photo_files = [f for f in os.listdir(photo_directory) if f.endswith("S-DD-260_BOX(04) 9.00-12.00.JPG")]

    # Process each photo file in the directory
    for file_index, photo_file in enumerate(photo_files, start=1):
        match = re.search(r"(\d+\.\d+)-(\d+\.\d+)", photo_file)
        if match:
            start = float(match.group(1))
            end = float(match.group(2))

            # Predict core regions in the photo
            photo_path = os.path.join(photo_directory, photo_file)
            results = model.predict(source=photo_path, save=True, imgsz=1024)

            # Extract and sort original bounding boxes by vertical position
            original_boxes = sorted([
                {
                    'x_min': float(box[0]),
                    'y_min': float(box[1]),
                    'x_max': float(box[2]),
                    'y_max': float(box[3]),
                    'width': float(box[2]) - float(box[0]),
                    'height': float(box[3]) - float(box[1])
                } for box in results[0].boxes.xyxy
            ], key=lambda b: b['y_min'])
            # Calculate rock type breakdown for the interval
            original_boxes = filter_bounding_boxes(original_boxes)
            photo_interval = []
            total_length = end - start
            prct = 0
            for interval in intervals:
                if interval['start'] < end and interval['end'] > start:
                    overlap_start = max(interval['start'], start)
                    overlap_end = min(interval['end'], end)
                    overlap_length = overlap_end - overlap_start
                    if overlap_length > 0:
                        normalized_length = (overlap_length / total_length) * 100
                        photo_interval.append({
                            'start': overlap_start,
                            'end': overlap_end,
                            'rock': interval['rock'],
                            'width': float(overlap_end) - float(overlap_start),
                            'percentage': round(normalized_length, 2)
                        })
                        prct = prct + round(normalized_length, 2)
                        # print(f"Piece percentage {round(normalized_length, 2)}%, {overlap_length}")
            # print(f"Total percentage {prct}")

            # Assign bounding boxes to photo intervals based on percentage
            remaining_boxes = original_boxes[:]
            total_boxes_width = sum(box['width'] for box in original_boxes)
            # print(f'>> Total width: {total_boxes_width}')

            annotated_boxes = []
            for segment in photo_interval:
                required_width = segment['percentage'] / 100 * total_boxes_width
                # print(f'>>>>>>')
                # print(f" {segment['percentage']} Req width {required_width}")
                segment_boxes = []
                while required_width > 0 and remaining_boxes:
                    # print(f'-------------')
                    # print(f'Required width: {required_width}')
                    box = remaining_boxes.pop(0)
                    # print(f'Box popped with width: {box["width"]}')
                    if box['width'] >= required_width:
                        new_box = {
                            'x_min': box['x_min'],
                            'y_min': box['y_min'],
                            'x_max': box['x_min'] + required_width,
                            'y_max': box['y_max'],
                            'rock': segment['rock'],
                            'start_depth': segment['start'],
                            'end_depth': segment['end']
                        }
                        segment_boxes.append(new_box)
                        box['x_min'] += required_width
                        box['width'] -= required_width
                        if box['width'] > 0:
                            # print(f'Return remaining box to stack {box["width"]}')
                            remaining_boxes.insert(0, box)
                        required_width = 0
                    else:
                        # print(f'Not enough width, occupy all box width')
                        segment_boxes.append({
                            **box,
                            'rock': segment['rock'],
                            'start_depth': segment['start'],
                            'end_depth': segment['end']
                        })
                        required_width -= box['width']
                        # print(f'Width to be covered {required_width}')


                annotated_boxes.extend(segment_boxes)

            # Draw boxes on the image
            draw_boxes_on_image(photo_path, original_boxes, annotated_boxes, output_directory, file_index)

# Функция для сохранения classes.txt в указанные директории
def save_classes_txt(classes_array, output_dirs):
    classes_txt_content = "\n".join(classes_array)  # Формируем содержимое файла

    for directory in output_dirs:
        os.makedirs(directory, exist_ok=True)  # Создаем папку, если её нет
        file_path = os.path.join(directory, "classes.txt")

        with open(file_path, "w") as f:
            f.write(classes_txt_content)  # Записываем данные

        print(f"classes.txt сохранен в {file_path}")


def filter_bounding_boxes(boxes):
    """
    Фильтрует bounding boxes, оставляя только те, у которых высота отличается
    от максимальной не более чем на 20%.

    Parameters:
        boxes (list of dict): Список bounding boxes с координатами.

    Returns:
        list of dict: Отфильтрованный список bounding boxes.
    """
    if not boxes:
        return []

    # Определяем максимальную высоту
    max_height = max(box['height'] for box in boxes)

    # Фильтруем bounding boxes
    filtered_boxes = [box for box in boxes if abs(box['height'] - max_height) / max_height <= 0.3]

    return filtered_boxes

# Example usage
if __name__ == "__main__":
    base_directory = "../ProcessorData/drill_core_samples/"  # Replace with your base directory
    skip_rows = 6  # Number of rows to skip for headers
    debug_mode = False  # Set to True to enable debug mode

    all_intervals = process_all_logs(base_directory, skip_rows)
    rock_classes_array = list(filter(lambda x: re.match(r'^[A-Za-z]+$', x), rock_classes_set))
    save_classes_txt(rock_classes_array, ["../ore_markup/labels/train", "../ore_markup/labels/val"])

    # Example of processing photos for a specific drill hole
    for file_path, intervals in all_intervals.items():
        if not debug_mode or  ("S-DD-260" in file_path):
            print(f"Processing photos for drill hole from {file_path}:")
            process_photos_for_drill_hole(file_path, intervals, debug=debug_mode)
