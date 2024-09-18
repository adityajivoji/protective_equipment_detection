import os
import cv2
import xml.etree.ElementTree as ET
from ultralytics import YOLO

def load_class_map(classes_file):
    class_map = {}
    with open(classes_file, 'r') as f:
        for idx, class_name in enumerate(f.readlines()):
            class_map[class_name.strip()] = idx
    return class_map

from ultralytics import YOLO

def crop_images_and_update_annotations(input_dir, output_dir, person_model_path):
    # Load the person detection model
    person_model = YOLO(person_model_path)

    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

    for img_name in os.listdir(os.path.join(input_dir, 'images')):
        img_path = os.path.join(input_dir, 'images', img_name)
        img = cv2.imread(img_path)

        # Detect persons in the image
        results = person_model(img)
        # print(type(person_model(img)))
        # print("gaseuidfgiuasedgfhuisagedfuigh",results[0].boxes.xyxy.shape, img_name)

        if len(results) == 0 or results[0].boxes.xyxy.shape[0] == 0:  # Check if results are empty or no bounding boxes
            print(f"No detections found in image {img_name}")
            continue
        for i, result in enumerate(results[0].boxes.xyxy):

            x1, y1, x2, y2 = map(int, result)

            # Crop the image
            cropped_img = img[y1:y2, x1:x2]

            # Save the cropped image
            cropped_img_name = f"{os.path.splitext(img_name)[0]}_person_{i}.jpg"
            cv2.imwrite(os.path.join(output_dir, 'images', cropped_img_name), cropped_img)

            # Update annotations for the cropped image
            print(cropped_img_name)
        # import sys
        # sys.exit()
            update_annotations(input_dir, output_dir, img_name, cropped_img_name, x1, y1, x2, y2)

def update_annotations(input_dir, output_dir, original_img_name, cropped_img_name, x1, y1, x2, y2):
    # Load the original annotation file
    xml_file = os.path.join(input_dir, 'xmls',  f"{os.path.splitext(original_img_name)[0]}.xml")
    tree = ET.parse(xml_file)
    root = tree.getroot()

    img_width = int(root.find('size/width').text)
    img_height = int(root.find('size/height').text)
    cropped_width = x2 - x1
    cropped_height = y2 - y1
    class_map = load_class_map("datasets/classes.txt")
    yolo_annotations = []

    for obj in root.findall('object'):
        class_name = obj.find('name').text

        # Only include PPE classes in the new annotations
        if class_name in ["hard-hat", "gloves", "mask", "glasses", "boots", "vest", "ppe-suit", "ear-protector", "safety-harness"]:
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            # Check if the PPE is within the person bounding box
            if xmin >= x1 and xmax <= x2 and ymin >= y1 and ymax <= y2:
                # Adjust coordinates relative to the cropped image
                new_xmin = xmin - x1
                new_ymin = ymin - y1
                new_xmax = xmax - x1
                new_ymax = ymax - y1

                # Convert to YOLO format
                x_center = (new_xmin + new_xmax) / 2 / cropped_width
                y_center = (new_ymin + new_ymax) / 2 / cropped_height
                width = (new_xmax - new_xmin) / cropped_width
                height = (new_ymax - new_ymin) / cropped_height

                class_id = class_map[class_name]
                yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")

    # Save the new annotations
    annotation_file_path = os.path.join(output_dir, 'new', f"{os.path.splitext(cropped_img_name)[0]}.txt")
    with open(annotation_file_path, 'w') as f:
        f.write('\n'.join(yolo_annotations))



# class_map = load_class_map("datasets/classes.txt")
if __name__ == "__main__":
    input_dir = 'datasets/partitioned/train' # images ka path
    output_dir = 'datasets/partitioned/train/ppe' # ppe wala path
    person_model_path = 'runs/detect/yolov8_person_detection/weights/best.pt' # weights
    crop_images_and_update_annotations(input_dir, output_dir, person_model_path)
