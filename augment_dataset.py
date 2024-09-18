import os
import cv2
import xml.etree.ElementTree as ET
import albumentations as A
from albumentations.augmentations import transforms
import numpy as np
from tqdm import tqdm

# Augment dataset
def load_class_map(classes_file):
    class_map = {}
    with open(classes_file, 'r') as f:
        for idx, class_name in enumerate(f.readlines()):
            class_map[class_name.strip()] = idx
    return class_map

def get_minority_class_images(input_dir, class_map, minority_classes):
    """
    Identify images containing minority classes based on annotations.
    """
    minority_images = []
    annotations_dir = os.path.join(input_dir, 'xmls')
    for xml_file in os.listdir(annotations_dir):
        xml_path = os.path.join(annotations_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name in minority_classes:
                image_name = os.path.splitext(xml_file)[0] + '.jpg'
                minority_images.append(image_name)
                break

    return minority_images

def augment_image(image, bboxes, class_labels, transforms):
    """
    Apply augmentation to the image and adjust bounding box coordinates accordingly.
    """
    if not bboxes:
        return image, bboxes

    augmented = transforms(image=image, bboxes=bboxes, class_labels=class_labels)

    # Ensure bboxes is not empty
    if len(augmented['bboxes']) == 0:
        return image, bboxes

    return augmented['image'], augmented['bboxes']

def save_augmented_image_and_annotations(aug_image, aug_bboxes, output_image_path, output_annotation_path, class_labels, class_map):
    """
    Save the augmented image and annotations in XML format.
    """
    # Save augmented image
    cv2.imwrite(output_image_path, aug_image)

    # Update the XML file
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "images"
    ET.SubElement(root, "filename").text = os.path.basename(output_image_path)
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(aug_image.shape[1])
    ET.SubElement(size, "height").text = str(aug_image.shape[0])
    ET.SubElement(size, "depth").text = "3"

    for bbox, class_label in zip(aug_bboxes, class_labels):
        x_min, y_min, x_max, y_max = bbox
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = [key for key in class_map.keys() if class_map[key] == class_label][0]
        bbox_xml = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox_xml, "xmin").text = str(int(x_min))
        ET.SubElement(bbox_xml, "ymin").text = str(int(y_min))
        ET.SubElement(bbox_xml, "xmax").text = str(int(x_max))
        ET.SubElement(bbox_xml, "ymax").text = str(int(y_max))

    tree = ET.ElementTree(root)
    tree.write(output_annotation_path)

def draw_boxes_on_image(image, bboxes, class_labels, class_map):
    """
    Draw bounding boxes on an image.
    """
    for bbox, class_label in zip(bboxes, class_labels):
        x_min, y_min, x_max, y_max = map(int, bbox)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        label = [key for key in class_map.keys() if class_map[key] == class_label][0]
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    return image

def augment_minority_classes(input_dir, output_dir, classes_file, minority_classes, n_augmentations=5):
    """
    Perform data augmentation on images containing minority classes.
    """
    class_map = load_class_map(classes_file)
    minority_images = get_minority_class_images(input_dir, class_map, minority_classes)

    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'xmls'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'verification'), exist_ok=True)

    # Define augmentations
    transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Blur(p=0.1),
        A.Rotate(limit=30, p=0.5),
        A.ColorJitter(p=0.2)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    for img_name in tqdm(minority_images):
        img_path = os.path.join(input_dir, 'images', img_name)
        xml_path = os.path.join(input_dir, 'xmls', f"{os.path.splitext(img_name)[0]}.xml")
        img = cv2.imread(img_path)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Gather all bounding boxes and their class labels
        bboxes = []
        class_labels = []

        for obj in root.findall('object'):
            class_name = obj.find('name').text
            bbox = obj.find('bndbox')
            x_min = int(bbox.find('xmin').text)
            y_min = int(bbox.find('ymin').text)
            x_max = int(bbox.find('xmax').text)
            y_max = int(bbox.find('ymax').text)

            bboxes.append([x_min, y_min, x_max, y_max])
            class_labels.append(class_map[class_name])

        for j in range(n_augmentations):
            aug_img, aug_bboxes = augment_image(img, bboxes, class_labels, transforms)

            if not aug_bboxes:
                continue

            aug_img_name = f"{os.path.splitext(img_name)[0]}_aug_{j}.jpg"
            aug_label_name = f"{os.path.splitext(img_name)[0]}_aug_{j}.xml"
            aug_img_path = os.path.join(output_dir, 'images', aug_img_name)
            aug_label_path = os.path.join(output_dir, 'xmls', aug_label_name)

            save_augmented_image_and_annotations(aug_img, aug_bboxes, aug_img_path, aug_label_path, class_labels, class_map)

            # Draw boxes and save to verification folder
            verification_img = draw_boxes_on_image(aug_img.copy(), aug_bboxes, class_labels, class_map)
            verification_img_path = os.path.join(output_dir, 'verification', aug_img_name)
            cv2.imwrite(verification_img_path, verification_img)

if __name__ == "__main__":
    input_dir = 'datasets/'
    output_dir = 'datasets/'
    classes_file = "datasets/classes.txt"
    minority_classes = ["mask", "vest", "glasses", "ear-protector"]
    augment_minority_classes(input_dir, output_dir, classes_file, minority_classes)
