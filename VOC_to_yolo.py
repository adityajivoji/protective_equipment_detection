
import os
import argparse
import xml.etree.ElementTree as ET

def load_class_map(classes_file):
    class_map = {}
    with open(classes_file, 'r') as f:
        for idx, class_name in enumerate(f.readlines()):
            class_map[class_name.strip()] = idx
    return class_map

def convert_voc_to_yolo(input_dir, output_dir, classes_file):
    class_map = load_class_map(classes_file)
    os.makedirs(output_dir, exist_ok=True)
    for xml_file in os.listdir(input_dir):
        if xml_file.endswith('.xml'):
            tree = ET.parse(os.path.join(input_dir, xml_file))
            root = tree.getroot()
            image_name = root.find('filename').text
            img_width = int(root.find('size/width').text)
            img_height = int(root.find('size/height').text)
            yolo_annotations = []
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name not in class_map or class_name != "person":
                    continue  # Skip classes not in our defined classes list
                class_id = class_map[class_name]
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                x_center = (xmin + xmax) / 2 / img_width
                y_center = (ymin + ymax) / 2 / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height
                yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")

            with open(os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.txt"), 'w') as f:
                f.write('\n'.join(yolo_annotations))
import os
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert PascalVOC to YOLO format.')
    parser.add_argument('input_dir',type=str, help='Input directory path containing PascalVOC annotations.')
    parser.add_argument('output_dir', type=str, help='Output directory path to save YOLO annotations.')
    args = parser.parse_args()
    
    convert_voc_to_yolo(os.path.join(args.input_dir, 'labels'), args.output_dir, os.path.join(args.input_dir, 'classes.txt'))