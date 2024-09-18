import os
import cv2
import numpy as np
from ultralytics import YOLO
import argparse
def inference(input_dir, output_dir, person_model_path, ppe_model_path):
    person_model = YOLO(person_model_path)
    ppe_model = YOLO(ppe_model_path)

    os.makedirs(output_dir, exist_ok=True)



    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        results = person_model(img)
        if len(results) == 0 or results[0].boxes.xyxy.shape[0] == 0:
            print(f"No persons detected in {img_name}")
            continue


        person_boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ppe_boxes = []
        ppe_labels = []
        for (x1, y1, x2, y2) in person_boxes:
            cropped_img = img[y1:y2, x1:x2]

            ppe_results = ppe_model(cropped_img)

            if len(ppe_results) == 0 or ppe_results[0].boxes.xyxy.shape[0] == 0:
                print(f"No PPE detected in cropped image of {img_name}")
                continue
            for ppe_result in ppe_results:
                for box in ppe_result.boxes:
                    # Convert tensor index to integer
                    class_idx = int(box.cls.item())
                    ppe_x1, ppe_y1, ppe_x2, ppe_y2 = map(int, box.xyxy[0].cpu().numpy())
                    ppe_boxes.append([ppe_x1 + x1, ppe_y1 + y1, ppe_x2 + x1, ppe_y2 + y1])
                    ppe_labels.append(ppe_result.names[class_idx-1])


        draw_boxes(img, ppe_boxes, ppe_labels)

        output_img_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_img_path, img)

def draw_boxes(img, boxes, labels):
    for (box, label) in zip(boxes, labels):
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform inference using YOLO models.')
    parser.add_argument('input_dir', type=str, help='Directory of input images.')
    parser.add_argument('output_dir', type=str, help='Directory to save annotated images.')
    parser.add_argument('person_det_model', type=str, help='Path to person detection model weights.')
    parser.add_argument('ppe_detection_model', type=str, help='Path to PPE detection model weights.')
    args = parser.parse_args()
    
    inference(os.path.join(args.input_dir, 'images'), args.output_dir, args.person_det_model, args.ppe_detection_model)
