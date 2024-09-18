from ultralytics import YOLO

# Load a YOLOv8 model (YOLOv8s in this case)
model = YOLO('yolov8s.pt')

# Train the model
model.train(
    data='person_detection.yaml',  # path to the dataset config file
    epochs=200,                    # total number of epochs you want to train
    imgsz=640,                     # image size for training
    batch=16,                      # batch size
    name='yolov8_person_detection',# name of the training session

)

