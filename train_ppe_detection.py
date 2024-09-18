from ultralytics import YOLO

# Load a YOLOv8 model (YOLOv8s in this case)
model = YOLO('yolov8s.pt')

# Train the model
model.train(
    data='/content/drive/MyDrive/Syook/ppe_detedtion.yaml',  # path to the dataset config file
    epochs=100,                    # number of training epochs
    imgsz=640,                     # image size for training
    batch=16,                      # batch size
    name='yolov8_ppe_detection' # name of the training session
)
