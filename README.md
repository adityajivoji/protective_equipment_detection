# protective_equipment_detection

## Standard way to use the repository

* Extract the dataset file
* Run the augmentation file if you wish to augment minority classes
* Run the VOC_to_yolo file
* Train the person detection model by running the train_person_detection.py file
* Run the crop_images_update_annotations.py file to create new annotations and images for the single person images
* Run train_ppe_detection.py file on this new dataset
* finally run the inference.py file for evaluating the models
