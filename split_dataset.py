import os
import shutil
import random
# partition
def split_dataset(data_dir, output_dir, split_ratio=0.8):
    # Paths for images, labels, and xml files
    images_dir = os.path.join(data_dir, 'images')
    labels_dir = os.path.join(data_dir, 'labels')
    xmls_dir = os.path.join(data_dir, 'xmls')

    # Paths for output directories
    train_images_dir = os.path.join(output_dir, 'train', 'images')
    val_images_dir = os.path.join(output_dir, 'val', 'images')
    train_labels_dir = os.path.join(output_dir, 'train', 'labels')
    val_labels_dir = os.path.join(output_dir, 'val', 'labels')
    train_xmls_dir = os.path.join(output_dir, 'train', 'xmls')
    val_xmls_dir = os.path.join(output_dir, 'val', 'xmls')

    # Create output directories
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    os.makedirs(train_xmls_dir, exist_ok=True)
    os.makedirs(val_xmls_dir, exist_ok=True)

    # List of all images
    image_files = os.listdir(images_dir)
    # image_files = [f for f in image_files if f.endswith('.jpg')]  # Filter only jpg files
    image_files.sort()

    # Shuffle images for randomness
    random.shuffle(image_files)

    # Split the dataset
    split_index = int(len(image_files) * split_ratio)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    # Helper function to move files
    def move_files(files, src_dir, dst_dir, ext):
        for file_name in files:
            if ext != ".jpg":
                file_path = f"{os.path.splitext(file_name)[0]}{ext}" # file_name.replace('.jpg', ext)
            else:
                file_path = file_name
            src_file_path = os.path.join(src_dir, file_path)
            dst_file_path = os.path.join(dst_dir, file_path)
            if os.path.exists(src_file_path):
                shutil.copy2(src_file_path, dst_file_path)
            else:
                print(f"Warning: {src_file_path} does not exist and will be skipped.")

    # Move images and corresponding labels and xml files
    for train_file in train_files:
        move_files([train_file], images_dir, train_images_dir, '.jpg')
        move_files([train_file], labels_dir, train_labels_dir, '.txt')
        move_files([train_file], xmls_dir, train_xmls_dir, '.xml')

    for val_file in val_files:
        move_files([val_file], images_dir, val_images_dir, '.jpg')
        move_files([val_file], labels_dir, val_labels_dir, '.txt')
        move_files([val_file], xmls_dir, val_xmls_dir, '.xml')

    print(f"Dataset split completed. {len(train_files)} files for training and {len(val_files)} files for validation.")

    # Verification step
    verify_dataset(train_images_dir, train_labels_dir, train_xmls_dir)
    verify_dataset(val_images_dir, val_labels_dir, val_xmls_dir)

def verify_dataset(images_dir, labels_dir, xmls_dir):
    image_files = os.listdir(images_dir)
    missing_files = False

    for image_file in image_files:
        label_file = f"{os.path.splitext(image_file)[0]}.txt"
        xml_file = f"{os.path.splitext(image_file)[0]}.xml"

        if not os.path.exists(os.path.join(labels_dir, label_file)):
            print(f"Missing label file for image: {image_file}")
            missing_files = True
        if not os.path.exists(os.path.join(xmls_dir, xml_file)):
            print(f"Missing XML file for image: {image_file}")
            missing_files = True

    if not missing_files:
        print(f"All files verified for directory {images_dir}.")
    else:
        print(f"Some files are missing in the dataset for directory {images_dir}.")

# Example usage
data_dir = 'datasets/'
output_dir = 'datasets/partitioned'
split_dataset(data_dir, output_dir)
