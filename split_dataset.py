import os
import shutil
import random

# Paths
dataset_path = "dataset"              # original dataset folder
output_path = "processed_dataset"     # new dataset folder


splits = {
    "train": 80,
    "val": 40,
    "test": 10
}


# Create folders
def create_folders(base_path):
    for split in ["train", "val", "test"]:
        for label in ["Positives", "Negatives"]:
            path = os.path.join(base_path, split, label)
            os.makedirs(path, exist_ok=True)

# Load images
pos_images = os.listdir(os.path.join(dataset_path, "Positives"))
neg_images = os.listdir(os.path.join(dataset_path, "Negatives"))

# Shuffle images
random.shuffle(pos_images)
random.shuffle(neg_images)

# Split and copy function
def split_and_copy(images, label):
    start = 0
    for split, count in splits.items():
        selected = images[start:start+count]
        for img in selected:
            src = os.path.join(dataset_path, label, img)
            dst = os.path.join(output_path, split, label, img)
            shutil.copy(src, dst)
        start += count

# Run everything
create_folders(output_path)

split_and_copy(pos_images, "Positives")
split_and_copy(neg_images, "Negatives")

print("Dataset successfully split!")