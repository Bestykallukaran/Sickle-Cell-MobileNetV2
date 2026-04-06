import os
import shutil
import random

# Original dataset path
source_dir = "dataset"

classes = ["Positives", "Negatives"]

# Target folders
for split in ["train", "val", "test"]:
    for cls in classes:
        os.makedirs(f"dataset/{split}/{cls}", exist_ok=True)

# Split ratio
train_ratio = 0.7
val_ratio = 0.2

for cls in classes:
    images = os.listdir(f"{source_dir}/{cls}")
    random.shuffle(images)

    total = len(images)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    for img in train_imgs:
        shutil.copy(f"{source_dir}/{cls}/{img}", f"dataset/train/{cls}/{img}")

    for img in val_imgs:
        shutil.copy(f"{source_dir}/{cls}/{img}", f"dataset/val/{cls}/{img}")

    for img in test_imgs:
        shutil.copy(f"{source_dir}/{cls}/{img}", f"dataset/test/{cls}/{img}")

print("✅ Dataset split completed!")