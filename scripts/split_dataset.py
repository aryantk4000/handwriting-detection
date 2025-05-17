import os
import shutil
import random

# Change this to your current dataset folder path:
dataset_dir = 'D:/Handwriting/dataset'  

train_dir = os.path.join(dataset_dir, 'train')
val_dir = os.path.join(dataset_dir, 'val')
test_dir = os.path.join(dataset_dir, 'test')

# Create directories if they don't exist
for folder in [train_dir, val_dir, test_dir]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# For each class folder (like 'a', 'b', 'Upper_A', etc)
for class_name in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    # Skip train/val/test if they exist in main dataset folder
    if class_name in ['train', 'val', 'test']:
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    n_total = len(images)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    # Create class subfolders in train, val, test
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    # Copy files to train
    for img in images[:n_train]:
        src = os.path.join(class_path, img)
        dst = os.path.join(train_dir, class_name, img)
        shutil.copy2(src, dst)

    # Copy files to val
    for img in images[n_train:n_train + n_val]:
        src = os.path.join(class_path, img)
        dst = os.path.join(val_dir, class_name, img)
        shutil.copy2(src, dst)

    # Copy files to test
    for img in images[n_train + n_val:]:
        src = os.path.join(class_path, img)
        dst = os.path.join(test_dir, class_name, img)
        shutil.copy2(src, dst)

print("Dataset split done!")
