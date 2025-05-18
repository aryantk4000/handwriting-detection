import os
import json

print("Script is running...")

# Point to the folder where the actual character class folders are
dataset_path = 'D:/Handwriting/dataset/train'

# Get list of class folders
class_names = sorted(os.listdir(dataset_path))

# Save class names to JSON
with open('D:/Handwriting/models/classes.json', 'w') as f:
    json.dump(class_names, f, indent=2)

print(f"Saved {len(class_names)} classes to classes.json")
