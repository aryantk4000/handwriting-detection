import os
import json

TRAIN_DIR = 'train'  # Change if needed
OUTPUT_JSON = 'models/classes.json'

folders = sorted(os.listdir(TRAIN_DIR))
class_names = []

for folder in folders:
    path = os.path.join(TRAIN_DIR, folder)
    if os.path.isdir(path):
        class_names.append(folder)

with open(OUTPUT_JSON, 'w') as f:
    json.dump(class_names, f, indent=2)

print(f"Generated classes.json with {len(class_names)} classes at: {OUTPUT_JSON}")
