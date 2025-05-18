# Handwriting Detection System ✍️🔍

A deep learning-based handwriting recognition system that detects handwritten characters and digits using a MobileNetV2 model. Includes a custom GUI for interactive predictions and visualizations.

---

## 🚀 Features

- 📷 Upload handwritten character/digit images.
- 🧠 Classifies into 62 classes: A–Z, a–z, 0–9.
- 📊 Displays step-by-step image processing:
  - Original input
  - Resized version
  - Grayscale version
  - Final processed version
- 🖼️ Shows the predicted label with confidence.
- 🔎 Confusion matrix visualization and sample predictions.
- 🛠️ Built with TensorFlow and a custom Python GUI (no Streamlit).

---

## 🧠 Model

- **Architecture**: MobileNetV2 (ImageNet weights)
- **Training**: Fine-tuned with custom handwritten dataset
- **Input Shape**: 224×224 RGB images
- **Output**: 62-class softmax classifier

---

## 📁 Folder Structure

Handwriting/
├── data/ # Temporary files or exports
├── dataset/ # Handwriting images (split into train, val, test)
├── models/ # Saved .keras models
├── results/ # Output graphs, confusion matrix, logs
├── scripts/ # Training & evaluation scripts
│ ├── train_mobilenetv2.py
│ ├── evaluate_model.py
│ ├── compare_models.py
│ └── gui_app.py
├── README.md # Project info
└── requirements.txt # Python dependencies

---

## 🖥️ GUI Preview

- Upload an image from the interface.
- View intermediate processing steps (resize, grayscale, etc.).
- See predicted class and sample predictions.
- Visualize model performance metrics like confusion matrix.

---

## ✅ How to Run

1. Install dependencies
pip install -r requirements.txt

2. Train the model (optional)
python scripts/train_mobilenetv2.py

3. Evaluate model on test set
python scripts/evaluate_model.py

4. Launch the GUI
python scripts/gui_app.py

📊 Dataset
Custom labeled dataset of handwritten characters and digits

Split into training, validation, and testing folders

Folder names:

Lowercase: a, b, ..., z

Uppercase: Upper_A, Upper_B, ..., Upper_Z

Digits: 0, 1, ..., 9

📎 Requirements
Python 3.8+

TensorFlow 2.x

NumPy, Matplotlib, scikit-learn

PIL, Tkinter (for GUI)

📬 License
MIT License

✨ Credits
Developed by AryanTK for personal/commercial handwriting recognition projects.

---

