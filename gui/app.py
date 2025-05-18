import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Avoid Tkinter errors in Flask
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import base64
from io import BytesIO
import json

# Set up Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model and class names from a JSON file
model = load_model('D:/Handwriting/models/handwritten_mobilenetv2.keras')

# Load class names from JSON file
with open('D:/Handwriting/models/classes.json', 'r') as f:
    class_names = json.load(f)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # MobileNetV2 input size
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0,1]
    return np.expand_dims(img_array, axis=0)

def plot_to_base64():
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()
    return img_base64

def plot_prediction_bar(pred_probs, class_names, top_n=10):
    plt.figure(figsize=(8, 5))
    top_indices = np.argsort(pred_probs)[-top_n:][::-1]
    top_probs = pred_probs[top_indices]
    top_labels = [class_names[i] for i in top_indices]

    sns.barplot(x=top_probs, y=top_labels, palette='viridis')
    plt.xlabel('Probability')
    plt.title('Top Predictions')
    plt.tight_layout()

    return plot_to_base64()

def plot_confusion_like(pred_probs, class_names, top_n=5):
    plt.figure(figsize=(5, 4))
    top_indices = np.argsort(pred_probs)[-top_n:][::-1]
    labels = [class_names[i] for i in top_indices]
    matrix = np.zeros((top_n, top_n))
    np.fill_diagonal(matrix, pred_probs[top_indices])

    sns.heatmap(matrix, annot=True, xticklabels=labels, yticklabels=labels, cmap='Blues', fmt='.2f')
    plt.title('Confusion-like Matrix (Top Predictions)')
    plt.ylabel('True (simulated)')
    plt.xlabel('Predicted')
    plt.tight_layout()
    
    return plot_to_base64()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            filename = secure_filename(uploaded_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(filepath)

            img_array = preprocess_image(filepath)
            preds = model.predict(img_array)

            print("Prediction vector length:", len(preds[0]))
            pred_index = np.argmax(preds[0])
            print("Predicted index:", pred_index)
            print("class_names length:", len(class_names))

            if pred_index >= len(class_names):
                return f"Error: predicted index {pred_index} out of range for class_names length {len(class_names)}"

            pred_label = class_names[pred_index]
            pred_confidence = float(preds[0][pred_index]) * 100

            barplot_base64 = plot_prediction_bar(preds[0], class_names)
            conf_matrix_base64 = plot_confusion_like(preds[0], class_names)

            return render_template('index.html',
                                   filename=filename,
                                   pred_label=pred_label,
                                   pred_confidence=f"{pred_confidence:.2f}",
                                   barplot_img=barplot_base64,
                                   conf_matrix_img=conf_matrix_base64)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
