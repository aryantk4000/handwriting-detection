import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Paths
test_dir = '../dataset/train'
model_save_path = '../models/handwritten_mobilenetv2.keras'

# Parameters
img_size = (224, 224)
batch_size = 32

# Load test data
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Load best saved model
best_model = tf.keras.models.load_model(model_save_path)

# Evaluate on test data
test_loss, test_acc = best_model.evaluate(test_generator)
print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

# Confusion matrix
y_true = test_generator.classes
y_pred_probs = best_model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

class_names = list(test_generator.class_indices.keys())

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 12))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.title("Confusion Matrix on Test Data")
plt.show()

# Show sample predictions
def show_sample_predictions(model, generator, class_names, num_samples=10):
    generator.reset()
    x_test, y_test = next(generator)
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    plt.figure(figsize=(15, num_samples * 2))
    for i in range(num_samples):
        plt.subplot(num_samples // 5 + 1, 5, i + 1)
        plt.imshow(x_test[i])
        plt.axis('off')
        true_label = class_names[y_true[i]]
        pred_label = class_names[y_pred[i]]
        color = 'green' if true_label == pred_label else 'red'
        plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)
    plt.tight_layout()
    plt.show()

show_sample_predictions(best_model, test_generator, class_names)
