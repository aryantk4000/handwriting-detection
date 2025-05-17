import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
dataset_dir = '../dataset'
test_dir = '../dataset/test'
model_path = '../models/handwritten_mobilenetv2_fixed.keras'

# Parameters
img_size = (224, 224)
batch_size = 32

# Data generators
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=False
)

val_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Load model
model = tf.keras.models.load_model(model_path)

# Evaluate
train_loss, train_acc = model.evaluate(train_generator)
val_loss, val_acc = model.evaluate(val_generator)
test_loss, test_acc = model.evaluate(test_generator)

print("\nModel Comparison (Accuracy):")
print(f"{'Model':<15} {'Train Acc':<10} {'Val Acc':<10} {'Test Acc':<10}")
print(f"{'MobileNetV2':<15} {train_acc:.4f}     {val_acc:.4f}     {test_acc:.4f}")
