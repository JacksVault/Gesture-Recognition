import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the directory where your gesture images are stored
dataset_dir = "dataset"

# Load the image data and labels
data = []
labels = []

# Parameters for data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Iterate through subdirectories (each subdirectory represents a gesture)
for gesture_dir in os.listdir(dataset_dir):
    gesture_path = os.path.join(dataset_dir, gesture_dir)
    if os.path.isdir(gesture_path):
        for image_file in os.listdir(gesture_path):
            if image_file.endswith(".jpg"):
                image = cv2.imread(os.path.join(gesture_path, image_file))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                image = cv2.resize(image, (128, 128))  # Resize to a consistent size
                data.append(image)
                labels.append(gesture_dir)

# Convert data and labels to NumPy arrays
data = np.array(data, dtype="float32") / 255.0  # Normalize pixel values
labels = np.array(labels)

# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# Define and compile the neural network model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),  # Dropout for regularization
    keras.layers.Dense(len(set(labels)), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Data augmentation during training
datagen.fit(train_data)

# Train the model
history = model.fit(datagen.flow(train_data, train_labels, batch_size=32),
                    steps_per_epoch=len(train_data) / 32, epochs=20, validation_data=(test_data, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_data, test_labels)
print(f"Test accuracy: {test_acc}")

# Save the trained model
model.save("models/gesture_recognition_model.h5")
