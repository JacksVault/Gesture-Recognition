import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras

# Define the directory where your gesture images are stored
dataset_dir = "dataset"
gesture_subdirs = ["raw_fist", "raw_left_hand", "raw_negative", "raw_open_hand", "raw_right_hand"]

# Load the image data and labels
data = []
labels = []

for gesture_dir in gesture_subdirs:
    gesture_path = os.path.join(dataset_dir, gesture_dir)
    for image_file in os.listdir(gesture_path):
        if image_file.endswith(".jpg"):
            image = cv2.imread(os.path.join(gesture_path, image_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            image = cv2.resize(image, (128, 128))  # Resize to a consistent size
            data.append(image)
            labels.append(gesture_dir)

# Encode the labels as numerical values
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Convert data and labels to NumPy arrays
data = np.array(data, dtype="float32") / 255.0  # Normalize pixel values

# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# Define and compile the neural network model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(len(set(labels)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, train_labels, epochs=10)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_data, test_labels)
print(f"Test accuracy: {test_acc}")

# Save the trained model
model.save("models/gesture_recognition_model.h5")
