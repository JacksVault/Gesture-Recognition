import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the trained gesture recognition model
model = keras.models.load_model("models/gesture_recognition_model.h5")

# Define labels for your gestures
labels = ["raw_fist", "raw_left_hand", "raw_negative", "raw_open_hand", "raw_right_hand"]

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 indicates the default camera

# Set an initial window size
cv2.namedWindow("Gesture Recognition", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture an image.")
        break

    # Preprocess the frame (resize, convert to grayscale, normalize)
    frame = cv2.resize(frame, (128, 128))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = frame.astype('float32') / 255.0

    # Make predictions with the model
    input_data = np.expand_dims(frame, axis=0)  # Add batch dimension
    predictions = model.predict(input_data)
    predicted_label = labels[np.argmax(predictions)]

    # Display the predicted label on the frame
    font_scale = min(frame.shape[0], frame.shape[1]) / 300.0
    thickness = int(min(frame.shape[0], frame.shape[1]) / 75.0)
    cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 255, 0), thickness)

    # Display the image in a resizable window
    cv2.imshow("Gesture Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
