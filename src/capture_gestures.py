import cv2
import os
import mediapipe as mp

# Define the directory to save captured images
gesture_name_face = "face_identifier"  # Replace with the name of the faces
gesture_name_finger = "raw_fist"  # Replace with the name of the finger/hand gesture
output_dir_face = f"dataset/{gesture_name_face}"
output_dir_finger = f"dataset/{gesture_name_finger}"

# Create the output directories if they don't exist
os.makedirs(output_dir_face, exist_ok=True)
os.makedirs(output_dir_finger, exist_ok=True)

# Initialize MediaPipe FaceMesh and Hands
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.2)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 indicates the default camera

# Counters for captured images
img_counter_face = 0
img_counter_finger = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture an image.")
        break

    # Convert the frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Use MediaPipe to detect face landmarks
    results_face = face_mesh.process(frame_rgb)

    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # Use MediaPipe to detect hands
    results_finger = hands.process(frame_rgb)

    if results_finger.multi_hand_landmarks:
        for hand_landmarks in results_finger.multi_hand_landmarks:
            for connection in mp_hands.HAND_CONNECTIONS:
                x0, y0 = int(hand_landmarks.landmark[connection[0]].x * frame.shape[1]), int(hand_landmarks.landmark[connection[0]].y * frame.shape[0])
                x1, y1 = int(hand_landmarks.landmark[connection[1]].x * frame.shape[1]), int(hand_landmarks.landmark[connection[1]].y * frame.shape[0])
                cv2.line(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)
                cv2.circle(frame, (x0, y0), 5, (255, 0, 0), -1)
                cv2.circle(frame, (x1, y1), 5, (255, 0, 0), -1)

    # Display the image in a window
    cv2.imshow("Capture Gesture", frame)

    # Press 'c' to capture an image for face gesture
    if cv2.waitKey(1) & 0xFF == ord('c'):
        img_name_face = f"{gesture_name_face}_image_{img_counter_face}.jpg"
        img_path_face = os.path.join(output_dir_face, img_name_face)
        cv2.imwrite(img_path_face, frame)
        print(f"Face Image {img_counter_face} captured as {img_name_face}")
        img_counter_face += 1

    # Press 'f' to capture an image for finger gesture
    if cv2.waitKey(1) & 0xFF == ord('f'):
        img_name_finger = f"{gesture_name_finger}_image_{img_counter_finger}.jpg"
        img_path_finger = os.path.join(output_dir_finger, img_name_finger)
        cv2.imwrite(img_path_finger, frame)
        print(f"Finger Image {img_counter_finger} captured as {img_name_finger}")
        img_counter_finger += 1

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
