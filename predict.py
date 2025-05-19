import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load the trained TensorFlow model
model = load_model(r'C:\Users\honey\OneDrive\Desktop\execute\sign_language_model.h5')  # Replace with your path

# Labels for sign language detection
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'blank', 'speak']

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Function to preprocess the frame for prediction (resize and normalize)
def preprocess_frame(frame):
    # Resize the frame to match the input shape of the model
    frame_resized = cv2.resize(frame, (300, 300))  # Adjust according to model input size
    frame_normalized = frame_resized.astype('float32') / 255.0  # Normalize pixel values
    frame_reshaped = np.reshape(frame_normalized, (1, 300, 300, 3))  # Add batch dimension
    return frame_reshaped

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Flip the frame horizontally (mirror effect)
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB for MediaPipe (OpenCV uses BGR by default)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(frame_rgb)  # Process the frame to detect hands

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Debug: Draw bounding box around the hand
            h, w, c = frame.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

            # Increase the bounding box size by adding a constant factor (e.g., 20% increase)
            padding = 40  # Adjust padding size to control the bounding box expansion
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            # Draw bounding box (for debugging)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Crop the region of interest (ROI) for the hand
            hand_roi = frame[y_min:y_max, x_min:x_max]

            # Check if the ROI is too small or empty
            if hand_roi.size == 0:
                continue

            # Preprocess the hand ROI for prediction
            processed_hand_roi = preprocess_frame(hand_roi)

            # Predict the sign language letter
            prediction = model.predict(processed_hand_roi)
            predicted_class = np.argmax(prediction, axis=1)
            predicted_label = labels[predicted_class[0]]

            # Display the predicted label on the frame
            cv2.putText(frame, f"Prediction: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frame with landmarks and predictions
    cv2.imshow("Sign Language Detection", frame)

    # Exit loop on pressing 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the windows
hands = mp_hands.Hands(min_detection_confidence=0.3, min_tracking_confidence=0.3)

cap.release()
cv2.destroyAllWindows()
