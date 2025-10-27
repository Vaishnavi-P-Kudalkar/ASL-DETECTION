import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import mediapipe as mp

# Constants
IMG_SIZE = 128
MODEL_PATH = "asl_mobilenetv2_generator (1).h5"
LABELS_PATH = "D:/asl_detector/asl_alphabet/asl_alphabet_train/asl_alphabet_train"

# Load model and labels
model = load_model(MODEL_PATH)
labels = sorted([
    folder for folder in os.listdir(LABELS_PATH)
    if os.path.isdir(os.path.join(LABELS_PATH, folder))
])
print("Loaded labels:", labels)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

def preprocess_hand_region(frame, bbox):
    x_min, y_min, x_max, y_max = bbox
    hand_img = frame[y_min:y_max, x_min:x_max]

    if hand_img.size == 0:
        return None

    hand_img = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
    hand_img = hand_img.astype('float32') / 255.0
    hand_img = np.expand_dims(hand_img, axis=0)
    return hand_img

def get_hand_bbox(hand_landmarks, image_shape):
    img_h, img_w = image_shape
    x_coords = [landmark.x * img_w for landmark in hand_landmarks.landmark]
    y_coords = [landmark.y * img_h for landmark in hand_landmarks.landmark]
    x_min, x_max = int(max(min(x_coords) - 20, 0)), int(min(max(x_coords) + 20, img_w))
    y_min, y_max = int(max(min(y_coords) - 20, 0)), int(min(max(y_coords) + 20, img_h))
    return x_min, y_min, x_max, y_max

# Start webcam
cap = cv2.VideoCapture(0)
print("Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            bbox = get_hand_bbox(hand_landmarks, frame.shape[:2])
            processed = preprocess_hand_region(frame, bbox)
            if processed is not None:
                preds = model.predict(processed)
                idx = np.argmax(preds)
                letter = labels[idx]
                confidence = preds[0][idx]
                display_text = f"{letter} ({confidence:.2f})"

                # Draw landmarks and prediction
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.putText(frame, display_text, (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
    else:
        cv2.putText(frame, "No hand detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("ASL Detection with MediaPipe", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# so this is media pipe+ mobilenet 
#mobilemnetmodel(1) works well 