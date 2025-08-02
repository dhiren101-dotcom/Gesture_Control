import cv2
import mediapipe as mp
import pandas as pd
import os
from datetime import datetime

# Initialize mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Output CSV file
csv_filename = "gesture_landmarks.csv"
if not os.path.exists(csv_filename):
    df = pd.DataFrame(columns=[f'{axis}{i}' for i in range(21) for axis in ['x', 'y', 'z']] + ['label'])
    df.to_csv(csv_filename, index=False)

# Define key-label mapping
key_label_map = {
    ord('u'): 'up',
    ord('d'): 'down',
    ord('m'): 'min',
    ord('x'): 'max',
    ord('n'): 'none'
}

cap = cv2.VideoCapture(0)

print("Press:")
print("  'u' for UP")
print("  'd' for DOWN")
print("  'm' for MIN")
print("  'x' for MAX")
print("  'n' for NONE")
print("Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame to avoid mirrored view
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Collect landmark data
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Check for key press to label
            key = cv2.waitKey(1) & 0xFF
            if key in key_label_map:
                label = key_label_map[key]
                row = landmarks + [label]
                df = pd.DataFrame([row])
                df.to_csv(csv_filename, mode='a', header=False, index=False)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved '{label}' gesture.")

    cv2.imshow("Collecting Hand Gestures", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
