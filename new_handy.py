import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import pyautogui
import time
import os
model = load_model("/Users/dhirenthakur/Documents/GESTURE_LANDMARKS/gesture_model.h5") 
class_names = ['label_down', 'label_max', 'label_min', 'label_none', 'label_up']


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)


last_action_time = time.time()
action_cooldown = 2  

def perform_action(predicted_label):
    global last_action_time
    current_time = time.time()
    if current_time - last_action_time < action_cooldown:
        return  

    if predicted_label == 'label_max':
        pyautogui.hotkey('ctrl', 'command', 'f')
    elif predicted_label == 'label_min':
        pyautogui.hotkey('command', 'm')             
    elif predicted_label == 'label_up':
        pyautogui.scroll(250)
    elif predicted_label == 'label_down':
        pyautogui.scroll(-250)
    
    last_action_time = current_time

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            features = []
            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])
            input_data = np.array([features])
    else:
        input_data = np.zeros((1, 63))  

    predictions = model.predict(input_data)[0]
    max_idx = np.argmax(predictions)
    predicted_label = class_names[max_idx]
    confidence = predictions[max_idx]

  
    cv2.putText(frame, f"Prediction: {predicted_label} ({confidence*100:.2f}%)", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    
    if confidence >= 0.9 and predicted_label != 'label_none':
        perform_action(predicted_label)

    
    y_offset = 70
    for i, prob in enumerate(predictions):
        text = f"{class_names[i]}: {prob*100:.2f}%"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 25

    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
