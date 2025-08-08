# collect_data.py
# This collects the data from your webcam and saves in a csv file

import cv2
import mediapipe as mp
import pandas as pd

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# letters
labels = list("ABC")  # can add more

data = []
current_label = 'A'

cap = cv2.VideoCapture(0)

print("Press key corresponding to hand sign (A–C). Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            landmarks.append(current_label)
            data.append(landmarks)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show current label
    cv2.putText(frame, f'Label: {current_label}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow("Hand Sign Data Collection", frame)

    key = cv2.waitKey(1)
    if key != -1:
        try:
            key_char = chr(key).upper()
            if key_char == 'Q':
                break
            elif key_char in labels:
                current_label = key_char
        except:
            pass

cap.release()
cv2.destroyAllWindows()

# Save data
df = pd.DataFrame(data)
df.to_csv('asl_hand_sign_data.csv', index=False)
print("Data saved to 'asl_hand_sign_data.csv'")
