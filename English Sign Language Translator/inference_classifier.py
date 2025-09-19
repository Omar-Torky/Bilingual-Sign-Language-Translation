import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque, Counter

# Load model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Camera
cap = cv2.VideoCapture(0)

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Labels
labels_dict = {i: chr(65 + i) for i in range(26)}
labels_dict[26] = 'Space'
labels_dict[27] = 'Backspace'  

# Sentence tracking
sentence = ""
predictions_queue = deque(maxlen=20)
last_added_char = ""
last_time_added = time.time()
ADD_LETTER_DELAY = 3.0  # seconds between letters

# Scanning effect variables
scan_start_time = 0
scan_duration = 0.6  # seconds
scanning = False

# Draw bounding box
def draw_camera_box(img, x1, y1, x2, y2, color=(0, 0, 255), thickness=3, corner_len=150, full_box=True):
    if full_box:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    else:
        cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, thickness)
        cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, thickness)
        cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, thickness)
        cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, thickness)
        cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, thickness)
        cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, thickness)
        cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, thickness)

# Main loop
while True:
    data_aux = []
    x_, y_ = [], []
    ret, frame = cap.read()
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    current_time = time.time()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                x_.append(x)
                y_.append(y)

            # بعد ما نجيب min/max مرة واحدة
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))


            x1 = int(min(x_) * W) - 20
            y1 = int(min(y_) * H) - 20
            x2 = int(max(x_) * W) + 20
            y2 = int(max(y_) * H) + 20

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]
            predictions_queue.append(predicted_character)

            most_common_char, count = Counter(predictions_queue).most_common(1)[0]

            if count > 15 and (most_common_char != last_added_char or current_time - last_time_added > ADD_LETTER_DELAY):
                if most_common_char == 'Space':
                    sentence += ' '
                elif most_common_char == 'Backspace':
                    sentence = sentence[:-1]
                else:
                    sentence += most_common_char
                last_added_char = most_common_char
                last_time_added = current_time
                scan_start_time = current_time
                scanning = True

            # Draw box
            draw_camera_box(frame, x1, y1, x2, y2, color=(0, 0, 255))

            # Scanning effect (moving green line)
            if scanning and current_time - scan_start_time < scan_duration:
                progress = (current_time - scan_start_time) / scan_duration
                scan_y = int(y1 + progress * (y2 - y1))
                cv2.line(frame, (x1, scan_y), (x2, scan_y), (0, 255, 0), 2)
            else:
                scanning = False

            # Show predicted char
            cv2.putText(frame, most_common_char, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

    # Show sentence
    cv2.rectangle(frame, (20, 400), (620, 450), (255, 255, 255), -1)
    cv2.putText(frame, sentence, (30, 435), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3, cv2.LINE_AA)

    # Show frame
    cv2.imshow('frame', frame)

    # Keyboard input
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):
        sentence = ""
        last_added_char = ""
    elif key == 32:  # Space key
        sentence += ' '
        last_added_char = ""
        predictions_queue.clear() 
    elif key == ord('z'):  # Backspace
        if sentence:
            sentence = sentence[:-1]
            last_added_char = ""
            predictions_queue.clear()

cap.release()
cv2.destroyAllWindows()
