import cv2
import numpy as np
import os
from collections import deque
from tensorflow.keras.models import load_model
import pyttsx3

IMG_SIZE = (128, 128)
MODEL_PATH = "mobilenetv2_finetuned.keras"
TRAIN_DIR = "my_webcam_data"
BOX_SIZE = 500
FONT = cv2.FONT_HERSHEY_SIMPLEX

model = load_model(MODEL_PATH)
class_names = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])

prediction_buffer = deque(maxlen=15)
collected_text = ""
spoken = False

engine = pyttsx3.init()
engine.setProperty('rate', 150)

cap = cv2.VideoCapture(0)
print("Press 'a' to add letter, 's' to speak, 'c' to clear, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    x1 = (w - BOX_SIZE) // 2
    y1 = (h - BOX_SIZE) // 2
    x2 = x1 + BOX_SIZE
    y2 = y1 + BOX_SIZE

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    roi = frame[y1:y2, x1:x2]
    if roi.shape[0] == 0 or roi.shape[1] == 0:
        continue

    roi_resized = cv2.resize(roi, IMG_SIZE)
    roi_normalized = roi_resized.astype("float32") / 255.0
    roi_input = np.expand_dims(roi_normalized, axis=0)

    pred = model.predict(roi_input, verbose=0)
    pred_idx = np.argmax(pred)
    confidence = pred[0][pred_idx]

    if confidence > 0.7:
        label = class_names[pred_idx]
    else:
        label = "_"

    if label == "space":
        display_label = "space"
    elif label == "del":
        display_label = "del"
    else:
        display_label = label

    prediction_buffer.append(display_label)
    stable_label = max(set(prediction_buffer), key=prediction_buffer.count)

    cv2.putText(frame, f"Letter: {stable_label}", (10, 95), FONT, 1.4, (255,0,0), 3)
    cv2.putText(frame, f"Text: {collected_text}", (10, 50), FONT, 1.4, (0,0,0), 3)
    cv2.putText(frame, f"Press 'a' to add letter, 's' to speak, 'c' to clear, 'q' to quit.", (w-990,h-15), FONT, 1, (0,0,0), 2)

    cv2.imshow("ASL to Text", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('a') and stable_label != "_":
        if stable_label == "space":
            collected_text += " "
        elif stable_label == "del":
            collected_text=collected_text[:-1]
        else:
            collected_text += stable_label
        spoken = False
    elif key == ord('s') and collected_text and not spoken:
        engine.say(collected_text)
        engine.runAndWait()
        spoken = True
    elif key == ord('c'):
        collected_text = ""
        spoken = False

cap.release()
cv2.destroyAllWindows()