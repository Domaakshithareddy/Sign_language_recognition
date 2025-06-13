import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from collections import deque

# === CONFIG ===
IMG_SIZE = (128, 128)  # must match training
MODEL_PATH = "mobilenetv2_finetuned.keras"
TRAIN_DIR = "my_webcam_data"  # where class folders are
BOX_SIZE = 300  # green box size

# === Load model and class names ===
model = load_model(MODEL_PATH)
class_names = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])

# === Prediction buffer for stability ===
buffer = deque(maxlen=15)

# === Webcam Setup ===
cap = cv2.VideoCapture(0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
print("Press 'q' to quit.")

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

    # Draw green box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Get ROI
    roi = frame[y1:y2, x1:x2]
    if roi.shape[0] == 0 or roi.shape[1] == 0:
        continue

    # Resize + normalize
    roi_resized = cv2.resize(roi, IMG_SIZE)
    roi_normalized = roi_resized.astype("float32") / 255.0
    roi_input = np.expand_dims(roi_normalized, axis=0)

    # Predict
    pred = model.predict(roi_input, verbose=0)
    pred_idx = np.argmax(pred)
    confidence = pred[0][pred_idx]

    # Apply threshold
    if confidence > 0.7:
        label = class_names[pred_idx]
    else:
        label = "Uncertain"

    buffer.append(label)

    # Get most frequent label
    stable_label = max(set(buffer), key=buffer.count)

    # Display
    text = f"{stable_label} ({confidence*100:.1f}%)" if stable_label != "Uncertain" else "Uncertain"
    cv2.putText(frame, text, (10, 40), FONT, 1, (255, 255, 255), 2)

    cv2.imshow("ASL Prediction", frame)

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()