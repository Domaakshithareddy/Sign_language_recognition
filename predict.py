import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

IMG_SIZE = (64, 64)
MODEL_PATH = "ASL_model.h5"
TRAIN_DIR = "Dataset/asl_alphabet_train/asl_alphabet_train"

model = load_model(MODEL_PATH)
class_names = sorted(os.listdir(TRAIN_DIR))

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access webcam.")
    exit()

FONT = cv2.FONT_HERSHEY_SIMPLEX

BOX_SIZE = 500 
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
    pred_label = class_names[pred_idx]
    confidence = pred[0][pred_idx]

    display_text = f"{pred_label} ({confidence*100:.1f}%)"
    cv2.putText(frame, display_text, (10, 40), FONT, 1, (255, 255, 255), 2)

    cv2.imshow("ASL Real-Time Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()