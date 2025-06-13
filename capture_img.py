import cv2
import os

IMG_SIZE = (64, 64)
SAVE_DIR = "my_webcam_data" 
BOX_SIZE = 500

FONT = cv2.FONT_HERSHEY_SIMPLEX
current_class = 'A'
counter = 0

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

ensure_dir(os.path.join(SAVE_DIR, current_class))

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Webcam not found.")
    exit()

print("Press letter key to change class (e.g., A-Z)")
print("Press 's' to save image, 'q' to quit")

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

    display_text = f"Class: {current_class} | Saved: {counter}"
    cv2.putText(frame, display_text, (10, 30), FONT, 1, (255, 255, 255), 2)

    cv2.imshow("ASL Data Collector", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        roi = frame[y1:y2, x1:x2]
        roi_resized = cv2.resize(roi, IMG_SIZE)
        save_path = os.path.join(SAVE_DIR, current_class)
        ensure_dir(save_path)
        cv2.imwrite(f"{save_path}/{current_class}_{counter}.jpg", roi_resized)
        counter += 1
        print(f"[+] Saved {current_class}_{counter}.jpg")

    elif 65 <= key <= 90: 
        current_class = chr(key)
        counter = len(os.listdir(os.path.join(SAVE_DIR, current_class))) if os.path.exists(os.path.join(SAVE_DIR, current_class)) else 0
        ensure_dir(os.path.join(SAVE_DIR, current_class))
        print(f"[>] Switched to class '{current_class}'")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
