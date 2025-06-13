from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

IMG_SIZE = (64, 64)
model = load_model("ASL_model.h5")
test_dir = "Dataset/asl_alphabet_test/asl_alphabet_test"
class_names = sorted(os.listdir("Dataset/asl_alphabet_train/asl_alphabet_train"))

correct = 0
total = 0

for filename in os.listdir(test_dir):
    if filename.lower().endswith(".jpg"):
        true_label = filename.split('_')[0] 

        img_path = os.path.join(test_dir, filename)
        img = image.load_img(img_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        pred = model.predict(img_array, verbose=0)
        predicted_label = class_names[np.argmax(pred)]

        print(f"{filename} ➤ Predicted: {predicted_label} | Actual: {true_label}")

        if predicted_label.lower() == true_label.lower():
            correct += 1
        total += 1

accuracy = correct / total if total > 0 else 0
print(f"\n Accuracy: {correct}/{total} = {accuracy * 100:.2f}%")
