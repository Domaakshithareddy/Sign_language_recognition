# ASL Sign Language Recognition with Real-Time Prediction and Voice Output

This project enables real-time recognition of American Sign Language (ASL) hand gestures using a webcam. It supports live prediction, string accumulation, and speech synthesis.

The system is trained using a custom dataset captured through webcam, and uses a deep learning model (MobileNetV2) for robust gesture classification.

---

## Project Structure

### `capture_img.py`

This script allows users to build their own dataset of ASL gestures using their webcam:

* A green box is displayed on the webcam feed.
* Press `s` to save the current image inside the green box.
* Press letter keys (`A`, `B`, ..., `Z`, etc.) to switch the label/class.
* Images are saved inside a structured folder format like `my_webcam_data/A/`, `B/`, etc.

This script is used to gather custom training data in a consistent format.

---

### `train.py`

This script trains a convolutional neural network (CNN) using MobileNetV2 as the backbone on the webcam-collected dataset:

* Uses Keras with MobileNetV2 pretrained on ImageNet.
* Applies data augmentation (rotation, zoom, brightness, etc.).
* Uses `ImageDataGenerator` to load the images from folders.
* Trains for multiple epochs, saving the best model using validation accuracy.
* Outputs a `.keras` model file suitable for real-time prediction.

This script must be run after dataset collection using `capture_img.py`.

---

### `predict.py`

This script uses OpenCV for real-time gesture recognition using the trained model:

* Displays a webcam window with a green ROI box.
* Predicts the current hand sign shown in the box.
* Stabilizes predictions using a buffer to avoid flickering.
* Displays the current detected letter below the box.
* Accumulates letters into a string at the top of the screen.
* Special keys:

  * Press `a` to add the current letter to the string.
  * Press `s` to read the full string aloud using text-to-speech.
  * Press `c` to clear the string.
  * Press `q` to quit the program.

Supports gesture-based prediction of `space` and `del`:

* If the model predicts `space`, it appends a space character when `a` is pressed.
* If the model predicts `del`, it appends the word "del" when `a` is pressed.

---

## Features

* Custom webcam dataset support
* Transfer learning with MobileNetV2
* Real-time prediction and display with OpenCV
* Stabilized predictions using a buffer
* Interactive control: build string, speak it, clear it
* Text-to-speech output using `pyttsx3`

---

## Usage Flow

1. Run `capture_img.py` to collect dataset.
2. Run `train.py` to train the model on captured images.
3. Run `predict.py` for real-time recognition and interaction.

---

## Requirements

* Python 3.8 or later
* TensorFlow
* OpenCV
* pyttsx3 (for speech synthesis)
