import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

train_images=np.load('encoded/train_img.npy')
train_labels=np.load('encoded/train_labels.npy')
test_images=np.load('encoded/test_img.npy')
test_labels=np.load('encoded/test_labels.npy')
label_classes=np.load('encoded/label_encoded_classes.npy')

train_images=train_images/255.0
test_images=test_images/255.0

model=Sequential([
    Conv2D(16,(3,3),activation='relu',input_shape=(224,224,3)),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Conv2D(32,(3,3),activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Conv2D(64,(3,3),activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64,activation='relu'),
    Dense(32,activation='relu'),
    Dense(18,activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=10,  # Stop if validation accuracy doesn't improve for 10 epochs
    restore_best_weights=True
)

history=model.fit(
    train_images,train_labels,
    epochs=50,
    validation_data=(test_images,test_labels),
    batch_size=32,
    callbacks=[early_stopping]
)

model.save('sign_language_model.h5')
print("Model saved as 'sign_language_model.h5'")

final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
print(f"Final training accuracy: {final_train_acc:.4f}")
print(f"Final validation accuracy: {final_val_acc:.4f}")