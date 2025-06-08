import cv2 as cv
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pyttsx3
import time

model=tf.keras.models.load_model('sign_language_model.h5')
label_clases=np.load('encoded/label_encoded_classes.npy')

mp_hands=mp.solutions.hands
hands=mp_hands.Hands(static_image_mode=False,max_num_hands=2,min_detection_confidence=0.5)
mp_drawing=mp.solutions.drawing_utils

engine=pyttsx3.init()
engine.setProperty('rate',150)
last_spoken_word=None
last_speak_time=0
speech_cooldown=2

cap=cv.VideoCapture(0)
if not cap.isOpened():
    print('Error: Could not open cam')
    exit()
    
while cap.isOpened():
    ret,frame=cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    frame_rgb=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    frame_rgb.flags.writeable=False
    results=hands.process(frame_rgb)
    frame_rgb.flags.writeable=True
    frame=cv.cvtColor(frame_rgb,cv.COLOR_RGB2BGR)
    
    predicted_word='No hands detected'
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)
            
            h,w,_=frame.shape
            x_min=w
            y_min=h
            x_max=0
            y_max=0
            
            for lm in hand_landmarks.landmark:
                x,y=int(lm.x*w),int(lm.y*h)
                x_min=min(x,x_min)
                y_min=min(y,y_min)
                x_max=max(x,x_max)
                y_max=max(y,y_max)
            padding=20
            x_min=max(0,x_min-padding)
            y_min=min(0,y_min-padding)
            x_max=max(w,x_max+padding)
            y_max=max(h,y_max+padding)
            
            cv.rectangle(frame,(x_min,y_min),(x_max,y_max),(0,255,0),2)
            
            hand_region=frame[y_min:y_max,x_min:x_max]
            if hand_region.size==0:
                continue
            hand_region_resize=cv.resize(hand_region,(224,224))
            hand_region_normalized=hand_region_resize/255.0
            hand_region_input=np.expand_dims(hand_region_normalized,axis=0)
            
            predictions=model.predict(hand_region_input,verbose=0)
            predicted_label=np.argmax(predictions,axis=1)[0]
            predicted_word=label_clases[predicted_label]
            
            current_time=time.time()
            if (predicted_word!=last_spoken_word or (current_time-last_speak_time)>speech_cooldown):
                engine.say(predicted_word)
                engine.runAndWait()
                last_spoken_word=predicted_word
                last_speak_time=current_time
    
    cv.putText(frame,f'Prediction:{predicted_word}',(10,30),cv.FONT_HERSHEY_SCRIPT_SIMPLEX,1,(0,255,0),2)
    cv.imshow('Sign Language Detection', frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
hands.close()
engine.stop()
print("Real-time prediction stopped.")