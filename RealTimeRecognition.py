import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os
import pyttsx3
#engine = pyttsx3.init()


# load saved model from PC
model = tf.keras.models.load_model(r'\ResNet50_ASL.h5')
model.summary()


#initiating the video source, 0 for internal camera
cap = cv2.VideoCapture(1)
while(True):
    
    _ , frame = cap.read()
    frame = cv2.flip(frame,1)
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 0, 255), 5) 
    #region of intrest
    roi = frame[100:300, 100:300]
    img = cv2.resize(roi, (96, 96))
    cv2.imshow('roi', roi)
    

    img = img

    prediction = model.predict(img.reshape(1, 96, 96, 3))
    char_index = np.argmax(prediction)
    # print(char_index)

    confidence = round(prediction[0, char_index] * 100, 1)
    predicted_char = labels[char_index]
    # print(predicted_char)
    # engine = pyttsx3.init()
    # engine.say(predicted_char)
    # engine.runAndWait()

    font = cv2.FONT_HERSHEY_TRIPLEX
    fontScale = 1
    color = (0,255,255)
    thickness = 2

    #writing the predicted char and its confidence percentage to the frame
    msg = predicted_char +', Conf: ' +str(confidence)+' %'
    cv2.putText(frame, msg, (80, 80), font, fontScale, color, thickness)
    
    cv2.imshow('frame',frame)

    #close the camera when press 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
        
#release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
