import os
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

#load model
model = model_from_json(open("fer.json", "r").read())
#load weights
model.load_weights('fer.h5')

face_haar_cascade = cv2.CascadeClassifier('venv/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')

cap=cv2.VideoCapture(0)

while True:
    ret,test_img=cap.read()
    if not ret:
        continue
    gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x,y,w,h) in faces_detected:

        roi_gray=gray_img[y:y+w,x:x+h]
        roi_gray=cv2.resize(roi_gray,(48,48))

        img_pixels = image.img_to_array(roi_gray)

        img_pixels = np.expand_dims(img_pixels, axis = 0)
        # img_pixels = np.expand_dims(img_pixels, -1)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        #find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        if predicted_emotion == 'angry':
            color =  np.asarray((191,0,191))
        elif predicted_emotion == 'disgust':
            color =  np.asarray((188,0,103))
        elif predicted_emotion == 'fear':
            color =  np.asarray((153,153,153))
        elif predicted_emotion == 'happy':
            color = np.asarray((86,255,255))
        elif predicted_emotion == 'sad':
            color = np.asarray((25,25,25))
        elif predicted_emotion == 'surprise':
            color = np.asarray((255,86,255))
        else:
            color = np.asarray((255,127,0))

        color = color.astype(int)
        color = color.tolist()

        cv2.rectangle(test_img, (x, y), (x + w, y + h), color, thickness=2)
        cv2.putText(test_img, predicted_emotion, (int(x), int(y -5)), cv2.FONT_HERSHEY_SIMPLEX, .9, color, 1)

    # resized_img = cv2.resize(test_img, (1000, 1000))
    cv2.namedWindow("Facial emotion analysis ", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Facial emotion analysis ", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Facial emotion analysis ',test_img)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break;

cap.release()
cv2.destroyAllWindows()