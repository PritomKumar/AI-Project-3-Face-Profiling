import os
import cv2
import numpy as np
from PIL import Image
import  pickle

face_cascade = cv2.CascadeClassifier('venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

BASE_DIR  = os.path.dirname("F:\IIT Sixth Semester\AI\Faces\\")
image_dir = os.path.join(BASE_DIR , "9th")

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root,dirs , files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root,file)
            label = os.path.basename(root).replace(" ","").lower()
            # print(label,path)

            if not label in label_ids:
                label_ids[label] = current_id
                current_id +=1
            id =  label_ids[label]
            # print(label_ids)
            # y_labels.append(label)
            # x_train.append(path)

            pil_image = Image.open(path).convert("L")
            final_image = pil_image.resize((500, 500), Image.ANTIALIAS)
            image_array = np.array(pil_image)


            # print(image_array)
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi = image_array[y:y+h , x:x+h]
                x_train.append(roi)
                y_labels.append(id)


# print(y_labels)
# print(x_train)

with open("labels.pickle" , "wb") as f:
    pickle.dump(label_ids,f)

recognizer.train(x_train , np.array(y_labels))
recognizer.save("trainner.yml")

print ("Training data Complete")