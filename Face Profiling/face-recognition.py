import cv2
import numpy as np
import  pickle

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {}
with open("labels.pickle" , "rb") as f:
    originalLabels = pickle.load(f)
    labels = {v:k for k,v in originalLabels.items()}

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray , scaleFactor=1.5 , minNeighbors=5)

    for(x,y,w,h) in faces:
        # print(x,y,w,h)
        roi_gray = gray[y:y+h , x:x+w]
        roi_color = frame[y:y+h , x:x+w]


        id , conf = recognizer.predict(roi_gray)
        if conf < 75 and conf > 45:
            print(id)
            print (labels[id])

            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id]
            color2 = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, ((int(x), int(y - 5))), font, 1, color2, stroke , cv2.LINE_AA)

        img_item = "me2.png"
        # cv2.imwrite(img_item ,roi_color)

        color = (255, 0 , 0)
        strokes  =2
        width = x+w
        height = y+h
        cv2.rectangle(frame , (x,y) , (width , height) , color , strokes)


    cv2.imshow('me' ,frame)


    # lower_red = np.array([100,150,50])
    # upper_red = np.array([180,255,180])
    #
    # mask = cv2.inRange(hsv , lower_red , upper_red)
    # res = cv2.bitwise_and(frame , frame , mask=mask)
    #
    # cv2.imshow('frame', frame)
    # cv2.imshow('mask', mask)
    # cv2.imshow('res', res)

    # edges = cv2.Canny(frame , 100,150)
    # cv2.imshow('Edges' , edges)

    k = cv2.waitKey(5) & 0xFF
    if k==27:
        break;

# img = cv2.imread('F:\\IIT Sixth Semester\\AI\\Faces\\group\\IMG-20181224-WA0025.jpg')
# # img = cv2.imread('F:\IIT Sixth Semester\AI\Faces\9th\hridi\\27973512_986796094829806_500801605147654488_n.jpg')
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# group = face_cascade.detectMultiScale(gray_img, scaleFactor=5, minNeighbors=5 )
#
# for(x,y,w,h) in group:
#     # print(x,y,w,h)
#     roi_gray = gray_img[y:y+h , x:x+w]
#     roi_color = img[y:y+h , x:x+w]
#
#     id, conf = recognizer.predict(roi_gray)
#     if conf > 45:
#         print(id)
#         print(labels[id])
#
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         name = labels[id]
#         color2 = (255, 255, 255)
#         stroke = 2
#         cv2.putText(img, name, ((int(x), int(y - 5))), font, 1, color2, stroke, cv2.LINE_AA)
#
#     # img_item = "me2.png"
#     # cv2.imwrite(img_item ,roi_color)
#
#     color = (255, 0, 0)
#     strokes = 2
#     width = x + w
#     height = y + h
#     cv2.rectangle(img, (x, y), (width, height), color, strokes)
# cv2.imshow("group" , img)

cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()