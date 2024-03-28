import cv2
import numpy as np
import sqlite3
import os #truy cap duong dan

face_cascade=cv2.CascadeClassifier("E:/PycharmProjects/OpenCV/Resources/haarcascade_frontalface_default.xml")

cap=cv2.VideoCapture(0)

sampleNum=0

while(True):
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)

    for(x,y,w,h) in faces:
        # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        faces = frame[y:y + h+15, x:x + w+15]
        if not os.path.exists('dataset'):
            os.makedirs('dataset')
        output=cv2.resize(faces,(160,160))
        sampleNum+=1
        cv2.imwrite('dataset/nhat'   + str(sampleNum) + '.jpg',output)
        # cv2.imwrite('dataset/User.'+str(2)+'.'+ str(sampleNum) +'.jpg',gray[y:y+h,x:x+w])
        print(sampleNum)
    cv2.imshow('frame',frame)
    cv2.waitKey(1)

    if sampleNum>90:
        break

cap.release() #giphong
cv2.destroyAllWindows()