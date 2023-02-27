import cv2
import numpy as np
import face_recognition

imgBean = face_recognition.load_image_file('data/Ned Stark.JPG')
imgBean = cv2.cvtColor(imgBean, cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('data/Boromir.JPG')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgBean)[0]
encodeBean = face_recognition.face_encodings(imgBean)[0]
cv2.rectangle(imgBean, (faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]), (255,0,255), 2)

# you can change test image file path to test other images
faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]), (255,0,255), 2)

results =  face_recognition.compare_faces([encodeBean], encodeTest)
faceDis = face_recognition.face_distance([encodeBean], encodeTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Sean Bean as Ned Stark', imgBean)
cv2.imshow('Test Image', imgTest)
cv2.waitKey(0)



