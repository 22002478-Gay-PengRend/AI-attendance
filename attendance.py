import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'img_attendance'
images = []
classNames = []
myList = os.listdir(path)

print(myList)

for name in myList:
    currentImg = cv2.imread(f'{path}/{name}')
    images.append(currentImg)
    classNames.append(os.path.splitext(name)[0])

print(classNames)

def findEncodings(images):
    encodeList = []
    
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []

        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

encodeListFound = findEncodings(images)
print(f'Encoding completed with {len(encodeListFound)} name(s) found')

# webcam initialization
cap = cv2.VideoCapture(0)

while True: # while the webcam is running
    success, frame = cap.read()

    if not success:
        break
    else:
        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)
        # reduce the size of image due to real time capture for faster analysis
        frameSmall = cv2.resize(frame, (0,0), None, 0.25,0.25)
        # then convert to RGB as cv2 takes in BGR
        frameSmall = cv2.cvtColor(frameSmall, cv2.COLOR_BGR2RGB)
        # since we are getting multiple faces from the webcam image
        # find the location of the faces then send the locations to the encoding function
        facesCurFrame = face_recognition.face_locations(frameSmall)
        # now let's get the encoding
        encodeCurFrame = face_recognition.face_encodings(frameSmall, facesCurFrame)

        for encodings, faceLocs in zip(encodeCurFrame,facesCurFrame):
            faceMatch = face_recognition.compare_faces(encodeListFound, encodings)
            faceDis = face_recognition.face_distance(encodeListFound, encodings)
            #print(faceDis)
            # the lowest value is the best match, so now we find the best match from the index
            matchIndex = np.argmin(faceDis)
            print(faceDis[matchIndex])
            # With the correct index, we know which person we are referring to, and then display bounding box and write the name
            if faceMatch[matchIndex]:
                name = classNames[matchIndex].capitalize()
                #print(name)
                y1,x2,y2,x1 = faceLocs
                # because we scaled down previously, we need to rescale up before showing the bounding boxes
                y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                # draw a green rectangle around the recognized face
                cv2.rectangle(frame, (x1,y1),(x2,y2), (0,255,0), 2)
                # draw a filled green rectangle below the recognized face
                cv2.rectangle(frame, (x1,y2+35), (x2,y2), (0,255,0), cv2.FILLED)
                # write the name of the recognized person on the frame
                cv2.putText(frame, name, (x1+6,y2+25), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
                # Mark the attendance for the recognized person
                markAttendance(name.capitalize())

    cv2.imshow('Webcam',frame)
    #cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
cv2.destroyAllWindows()


# faceLoc = face_recognition.face_locations(imgBean)[0]
# encodeBean = face_recognition.face_encodings(imgBean)[0]
# cv2.rectangle(imgBean, (faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]), (255,0,255), 2)

# # you can change test image file path to test other images
# faceLocTest = face_recognition.face_locations(imgTest)[0]
# encodeTest = face_recognition.face_encodings(imgTest)[0]
# cv2.rectangle(imgTest, (faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]), (255,0,255), 2)

# results =  face_recognition.compare_faces([encodeBean], encodeTest)
# faceDis = face_recognition.face_distance([encodeBean], encodeTest)