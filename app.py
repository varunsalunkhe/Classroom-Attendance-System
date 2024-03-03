import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import csv

path = 'training_images'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        #for encodding the face
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


new = []
def markAttendance(name):
    print(name)
    attendances = []
    
    if name not in new:
        now = datetime.now()
        dtString = now.strftime('%H:%M:%S')
        date = now.strftime("%d-%m-%y")
        attendances.append([name ,date, dtString])
        new.append(name)
        fields = ["Names", "Time"]
        # print(attendances)

        with open('Attendance.csv', 'a') as csvfile:
            csvwriter = csv.writer(csvfile)

            # writing the data rows
            csvwriter.writerows(attendances)

encodeListKnown = findEncodings(images)
print('Encoding Complete')


def main():
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        # img = captureScreen()

        #resizing the image
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        #Returns an array of bounding boxes of human faces in a image
        facesCurFrame = face_recognition.face_locations(imgS)

        #Given an image, return the 128-dimension face encoding for each face in the image.
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
     
        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            #Compare a list of face encodings against a candidate encoding to see if they match.
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)

            #Given a list of face encodings, compare them to a known face encoding and get a 
            #euclidean distance for each comparison face. The distance tells you how similar the faces are.
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()

                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                markAttendance(name)

        cv2.imshow('Webcam', img)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__": 
    main() 