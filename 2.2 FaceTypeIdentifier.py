import cv2
import dlib
from imutils import face_utils
import os

camera=cv2.VideoCapture(0)

     

face_detector=dlib.get_frontal_face_detector()
#loading the pre-trained algorithm for face detection available in the dilib

landmark_detector=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
#loading the pre-trained algorithm for 68 landmark detecting externally
#os.mkdir('MyFolder')

datapath='H://Machine Learning & Image Processing Course - AIE/Week 04/FaceTypeIdentifier/Codes/Face Shapes'

labels=os.listdir(datapath)

label_dict={}

for i in range(len(labels)):

    label_dict[i]=labels[i]
    
print(label_dict)

import joblib

algorithm=joblib.load('KNN_model.sav')


def predict_face_type(img):

    

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    rects=face_detector(img)
    #same like - algorithm.predict(test_data)

    for rect in rects:

        x1=rect.left()
        y1=rect.top()
        x2=rect.right()
        y2=rect.bottom()

        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

        points=landmark_detector(img,rect)
        #passing the gray image and ROI (Dtected face rectangle) to the landmark_detector
        #return is 68 points (x,y) (Note this is not a numpy array)

        points=face_utils.shape_to_np(points)
        #converting the points in to numpy array using imutils module

        for point in points:

            xp=point[0]
            yp=point[1]

            cv2.circle(img,(xp,yp),2,(0,0,255),-1)

        myPoints=points[2:9,0]
        #getting the x cordinates of print 2-8

        D1=myPoints[6] - myPoints[0]
        D2=myPoints[6] - myPoints[1]
        D3=myPoints[6] - myPoints[2]
        D4=myPoints[6] - myPoints[3]
        D5=myPoints[6] - myPoints[4]
        D6=myPoints[6] - myPoints[5]

        d1=D2/D1
        d2=D3/D1
        d3=D4/D1
        d4=D5/D1
        d5=D6/D1

        result=algorithm.predict([[d1,d2,d3,d4,d5]])

        img[0:50,:] = [0,255,0]
        face_type=label_dict[result[0]]

        cv2.putText(img,face_type,(10,40),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),2)

        print(result) 
      
        

while(True):

    ret,img=camera.read()
    predict_face_type(img)
    cv2.imshow('LIVE',img)

    cv2.waitKey(1)
    
