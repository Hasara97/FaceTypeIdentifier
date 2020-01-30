import cv2
import dlib

camera=cv2.VideoCapture(0)
face_detector=dlib.get_frontal_face_detector()
#loading the pre-trained algorithm for face detection available in the dilib

while(True):

    ret,img=camera.read()

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    rects=face_detector(gray)
    #same like - algorithm.predict(test_data)

    for rect in rects:

        x1=rect.left()
        y1=rect.top()
        x2=rect.right()
        y2=rect.bottom()

        cv2.rectangle(gray,(x1,y1),(x2,y2),(0,255,0),2)

        
    
    cv2.imshow('LIVE',img)
    cv2.imshow('GRAY',gray)
    
    cv2.waitKey(1)
