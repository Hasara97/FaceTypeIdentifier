import cv2
import dlib
from imutils import face_utils 

camera=cv2.VideoCapture(0)
face_detector=dlib.get_frontal_face_detector()
#loading the pre-trained algorithm for face detection available in the dilib

landmark_detector=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
#loading the pre-trained algorithm for 68 landmark detecting externally

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

        points=landmark_detector(gray,rect)
        #passing the gray image and ROI (Dtected face rectangle) to the landmark_detector
        #return is 68 points (x,y) (Note this is not a numpy array)

        points=face_utils.shape_to_np(points)
        #converting the points in to numpy array using imutils module
        print(points)
        for point in points:

            xp=point[0]
            yp=point[1]

            cv2.circle(img,(xp,yp),2,(0,0,255),-1)

        
    
    cv2.imshow('LIVE',img)
    cv2.imshow('GRAY',gray)
    
    cv2.waitKey(1)
