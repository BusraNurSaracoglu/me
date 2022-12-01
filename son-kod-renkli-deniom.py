from collections import deque
import numpy as np
import argparse
import imutils
import cv2
import urllib 

frameWidth = 640
frameHeight = 480



lower = {'red':(166, 84, 80), 'green':(66, 122, 129), 'blue':(97, 100, 117), 'yellow':(23, 59, 119)} 
upper = {'red':(186,255,255), 'green':(86,255,255), 'blue':(117,255,255), 'yellow':(54,255,255)}
 

colors = {'red':(0,0,255), 'green':(0,255,0), 'blue':(255,0,0), 'yellow':(0, 255, 217)}
 


cap = cv2.VideoCapture("python/test1.mp4")
cap.set(3, frameWidth)
cap.set(4, frameHeight)

def empty(a):
    pass

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters",640,240)
cv2.createTrackbar("Threshold1","Parameters",23,255,empty)
cv2.createTrackbar("Threshold2","Parameters",20,255,empty)
cv2.createTrackbar("Area","Parameters",5000,30000,empty)

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def getCon(image,imgContour):
    cons, hrhy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in cons:
        area = cv2.contourArea(contour)
        if area>7000:
            cv2.drawContours(imgContour, contour, -1, (0, 0, 255), 3)
            perimeter = cv2.arcLength(contour,True)
            corners = cv2.approxPolyDP(contour, 0.01*perimeter, True)
            x, y, width, height = cv2.boundingRect(corners)
            shape = len(corners)
            if shape == 3: type = "triangle"
            elif shape == 4:
                ratio = width/float(height)
                if ratio >=0.9 and ratio <= 1.04:
                    type = 'square'
                else:   type = "rectangle"
            else: type = "circle"
            cv2.rectangle(imgContour, (x-20,y-20), (x+width+15, y+height+15), (255,0,0), 3)
            cv2.putText(imgContour, type,
                        (x+int(width/2)-15, y+int(height/2)), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,0,0),1)


while True:


    success, img = cap.read()
    imgContour = img.copy()
    img = imutils.resize(img, width=900)
   
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (11, 11),0)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    for key, value in upper.items():
    
        kernel1 = np.ones((9,9),np.uint8)
        mask = cv2.inRange(hsv, lower[key], upper[key])
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel1)
               
    
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None

        if len(cnts) > 0:
            
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
       

            if radius > 0.5:
            
                cv2.circle(img, (int(x), int(y)), int(radius), colors[key], 5)
                cv2.putText(img,key + "color", (int(x-radius),int(y-radius)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,colors[key],2)
 
     
    
    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    
    imgCanny = cv2.Canny(imgGray,threshold1,threshold2)

    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
    getCon(imgDil,imgContour)
    imgStack = stackImages(0.6,([img,imgCanny],
                                [imgDil,imgContour]))

    cv2.imshow("Result", imgStack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break