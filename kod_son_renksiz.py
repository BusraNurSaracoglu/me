import cv2
import numpy as np

frameWidth = 640
frameHeight = 480
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
    
   
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
    
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