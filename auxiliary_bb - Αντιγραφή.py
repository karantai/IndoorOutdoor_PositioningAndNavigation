# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 14:43:52 2020

@author: Admin
"""

import cv2
import numpy as np
from change_detection_boxes import Detects

def change_det(substruction, frame, num):

    
        
    bb = []
    contours_list = []
    

        
    # diff = cv2.subtract(left_back, frame)  
    # gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    mask = substruction.apply(frame)

    blur = cv2.GaussianBlur(mask, (5, 5), 0)


    
    _, threshold = cv2.threshold(blur, 130, 255, cv2.THRESH_BINARY)

    
    
    # applying erosion - dilation to eliminate noise
    
    kernel_dil = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(threshold, kernel_dil, iterations = 1)   
    contours, hier = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return([], [])

    
    
    cntrs = sorted(contours, key = cv2.contourArea, reverse = True)[:2]
    
    cv2.namedWindow(f'Camera{num}', cv2.WINDOW_NORMAL)
    
    for i in cntrs:
        
            if cv2.contourArea(i) > 5000 :
                # (x,y),(MA,ma), angle = cv2.fitEllipse(i)
                # print(angle)
            
                # making contours convex
                hull = cv2.convexHull(i, clockwise = True, returnPoints = True)
                
                # contour approximation
                epsilon = 0.0001 * cv2.arcLength(hull,  False)
                approx = cv2.approxPolyDP(hull, epsilon, True)
                
                
                x, y, w, h = cv2.boundingRect(i)
                bb.append([x, y, w, h])
                contours_list.append(i)
                
                # drawing and showing convexed aproximated contours
                img2 = cv2.drawContours(frame , [i], -1, (255,0,0), 3)

                # drawing bounding boxes through contours
                cv2.rectangle(img2, (x, y), (x + w, y + h), (0,255,0),2)
            
                
                cv2.imshow(f'Camera{num}', img2)
                key = cv2.waitKey(20)
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    break

            else:
                continue
    return(bb, contours_list)

    



                
    

    

if __name__ == '__main__':
    
    
    substructionL = cv2.createBackgroundSubtractorMOG2()
    substructionR = cv2.createBackgroundSubtractorMOG2()
    
    video_right = cv2.VideoCapture('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/right_cutt_video.mp4')
    video_left = cv2.VideoCapture('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/left_cutt_video.mp4')
    
    while True:
        
        _, frameL = video_left.read()
        _, frameR = video_right.read()
        
        
        bbL, contoursL = change_det(substructionL, frameL, num = 4)
        bbR, contoursR = change_det(substructionR, frameR, num = 3)
        
        if not bbL or not bbR :
            continue

        detectionsL = [Detects(box, contourL) for (box, contourL) in zip(bbL, contoursL)]
        detectionsR = [Detects(box, contourR) for (box, contourR) in zip(bbR, contoursR)]
    

                
                
        x=[Detects.matching(i, j, frameL, frameR, numL, numR) for numL, i in enumerate(detectionsL) for numR, j in enumerate(detectionsR)]
        

    
    
    
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    