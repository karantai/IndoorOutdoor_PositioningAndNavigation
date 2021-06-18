 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 14:44:00 2020

@author: johny
"""

import numpy as np
import cv2
from pose_estimation import exteriorOrient_points
import pandas as pd


def coords(event, x, y, flags, param):
    global k
    global l

    if event == cv2.EVENT_LBUTTONDOWN:
        k += 1
        print(x,y)
        if k <= 4:
            imgp4[k] = (x, y)
        else:    
            l += 1
            imgp3[l] = (x, y)


def triangulate(imgp4, imgp3, projL = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/left_camera_info/projection_matrixL.npy'), \
                projR = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/right_camera_info/projection_matrixR.npy')):
    test_points = cv2.triangulatePoints(projL, projR, imgp4, imgp3)
    test_points = test_points[:3, :] / test_points[3, :]
    test_points = np.around(test_points, 3)
    return test_points






def diffs(test_coordinates, Coords):
    
    cols = ['dx', 'dy', 'dz']
    test_list = []
    rms = []

    inputt = input('Give the points you hit: ').strip().split(',')
    for i, inp in enumerate(inputt):
        if int(inp) in Coords[0]:
            test_list.append(Coords.iloc[int(inp) - 1, 1:] - \
                             test_coordinates[:, i])
        rms.append(np.sqrt(np.mean(test_list[i]**2)))
                
    
    result = pd.DataFrame(test_list, index = inputt)
    result.columns = cols
    result['RMSE (m)'] = rms
    print(np.round(result, 3))
    return (result)
    
    result = pd.DataFrame(test_list, index = inputt)
    result.columns = cols
    print(np.round(result, 3))
    return (result)


imgp3 = {}
imgp4 = {}
k = 0
l = 0
    
mtxR = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/right_camera_info/right_camera.npy')
distR = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/right_camera_info/right_distortion.npy')

mtxL = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/left_camera_info/left_camera.npy')
distL = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/left_camera_info/left_distortion.npy')



video_left = cv2.VideoCapture('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/left_cutt_video.mp4')
video_right = cv2.VideoCapture('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/right_cutt_video.mp4')

Coords = pd.read_csv('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/coords.txt', sep=",", header=None)
   

while(1):
    

    try:
        
        ret4, frame4 = (lambda video_left : video_left.read())(video_left)
        ret3, frame3 = (lambda video_right : video_right.read())(video_right)
    
        if ret4 == False or ret3 == False:
            raise ValueError('Frame has not been read properly.')
            
    except(IndexError):
        print('something went wrong')
        continue

    
    frames = [cv2.undistort(np.float64(frame4), mtxL, distL),\
                cv2.undistort(np.float64(frame3), mtxR, distR)]
    
       
    # frames = [frame4, frame3]
    
    
   
    for i, frame in enumerate(frames):        
        cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
        cv2.imshow( 'Camera', frame / 255.0)
        cv2.setMouseCallback('Camera', coords)
         
        
        while True:

            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyAllWindows()
                break   
    break
    
imgp4 = exteriorOrient_points(imgp4)
imgp3 = exteriorOrient_points(imgp3)




test_coordinates = triangulate(imgp4, imgp3)
results = diffs(test_coordinates, Coords)


np.save("test_points.npy", test_coordinates)