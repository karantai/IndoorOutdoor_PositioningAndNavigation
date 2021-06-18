# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 14:43:52 2020

@author: Admin
"""

import cv2
import numpy as np
from change_detection_boxes import Detects
import generate_detections as gdet
from scipy import spatial
import create_3d_bb
import skgeom as sg
import create_3d_bb_contours
from bbox.metrics import jaccard_index_2d
from bbox import BBox2D, XYXY
import itertools


def change_det(substruction, frame, num):
 
        
    bb = []
    contours_list = []
    cont_count = 0

    mask = substruction.apply(frame, learningRate = -1)
    blur = cv2.GaussianBlur(mask, (5, 5), 0)
    _, threshold = cv2.threshold(blur, 130, 255, cv2.THRESH_BINARY)
    
    kernel_dil = np.ones((11, 11), np.uint8)
    dilation = cv2.dilate(threshold, kernel_dil, iterations = 1)   
    contours, hier = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return([], []) 
    cntrs = sorted(contours, key = cv2.contourArea, reverse = True)[:2]
    
    for i in cntrs:
        
            x, y, w, h = cv2.boundingRect(i)
            if cv2.contourArea(i) > 10_000:

                # making contours convex
                hull = cv2.convexHull(i, clockwise = True, returnPoints = True)
                
                # contour approximation
                epsilon = 0.0001 * cv2.arcLength(hull,  False)
                approx = cv2.approxPolyDP(hull, epsilon, True) 
                bb.append([x, y, w, h])
                contours_list.append(i)


            else:
                continue
    return(bb, contours_list)


def triangulate(imgp4, imgp3, projL, projR):
    test_points = cv2.triangulatePoints(projL, projR, np.float32(imgp4), np.float32(imgp3))

    test_points = test_points[:3, :] / test_points[3, :]
    test_points = np.around(test_points, 3).T
    projected = np.int32(cv2.projectPoints(test_points, cv2.Rodrigues(
        rot_mat3)[0], 
        tvecs3, 
        mtxR, np.zeros((1,5)))[0])
    return projected, test_points
                
    

    

if __name__ == '__main__':
    model_filename = 'model_data/mars-small128.pb'
    encoderR = gdet.create_box_encoder(model_filename, batch_size=1)
    encoderL = gdet.create_box_encoder(model_filename, batch_size=1)
    
    mtxL = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/left_camera_info/left_camera.npy')
    mtxR = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/right_camera_info/right_camera.npy')
        
    
    substructionL = cv2.createBackgroundSubtractorMOG2()
    substructionR = cv2.createBackgroundSubtractorMOG2()
    
    results = {}
    
    case4 = np.load('table_coords_left.npy')
    case3 = np.load('table_coords_right.npy')
    
    
    
    lines_table = []
    projL = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/left_camera_info/projection_matrixL.npy')
    projR = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/right_camera_info/projection_matrixR.npy')
    
    rot_mat4 = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/left_camera_info/rotation_matrixL.npy')
    rot_mat3 = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/right_camera_info/rotation_matrixR.npy')
 
    video_right = cv2.VideoCapture('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/right_cut_video.mp4')
    video_left = cv2.VideoCapture('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/left_cut_video.mp4')
    
    distL = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/left_camera_info/left_distortion.npy')
    distR = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/right_camera_info/right_distortion.npy')
    
    tvecs4 = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/left_camera_info/tvecs4.npy')
    tvecs3 = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/right_camera_info/tvecs3.npy')
    table, table_3d = triangulate(case4, case3, projL, projR)
    table_3d = table_3d[:, :2]
    for  i in range(4):
        
        point1 = table_3d[i]
        
        
        if i ==3 :
            point2 = table_3d[0]
        else:
            point2 = table_3d[i + 1]
        

        lines_table.append(sg.Segment2(sg.Point2(float(point1[0]), float(point1[1])), sg.Point2(float(point2[0]), float(point2[1]))))   
    while True:
        
        _, frameL = video_left.read()
        _, frameR = video_right.read()
        

        
        frameL = cv2.undistort(frameL, mtxL, distL)
        frameR = cv2.undistort(frameR, mtxR, distR)
        bbL, contoursL = change_det(substructionL, frameL, num = 4)
        bbR, contoursR = change_det(substructionR, frameR, num = 3)
        for  i in range(4):
            
            point1 = table[i][0]
            
            
            if i ==3 :
                point2 = table[0][0]
            else:
                point2 = table[i + 1][0]
            
            
            cv2.line(frameR, (point1[0], point1[1]), (point2[0], point2[1]), (0, 255, 0), thickness = 3)
        
        if not bbL or not bbR :
            print('adeia koutia')
            continue
    
        featuresL = encoderL(frameL, bbL)
        featuresR = encoderR(frameR, bbR)    
        frames = [frameR, frameL]
        detectionsL = [Detects(numL, boxl, mtxL, rot_mat4, projL, tvecs4, distL, frameL, 'L')
                       for numL, boxl in enumerate(bbL)]
        detectionsR = [Detects(numR, boxr, mtxR, rot_mat3, projR, tvecs3, distR, frameR, 'R') 
                       for numR, boxr in enumerate(bbR)]
        
        # [Detects.plot_2d(frames[1], detectionsL, 1)]
        # [Detects.plot_2d(frames[0], detectionsR, 0 )]
        
        if len(detectionsL) > 1:
            detectionsL_iou = [Detects.compute_iou(detectionsL[i], detectionsL[j]) 
                               for i in range(len(detectionsL)) for j in range(i+1, len(detectionsL))]
            if any(isinstance(el, list) for el in detectionsL_iou):
                detectionsL_iou = list(itertools.chain(*detectionsL_iou))
        if len(detectionsR) > 1:
            detectionsR_iou = [Detects.compute_iou(detectionsR[i], detectionsR[j])
                               for i in range(len(detectionsR)) for j in range(i+1, len(detectionsR))]
            if any(isinstance(el, list) for el in detectionsR_iou):
                detectionsR_iou = list(itertools.chain(*detectionsR_iou))
        
       
        if len(detectionsL) > 1 and len(detectionsR) > 1:
            create_3d_bb_contours.bb(frames, detectionsL_iou, detectionsR_iou, lines_table)
            
        elif len(detectionsL) > 1 and len(detectionsR) <= 1:
            create_3d_bb_contours.bb(frames, detectionsL_iou, detectionsR, lines_table)
            
        elif len(detectionsL) <= 1 and len(detectionsR) > 1:
            create_3d_bb_contours.bb(frames, detectionsL, detectionsR_iou, lines_table)
        elif len(detectionsL) <=1 and len(detectionsR) <= 1:
             create_3d_bb_contours.bb(frames, detectionsL, detectionsR, lines_table)
        else:
            print('tipota')
    
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    