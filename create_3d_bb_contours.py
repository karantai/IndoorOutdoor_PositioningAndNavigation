#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 21:03:38 2020

@author: karas
"""



import cv2
import numpy as np
import operator
import time
from sympy.geometry import *
import skgeom as sg
from change_detection_boxes import Detects
from bbox.metrics import jaccard_index_2d
from bbox import BBox2D, XYXY


# out_left = cv2.VideoWriter('single_person_left.avi',cv2.VideoWriter_fourcc(*'mp4v'), 25, (1920, 1080))
# out_right = cv2.VideoWriter('single_person_right.avi',cv2.VideoWriter_fourcc(*'mp4v'), 25, (1920, 1080))

# projection_matrixR = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/right_camera_info/projection_matrixR.npy')
# projection_matrixL = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/left_camera_info/projection_matrixL.npy')

# rotationL = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/left_camera_info/rotation_matrixL.npy')
# rotationR = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/right_camera_info/rotation_matrixR.npy')

# mtxL = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/left_camera_info/left_camera.npy')
# mtxR = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/right_camera_info/right_camera.npy')

# tvecs4 = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/left_camera_info/tvecs4.npy')
# tvecs3 = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/right_camera_info/tvecs3.npy')


def triangulate_3d(L, R, 
                    left_up_projected_2D,
                    right_up_projected_2D,
                    three_d,
                   tvecs4 = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/left_camera_info/tvecs4.npy'),
                   rvecs4 = cv2.Rodrigues(np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/left_camera_info/rotation_matrixL.npy'))[0],
                   camera4 = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/left_camera_info/left_camera.npy'),
                   dist4 = np.zeros((1,5)),
                   
                    tvecs3 = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/right_camera_info/tvecs3.npy'),
                    rvecs3 = cv2.Rodrigues(np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/right_camera_info/rotation_matrixR.npy'))[0],
                    camera3 = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/right_camera_info/right_camera.npy'),
                    dist3 = np.zeros((1,5)),
                   
                   # dist = np.load('se_ligo4_distortion.npy'),
                   projL = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/left_camera_info/projection_matrixL.npy'),
                   projR = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/right_camera_info/projection_matrixR.npy')
                   ):
    

            for ll in L:
                for rr in R:
                    
                    ### left camera
                    
                    # 3D points up 
                    final_points = cv2.triangulatePoints(projL, projR, ll, rr)
                    final_points = final_points[:3, :] / final_points[3, :]

                    
                    t3Dpoints = np.around(final_points, 3).T
                    three_d.append(t3Dpoints)
                    left_2Dpoints_up = np.int32(cv2.projectPoints(t3Dpoints, rvecs4, tvecs4,  camera4, dist4)[0])
                    left_up_projected_2D.append(left_2Dpoints_up)
                    
                    right_2Dpoints_up = np.int32(cv2.projectPoints(t3Dpoints, rvecs3, tvecs3,  camera3, dist3)[0])
                    right_up_projected_2D.append(right_2Dpoints_up)

                    


            
  
def making_3d_bb(boxes_3d, frames, number, line, lines_table, detections, all_3d):

    
    for num, (one_3d, bbox_3d) in enumerate(zip(all_3d, boxes_3d )):


        for i in range(8):
        
            cv2.circle(frames[number], (bbox_3d[i][0], bbox_3d[i][1]), 6, (25,50,190), -1)
        

        for i in range(4):
                point_1_ = bbox_3d[2 * i]
                point_2_ = bbox_3d[2 * i + 1]
                cv2.line(frames[number], (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), (255, 0, 0), 2)
                
                point_1 = one_3d[2 * i]
                point_2 = one_3d[2 * i + 1]                
                line1 = sg.Segment2(sg.Point2(float(point_1[0]), float(point_1[1])), sg.Point2(float(point_2[0]), float(point_2[1])))
                line.append(line1)
        
        
        for i in range(4):
                point_1_ = bbox_3d[i]
                point_2_ = bbox_3d[i + 4]
                cv2.line(frames[number], (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), (255, 0, 0), 2)
                
                point_1 = one_3d[i]
                point_2 = one_3d[i+4]                 
                line2 =  sg.Segment2(sg.Point2(float(point_1[0]), float(point_1[1])), sg.Point2(float(point_2[0]), float(point_2[1])))
                line.append(line2)
                
        for i in range(2):
                point_1_ = bbox_3d[i]
                point_2_ = bbox_3d[i + 2]
                cv2.line(frames[number], (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), (255, 0, 0), 2) 
                  
                point_1 = one_3d[i]
                point_2 = one_3d[i+2] 
                line3 = sg.Segment2(sg.Point2(float(point_1[0]), float(point_1[1])), sg.Point2(float(point_2[0]), float(point_2[1])))
                line.append(line3)
                
        for i in range(4,6):
                point_1_ = bbox_3d[i]
                point_2_ = bbox_3d[i + 2]
                cv2.line(frames[number], (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), (255, 0, 0), 2)
                  
                point_1 = one_3d[i]
                point_2 = one_3d[i+2]                  
                line4 = sg.Segment2(sg.Point2(float(point_1[0]), float(point_1[1])), sg.Point2(float(point_2[0]), float(point_2[1])))
                line.append(line4)
        
        for i in range(8):
            point_1_ = bbox_3d[i]
            cv2.putText(frames[number], text = f'{i}',org = (point_1_[0],point_1_[1]), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (21,21,249), thickness = 2)
        
        img_inter = frames[number].copy()
        img_dinter = frames[number].copy() 
        cv2.namedWindow(f'Camera{number}', cv2.WINDOW_NORMAL)

        inters = [sg.intersection(i,j) for i in lines_table for j in line]
        
        if not any(inters):
            cv2.putText(img_dinter, 'LINES DO NOT INTERSECT', (30,70), cv2.FONT_HERSHEY_DUPLEX, 2, [255, 0, 0], 4)
            # cv2.imshow(f'Camera{number}', img_dinter)
            
            for i in detections:
                cv2.rectangle(img_dinter, (i.x, i.y), (i.x + i.w, i.y + i.h), (0,255,0),2)
                
                cv2.rectangle(img_dinter,(i.x, i.y-45), (i.x + 100, i.y), (0, 255,0), -1)
                cv2.putText(img_dinter, f'ID:{i.id}', (i.x, i.y-5), 1, 3, (0, 0, 255), 2)
                
            cv2.imshow(f'Camera{number}', img_dinter)
            # cv2.imwrite(f'Camera{number}.jpg', img_dinter)
            if cv2.waitKey(1) & 0xFF==27:
                cv2.destroyAllWindows()
                break
            else:
                pass

        else:
                cv2.putText(img_inter, 'LINES ARE BEING INTERSECTED', (30,70), cv2.FONT_HERSHEY_DUPLEX, 2, [0, 0, 255], 4)
                # cv2.imshow(f'Camera{number}', img_inter)
                
                for i in detections:
                    cv2.rectangle(img_inter, (i.x, i.y), (i.x + i.w, i.y + i.h), (0,255,0),2)
                    
                    cv2.rectangle(img_inter,(i.x, i.y-45), (i.x + 100, i.y), (0, 255,0), -1)
                    cv2.putText(img_inter, f'ID:{i.id}', (i.x, i.y-5), 1, 3, (0, 0, 255), 2)
                    
                cv2.imshow(f'Camera{number}',  img_inter)
                # cv2.imwrite(f'Camera{number}.jpg', img_inter)
                
                if cv2.waitKey(1) & 0xFF==27:
                    cv2.destroyAllWindows()
                    break
                else:
                    pass
        
        

    return(line, inters)



def bb(frames, 

       detectionsL,
       detectionsR,
       lines_table):

    
    right_boxes_3d = []
    left_boxes_3d = []
    alll = []   
    all_3d = []
    

    detectionsL = [Detects.triangulation_z(i)  for i in detectionsL]
    detectionsR = [Detects.triangulation_z(i)  for i in detectionsR]
    
    [Detects.correlation_distance(i, j) for i in detectionsL for j in detectionsR]
    
    diction = Detects.diction
    

    left_index = set(i[0] for i in diction.keys())


    
    keys_list = list(diction.keys())
    
    last_dict = {}
    for l_idx in left_index:
        inter_dict = {}
        for j in keys_list:
            if  l_idx == j[0]:
                inter_dict[j] = diction[j]
                
                
        
        x = min(inter_dict.items(), key = operator.itemgetter(1))[0]
        last_dict[x] = diction[x]
        

    





    # for key in last_dict.keys():
        


        
    for tL in detectionsL:
        
            for tR in detectionsR: 
                
                tup = (tL.id, tR.id)
                print(tup)
                if  tup in last_dict.keys():
                    
                        # left camera  - up left point
                        x1L = tL.x
                        y1L = tL.y
                        
                        # left camera  - up right point / y1     
                        x2L = tL.x + tL.w
                        # y2L = tL[keys[1]].to_tlwh()[1]
                        
                        # left camera  - down left point
                        x4L = tL.x
                        y4L = tL.y + tL.h 
                        
                        # left camera  - down right point
                        x3L = tL.x + tL.w
                        y3L = tL.y + tL.h
                        
                        
                        # right camera  - up left point
                        x1R = tR.x
                        y1R = tR.y
                        
                        # right camera  - up right point / y1     
                        x2R = tR.x + tR.w
                        # y2R = tR[keys[0]].to_tlwh().y
                        
                        # right camera  - down left point
                        x4R = tR.x
                        y4R = tR.y + tR.h 
                        
                        # right camera  - down right point
                        x3R = tR.x + tR.w
                        y3R = tR.y + tR.h
                        
                                            
                                            
                        Lup = np.array([x2L, y1L, x1L, y1L]).reshape(2,2).astype(np.float32)
                        Ldown = np.array([x3L, y3L, x4L, y4L]).reshape(2,2).astype(np.float32)

                        Rup = np.array([x2R, y1R, x1R, y1R]).reshape(2,2).astype(np.float32)
                        Rdown = np.array([x3R, y3R, x4R, y4R]).reshape(2,2).astype(np.float32)
    
                        left_up_projected_2D = []
                        left_down_projected_2D = []
                        
                        right_up_projected_2D = []
                        right_down_projected_2D = []
                        up_3d = []
                        down_3d = []
                        

                        triangulate_3d(Lup, Rup, left_up_projected_2D, right_up_projected_2D, up_3d)
                        triangulate_3d(Ldown, Rdown, left_down_projected_2D, right_down_projected_2D, down_3d)
                        
                            
                        left_unite = left_up_projected_2D + left_down_projected_2D
                        right_unite = right_up_projected_2D + right_down_projected_2D
                        threed = up_3d + down_3d
                        left_boxes_3d.append(np.array(left_unite).reshape(-1, 2))
                        right_boxes_3d.append(np.array(right_unite).reshape(-1, 2))
                        threed = np.concatenate(threed, axis = 0)
                        threed = threed[:, :2]
                        all_3d.append(threed)
                        alll = [right_boxes_3d, left_boxes_3d]
                        
                        alll = [right_boxes_3d, left_boxes_3d]
                        
                        lineL = []
                        lineR = []
                    
            
                    
                       
                        for num, i in enumerate(alll):
                            
                            if num == 0:
                                
                                linesR, intersR = making_3d_bb(i, frames, num, lineR, lines_table, detectionsR, all_3d)
                    
                            else:
                                
                                linesL, intersL = making_3d_bb(i, frames, num, lineL, lines_table, detectionsL, all_3d)
                        
                else:
                    print('MPIKE EDW')
                    continue
        

    
