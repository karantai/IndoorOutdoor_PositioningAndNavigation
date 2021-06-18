# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 20:00:48 2021

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 15:58:20 2021

@author: Admin
"""

import cv2
import numpy as np
import itertools

from XYZ_WFK import camera_pos, calcWFKfromPhotoMatrix, calcPhotogrammetryMatrixFromSolvePnp
import time
from collections import namedtuple
from bbox.metrics import jaccard_index_2d
from bbox import BBox2D, XYXY

width = 5.37 ## mm 
height = 4.04 ## mm

Xpixel= width/1920 
Ypixel= height/1080 

pixel = (Xpixel + Ypixel) / 2

class Detects:
    
    diction = {}
    
    def __init__(self, 
                 num, 
                 box_list, 
                 camera_matrix,
                 rotation_matrix,
                 projection_matrix,
                 tvecs,
                 dist,
                 frame,
                 camera):
        
        self.x = box_list[0]
        self.y = box_list[1]
        self.w = box_list[2]
        self.h = box_list[3]
        self.list = box_list
        self.dist = dist
        self.frame = frame
        self.id = num
        self.box = BBox2D(self.list)
        self.cam_id = camera
        # exterior openCV orientation
        self.rotation_matrix = rotation_matrix
        self.projection_matrix = projection_matrix
        self.tvecs = tvecs
        
        self.down_mid_x = (self.x + self.w / 2.0) 
        self.down_mid_y = (self.y + self.h) 
        
        self.camera_matrix = camera_matrix
        # interior orientation
        
        self.xo = (self.camera_matrix[0, 2] - 1920.0 / 2.0)
        self.yo = (-self.camera_matrix[1, 2] + 1080.0 / 2.0)
        
        self.f = ((self.camera_matrix[0, 0] + self.camera_matrix[1, 1]) / 2.0)
        
        
        # exterior photogrammetric orientation
        self.gXo, self.gYo, self.gZo = camera_pos(self.rotation_matrix, self.tvecs)
        
        # self.omega, self.phi, self.kappa = calcWFKfromPhotoMatrix(calcPhotogrammetryMatrixFromSolvePnp(self.rotation_matrix))
        
        self.photogram_rot_mat = calcPhotogrammetryMatrixFromSolvePnp(self.rotation_matrix)
        
        # ground coordinate system
        self.X = None
        self.Y = None
        
        self.down_mid_x_corrected = (self.down_mid_x - 1920.0 / 2.0)
        self.down_mid_y_corrected =  (1080.0 / 2.0 - self.down_mid_y )
        
        # print(self.down_mid_x_corrected, self.down_mid_y_corrected)
        
        # self.x_corrected = self.x - 1920 / 2
        # self.y_corrected = 1080/2 - self.y 
        
        
       
    def triangulation_z(self, z = 0.000):
        
        X = self.gXo + (z - self.gZo) * ((self.photogram_rot_mat[0, 0]*(self.down_mid_x_corrected - self.xo) + \
                                          self.photogram_rot_mat[1, 0]*(self.down_mid_y_corrected - self.yo) - \
                                          self.photogram_rot_mat[2, 0]*self.f) / (self.photogram_rot_mat[0, 2]*(self.down_mid_x_corrected - self.xo)+ \
                                                                                   self.photogram_rot_mat[1, 2]*(self.down_mid_y_corrected - self.yo)- \
                                                                                   self.photogram_rot_mat[2, 2]*self.f))
                                                                                   
        Y = self.gYo +    (z - self.gZo)* ((self.photogram_rot_mat[0, 1]*(self.down_mid_x_corrected - self.xo) + \
                                          self.photogram_rot_mat[1, 1]*(self.down_mid_y_corrected - self.yo) - \
                                          self.photogram_rot_mat[2, 1]*self.f) / (self.photogram_rot_mat[0, 2]*(self.down_mid_x_corrected - self.xo)+ \
                                                                                   self.photogram_rot_mat[1, 2]*(self.down_mid_y_corrected - self.yo)- \
                                                                                   self.photogram_rot_mat[2, 2]*self.f))                                                               
    
            
        self.X = X
        self.Y = Y 
        # print(self.X, self.Y)
        coordinates = np.float32(np.array([self.X, self.Y, z]))
        mid_down = np.int32(cv2.projectPoints(coordinates.reshape(1, -1), cv2.Rodrigues(self.rotation_matrix)[0], self.tvecs, self.camera_matrix, np.zeros((1,5)) )[0])     
        
    
        # cv2.circle(self.frame, (mid_down[0][0][0], mid_down[0][0][1]), 10, (0, 0, 0), -1)
        # print(mid_down)
        return(self)

        

    def correlation_distance(detection1, detection2):
        
        
        dist = np.sqrt((detection1.X - detection2.X)**2 + (detection1.Y - detection2.Y)**2)
        # print(dist)

        Detects.diction[detection1.id , detection2.id] = dist
        

        
        
                        
                        
    def compute_iou(detection1, detection2):
        
        x = jaccard_index_2d(detection1.box, detection2.box)
        
        if x > 0.0001:
            print(x)
            
            xmin = min(detection1.x, detection2.x)
            ymin = min(detection1.y, detection2.y)
            
            xmax = max(detection1.x + detection1.w, detection2.x + detection2.w)
            ymax = max(detection1.y + detection1.h, detection2.y + detection2.h)
            
            w = xmax - xmin
            h = ymax - ymin
                        
            if detection1.cam_id == 'L':
                det = Detects(detection1.id , [xmin, ymin, w, h], detection1.camera_matrix,
                        detection1.rotation_matrix, detection1.projection_matrix, detection1.tvecs, 
                        detection1.dist, detection1.frame, 'L')
            else:
                
                det = Detects(detection1.id , [xmin, ymin, w, h], detection1.camera_matrix,
                        detection1.rotation_matrix, detection1.projection_matrix, detection1.tvecs, 
                        detection1.dist, detection1.frame, 'R')
            
            
            return(det)
        
        else:
            
            return([detection1, detection2])
                        
                
                        
    def plot_2d(frame, detections, cam_num):
        # cv2.namedWindow(f'Camera{number}', cv2.WINDOW_NORMAL)
        
        for i in detections:
            cv2.rectangle(frame, (i.x, i.y), (i.x + i.w, i.y + i.h), (0,255,0),2)
            
            cv2.rectangle(frame,(i.x, i.y-45), (i.x + 100, i.y), (0, 255,0), -1)
            cv2.putText(frame, f'ID:{i.id}', (i.x, i.y-5), 1, 3, (0, 0, 255), 2)
            
            cv2.imshow(f'Camera{cam_num}', frame)
            if cv2.waitKey(1) & 0xFF==27:
                cv2.destroyAllWindows()
                break
            else:
                pass

                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
            



     